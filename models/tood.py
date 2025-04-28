import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameter N = 6

class THead(nn.Module):
    def __init__(self, in_channels_list, N=6):
        super(THead, self).__init__()
        self.N = N
        self.convLists = nn.ModuleList()
        
        for in_channels in in_channels_list:
            convList = nn.ModuleList()
            for _ in range(N):
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU()
                )
                nn.init.kaiming_normal_(conv[0].weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(conv[0].bias, 0)
                convList.append(conv)
            self.convLists.append(convList)

    def forward(self, x, level_idx=None):
        # Handle both single tensor and list inputs
        if isinstance(x, (list, tuple)):
            return [self.forward_single_level(f, i) for i, f in enumerate(x)]
        else:
            if level_idx is None:
                raise ValueError("level_idx must be provided for single tensor input")
            return self.forward_single_level(x, level_idx)
            
    def forward_single_level(self, x, level_idx):
        xk_list = []
        convList = self.convLists[level_idx]
        for k in range(self.N):
            X_inter_k = convList[k](x if k == 0 else xk_list[-1])
            xk_list.append(X_inter_k)
        return xk_list
  
def align_b(B_box, O):
    B_batch, C, H, W = B_box.shape # (B, 4, H, W)
    device = B_box.device

    # Compute pixel grid - FIXED: Use device from input tensor
    yv, xv = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((xv, yv), dim=0).float() # (2, H, W)

    grid = grid.unsqueeze(0).unsqueeze(0).repeat(B_batch, 4, 1, 1, 1)  # (B, 4, 2, H, W)

    offsets = []
    for i in range(4):
        dx = O[:, 2*i, :, :]   # [B, H, W]
        dy = O[:, 2*i+1, :, :] # [B, H, W]
        offset_pair = torch.stack((dx, dy), dim=1)  # [B, 2, H, W]
        offsets.append(offset_pair)

    offset = torch.stack(offsets, dim=1)  # [B, 4, 2, H, W]

    # Compute sampling grid
    sampling_grid = grid + offset  # [B, 4, 2, H, W]

    # Normalize grid
    sampling_grid[:, :, 0, :, :] = 2.0 * sampling_grid[:, :, 0, :, :] / (W - 1) - 1.0
    sampling_grid[:, :, 1, :, :] = 2.0 * sampling_grid[:, :, 1, :, :] / (H - 1) - 1.0

    sampling_grid = sampling_grid.permute(0, 1, 3, 4, 2).reshape(B_batch * 4, H, W, 2)
    B_reshaped = B_box.reshape(B_batch * 4, 1, H, W)

    # Sample using bilinear interpolation
    B_aligned = F.grid_sample(B_reshaped, sampling_grid, mode='bilinear',
                              padding_mode='zeros', align_corners=True)

    B_aligned = B_aligned.view(B_batch, 4, H, W)

    return B_aligned

def distances_to_bboxes(distances, stride):
    B, C, H, W = distances.shape
    device = distances.device

    # FIXED: Create grid on the same device as input tensor
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    grid_x = grid_x.float() * stride  # [H, W]
    grid_y = grid_y.float() * stride

    # Expand to batch size
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    # Split distances: [B, 4, H, W] → l, t, r, b
    l = distances[:, 0]
    t = distances[:, 1]
    r = distances[:, 2]
    b = distances[:, 3]

    # Compute bbox corners
    x1 = grid_x - l * stride
    y1 = grid_y - t * stride
    x2 = grid_x + r * stride
    y2 = grid_y + b * stride

    # Stack to shape [B, 4, H, W]
    bboxes = torch.stack([x1, y1, x2, y2], dim=1)
    return bboxes

class TAPBlock_class(nn.Module):
  def __init__(self, c_in, N=6, num_classes=80):
    super(TAPBlock_class, self).__init__()

    self.ztask = nn.Sequential(
        nn.Conv2d(N * c_in, c_in, kernel_size=1), # 1x1 conv layer
        nn.ReLU(),
        nn.Conv2d(c_in, num_classes, kernel_size=3, padding=1)
    )

    self.mask = nn.Sequential(
        nn.Conv2d(N * c_in, c_in, kernel_size=1), # 1x1 conv layer
        nn.ReLU(),
        nn.Conv2d(c_in, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    ) # shape of (B, 1, H, W)

    for m in [self.ztask, self.mask]:
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def forward(self, x_inter, xtaskcat):
    m_align = self.mask(x_inter) # (B, 1, H, W)
    z = self.ztask(xtaskcat) # (B, 80, H, W)
    #print(f"Z: {z.shape}")

    # Dense classification (P) (B, 80, H, W)
    P = torch.sigmoid(z)
    p_align = torch.sqrt( P * m_align )
    #print(f"P: {P.shape} --- P_align: {p_align.shape}")
    #print("Done with Classification stuff\n\n")

    return p_align

class TAPBlock_loc(nn.Module):
  def __init__(self, c_in, N=6):
    super(TAPBlock_loc, self).__init__()
    self.register_buffer('stride', torch.tensor(1.))  # Will be set later

    self.ztask = nn.Sequential(
        nn.Conv2d(N * c_in, c_in, kernel_size=1), # 1x1 conv layer
        nn.ReLU(),
        nn.Conv2d(c_in, 4, kernel_size=3, padding=1)
    )

    self.offset = nn.Sequential(
        nn.Conv2d(N * c_in, c_in, kernel_size=1), #1x1 conv layer
        nn.ReLU(),
        nn.Conv2d(c_in, 8, kernel_size=3, padding=1)
    ) # shape of (B, 8, H, W)

    for m in [self.ztask, self.offset]:
      if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.01)  # Small std for box/offset predictions
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

  def forward(self, x_inter, xtaskcat):
    o_align = self.offset(x_inter) # (B, 8, H, W)
    z = self.ztask(xtaskcat) # (B, 4, H, W)
    #print(f"Z: {z.shape}")

    # Object bounding boxes (B) (B, 4, H, W)
    B = distances_to_bboxes(z, self.stride.item())
    b_align = align_b(B, o_align)
    #print(f"B: {B.shape} --- B_align: {b_align.shape}")
    #print("Done with Localization stuff\n\n")

    return b_align
  
def xyxy_to_xywh(b_align):
    # Convert to YOLO format of (x_center, y_center, width, height) from (x1, y1, x2, y2)
    x1, y1, x2, y2 = torch.unbind(b_align, dim=1)
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return torch.stack([x_center, y_center, width, height], dim=1)

class TOODHead(nn.Module):

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True, N=6):
        super(TOODHead, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2 if isinstance(anchors, (list, tuple)) else anchors  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output convs

        self.N = N
        in_channels = ch[0]

        # Bottom Branch
        self.w = nn.ModuleList()  # One weight branch per level
        for c in ch:
            self.w.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),  # [B, N*c, 1, 1]
                    nn.Flatten(),             # [B, N*c]
                    nn.Linear(c * N, c),     # Dynamic input size
                    nn.ReLU(),
                    nn.Linear(c, N),         # [B, N]
                    nn.Sigmoid()
                )
            )

        # Make all components multi-scale
        self.thead = THead(ch, N)
        self.cls_tap = nn.ModuleList([TAPBlock_class(c, N, nc) for c in ch])
        self.reg_tap = nn.ModuleList([TAPBlock_loc(c, N) for c in ch])

        self.obj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(N * c, c, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(c, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            ) for c in ch
        ])

        # Objectness score for YOLO
        self.obj_layer = nn.Sequential(
            nn.Conv2d(N * in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        # Initialize Weight FCs
        for m in self.w:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize objectness layer
        for layer in self.obj_layer:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # Set strides for localization branches
        if not self.training and isinstance(self.stride, torch.Tensor):
            for i in range(self.nl):
                self.reg_tap[i].stride = self.stride[i].to(self.reg_tap[i].stride.device)
        
        # Handle both list and tensor inputs
        if isinstance(x, (list, tuple)):
            # Process all pyramid levels
            outputs = [self.process_level(x[i], i) for i in range(self.nl)]
            
            if not self.training:
                # Inference: concatenate all predictions
                return torch.cat([out.view(out.shape[0], -1, self.no) for out in outputs], 1)
            else:
                # Training: return all levels separately
                return outputs
        else:
            # Single level input
            out = self.process_level(x, 0)
            return out.view(out.shape[0], -1, self.no) if not self.training else out
        
    def _make_grid(self, nx, ny, device):
        """Modified grid generation with device parameter"""
        yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing='ij')
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()
        
    def process_level(self, x, level_idx):
        xk_list = self.thead(x, level_idx)  # Note the level_idx parameter
        x_inter = torch.cat(xk_list, dim=1) # (B, N*C, H, W)

        weight = self.w[level_idx](x_inter)  # [B, N]
        batch = x_inter.size(0)
        xtask = [
            xk_list[k] * weight[:, k].view(batch, 1, 1, 1)
            for k in range(self.N)
        ]
        xtaskcat = torch.cat(xtask, dim=1)  # [B, N*C, H, W]

        # TAP Outputs
        p_align = self.cls_tap[level_idx](x_inter, xtaskcat)
        b_align = self.reg_tap[level_idx](x_inter, xtaskcat) 
        obj = self.obj_layers[level_idx](x_inter)

        # TOOD doesn't use anchors so just copy for anchor dim
        p_align = p_align.unsqueeze(1).expand(-1, self.na, -1, -1, -1)  # [B,na,80,H,W]
        b_align = b_align.unsqueeze(1).expand(-1, self.na, -1, -1, -1)   # [B,na,4,H,W]
        obj = obj.unsqueeze(1).expand(-1, self.na, -1, -1, -1)           # [B,na,1,H,W]

        yolo_output = torch.cat([b_align, obj, p_align], dim=2)  # [B, na, 5+nc, H, W]
        bs, _, _, ny, nx = yolo_output.shape
        yolo_output = yolo_output.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        if not self.training:
            device = yolo_output.device
            if self.grid[level_idx].shape[2:4] != yolo_output.shape[2:4] or self.grid[level_idx].device != device:
                self.grid[level_idx] = self._make_grid(nx, ny, device)  # Added device parameter
                self.anchor_grid[level_idx] = self.anchors[level_idx].to(device).view(1, -1, 1, 1, 2)

            y = yolo_output.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[level_idx]) * self.stride[level_idx]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[level_idx]  # wh
        
            # Return just the reshaped output for inference concatenation
            return y.view(y.shape[0], -1, self.no)

        return yolo_output