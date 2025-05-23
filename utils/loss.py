# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441."""
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        # Ensure inputs are on the same device
        if pred.device != true.device:
            true = true.to(pred.device)
            
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        # Ensure inputs are on the same device
        if pred.device != true.device:
            true = true.to(pred.device)
            
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        # Ensure inputs are on the same device
        if pred.device != true.device:
            true = true.to(pred.device)
            
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss

def safe_iou(box1, box2):
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[..., 0] * wh[..., 1]
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]).clamp(min=1e-4)
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]).clamp(min=1e-4)
    
    return overlap / (area1[:, None] + area2[None, :] - overlap + 1e-7)

class TaskAlignedAssigner:
    def __init__(self, topk=13, alpha=1.0, beta=6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
    
    def assign(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes):
        device = pred_scores.device
        num_anchors = pred_scores.shape[0]
        num_gt = gt_boxes.shape[0]
        
        if num_gt == 0:
            return (torch.zeros_like(pred_scores),
                    torch.zeros_like(pred_boxes),
                    torch.zeros((num_anchors,), dtype=torch.long, device=device))
        
        # Safer value clamping
        pred_scores = torch.sigmoid(pred_scores).clamp(min=1e-4, max=1.0-1e-4)
        
        # More stable IoU calculation        
        iou = safe_iou(pred_boxes, gt_boxes).clamp(min=1e-4, max=1.0-1e-4)
        
        # More stable alignment metric calculation
        scores = pred_scores.gather(1, gt_labels.view(-1, 1).expand(-1, num_anchors).t())
        alignment_metric = torch.exp(self.alpha * torch.log(scores) + self.beta * torch.log(iou))

        # Initialize assignment tensors
        assigned_gt_idx = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        assigned_metrics = torch.zeros((num_anchors,), device=device)
        
        # For each GT, select top-k anchors using a safer approach
        for gt_idx in range(num_gt):
            # Use argsort and slice instead of topk for potentially more stability
            sorted_idx = torch.argsort(alignment_metric[gt_idx], descending=True)
            topk_idx = sorted_idx[:min(self.topk, num_anchors)]
            
            # Assign based on highest metric
            metric_gt = alignment_metric[gt_idx, topk_idx]
            is_better = metric_gt > assigned_metrics[topk_idx]
            assigned_gt_idx[topk_idx[is_better]] = gt_idx
            assigned_metrics[topk_idx[is_better]] = metric_gt[is_better]
        
        # Create mask for positive samples
        pos_mask = assigned_gt_idx >= 0
        
        # Create output tensors
        assigned_scores = torch.zeros_like(pred_scores)
        assigned_boxes = torch.zeros_like(pred_boxes)
        assigned_labels = torch.zeros((num_anchors,), dtype=torch.long, device=device)
        
        if pos_mask.sum() > 0:
            # Create one-hot labels for positive samples
            pos_gt_labels = gt_labels[assigned_gt_idx[pos_mask]]
            
            # Fill assigned scores with safer element-by-element approach
            for i, idx in enumerate(torch.where(pos_mask)[0]):
                if 0 <= pos_gt_labels[i] < assigned_scores.shape[1]:  # Bounds check
                    assigned_scores[idx, pos_gt_labels[i]] = 1.0
            
            # Assign boxes and labels safely
            assigned_boxes[pos_mask] = gt_boxes[assigned_gt_idx[pos_mask]]
            assigned_labels[pos_mask] = pos_gt_labels
        
        return assigned_scores, assigned_boxes, assigned_labels
    

class TaskAlignedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=6.0, eps=1e-7):
        super(TaskAlignedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, pred_scores, pred_boxes, assigned_scores, assigned_boxes):
        device = pred_scores.device
        
        # Early return checks
        if pred_scores.numel() == 0 or assigned_scores.numel() == 0:
            return (torch.tensor(0.0, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device, requires_grad=True))
        
        pos_mask = assigned_scores.sum(dim=1) > 0
        num_pos = pos_mask.sum().item()
        
        if num_pos == 0:
            return (torch.tensor(0.0, device=device, requires_grad=True),
                    torch.tensor(0.0, device=device, requires_grad=True))
        
        # Safer classification loss
        pred_scores = pred_scores.clamp(min=-10, max=10)  # Prevent extreme logits
        cls_loss = self.bce(pred_scores, assigned_scores)
        
        # More stable alignment metric calculation
        with torch.no_grad():
            assigned_labels = assigned_scores.argmax(dim=1)
            pos_scores = torch.sigmoid(pred_scores[pos_mask].gather(1, assigned_labels[pos_mask].unsqueeze(1)))
            pos_scores = pos_scores.clamp(min=1e-4, max=1.0-1e-4)
            
            # Use safe_iou from above
            pos_pred_boxes = pred_boxes[pos_mask]
            pos_assigned_boxes = assigned_boxes[pos_mask]
            iou = safe_iou(pos_pred_boxes, pos_assigned_boxes).clamp(min=1e-4, max=1.0-1e-4)
            
            alignment_metric = torch.exp(self.alpha * torch.log(pos_scores) + self.beta * torch.log(iou))
            alignment_metric = alignment_metric / (alignment_metric.sum() + 1e-7) * num_pos
        
        # Calculate losses
        pos_cls_loss = cls_loss[pos_mask].sum(dim=1)
        cls_loss = (pos_cls_loss * alignment_metric.squeeze(1)).sum() / (alignment_metric.sum() + 1e-7)
        
        reg_loss = self.smooth_l1(pos_pred_boxes, pos_assigned_boxes).sum(dim=1)
        reg_loss = (reg_loss * alignment_metric.squeeze(1)).sum() / (alignment_metric.sum() + 1e-7)
        
        return cls_loss, reg_loss

class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        
        # Initialize TaskAlignedAssigner and TaskAlignedLoss if using TOOD head
        self.use_tal = hasattr(model, 'head_type') and model.head_type == 'TOODHead'
        if self.use_tal:
            self.tal_assigner = TaskAlignedAssigner(
                topk=h.get('tal_topk', 13),
                alpha=h.get('tal_alpha', 1.0),
                beta=h.get('tal_beta', 6.0)
            )
            self.tal_loss = TaskAlignedLoss(
                alpha=h.get('tal_alpha', 1.0),
                beta=h.get('tal_beta', 6.0)
            )

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        # Ensure targets are on the correct device
        if targets.device != self.device:
            targets = targets.to(self.device)
            
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        
        # If using TAL for TOOD head
        if self.use_tal:
            return self._compute_tal_loss(p, targets)
        
        # Standard YOLOv5 loss computation
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                try:
                    pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
                except IndexError as e:
                    print(f"IndexError in loss computation: {e}")
                    print(f"Shape info - pi: {pi.shape}, b: {b.shape}, a: {a.shape}, gj: {gj.shape}, gi: {gi.shape}")
                    # Continue with next iteration if there's an error
                    continue

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                
                # Safely set values in tobj
                for idx in range(len(b)):
                    if 0 <= b[idx] < tobj.shape[0] and 0 <= a[idx] < tobj.shape[1] and \
                       0 <= gj[idx] < tobj.shape[2] and 0 <= gi[idx] < tobj.shape[3]:
                        tobj[b[idx], a[idx], gj[idx], gi[idx]] = iou[idx]

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    
    def _compute_tal_loss(self, p, targets):
        device = self.device
        lcls = torch.zeros(1, device=device)  # class loss
        lbox = torch.zeros(1, device=device)  # box loss
        lobj = torch.zeros(1, device=device)  # object loss
        
        # Ensure targets are on the correct device
        if targets.device != device:
            targets = targets.to(device)
        
        # Get image, class, box from targets - with try-except for safety
        try:
            tcls, tbox, indices, anch = self.build_targets(p, targets)  # targets
        except Exception as e:
            print(f"Error in build_targets: {e}")
            bs = len(targets)  # batch size
            return torch.tensor(0., device=device, requires_grad=True) * bs, torch.zeros(3, device=device)
        
        # Process each FPN layer
        for i, pi in enumerate(p):  # layer index, layer predictions
            try:
                b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
                
                # Extract predictions
                n = b.shape[0]  # number of targets
                if n == 0:
                    continue
                    
                # Get predictions for positive samples
                pxy, pwh, pobj, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)
                
                # Convert to absolute box coordinates
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box (center_x, center_y, width, height)
                
                # Convert to xyxy format for IoU calculation
                pred_boxes_xyxy = self._box_cxcywh_to_xyxy(pbox)
                gt_boxes_xyxy = self._box_cxcywh_to_xyxy(tbox[i])
                
                # Get anchor points (grid centers)
                anchor_points = torch.stack([gi.float(), gj.float()], dim=1).to(device)
                
                # Use TaskAlignedAssigner with try-except for robustness
                try:
                    assigned_scores, assigned_boxes, assigned_labels = self.tal_assigner.assign(
                        pcls, pred_boxes_xyxy, anchor_points, tcls[i], gt_boxes_xyxy
                    )
                    
                    # Calculate losses using TaskAlignedLoss
                    cls_loss, reg_loss = self.tal_loss(pcls, pred_boxes_xyxy, assigned_scores, assigned_boxes)
                    
                    # Add to total losses - check for NaN values
                    if not torch.isnan(cls_loss) and not torch.isinf(cls_loss):
                        lcls += cls_loss
                    if not torch.isnan(reg_loss) and not torch.isinf(reg_loss):
                        lbox += reg_loss
                    
                    # Calculate objectness loss - use IoU as target
                    tobj = torch.zeros_like(pi[..., 0])  # target obj
                    
                    # Safe IoU calculation
                    iou = bbox_iou(pred_boxes_xyxy, gt_boxes_xyxy, CIoU=True).detach().clamp(0, 1)
                    
                    # Safely set values in tobj
                    for idx in range(len(b)):
                        if 0 <= b[idx] < tobj.shape[0] and 0 <= a[idx] < tobj.shape[1] and \
                        0 <= gj[idx] < tobj.shape[2] and 0 <= gi[idx] < tobj.shape[3]:
                            tobj[b[idx], a[idx], gj[idx], gi[idx]] = iou[idx]
                            
                    obj_loss = self.BCEobj(pi[..., 4], tobj) * self.balance[i]
                    if not torch.isnan(obj_loss) and not torch.isinf(obj_loss):
                        lobj += obj_loss
                        
                except Exception as e:
                    print(f"Error in TAL assignment or loss calculation: {e}")
                    # Continue with next iteration if assignment fails
                    continue
                    
            except (IndexError, RuntimeError) as e:
                print(f"Error in _compute_tal_loss: {e}")
                # Continue with next iteration if there's an error
                continue
            
        # Apply loss weights
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        
        # Check for NaN values in final losses
        lbox = torch.nan_to_num(lbox, nan=0.0, posinf=1.0, neginf=0.0)
        lobj = torch.nan_to_num(lobj, nan=0.0, posinf=1.0, neginf=0.0)
        lcls = torch.nan_to_num(lcls, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Return total loss and component losses
        bs = len(targets)  # batch size
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls)).detach()
    
    def _box_cxcywh_to_xyxy(self, x):
        """Convert boxes from center-width-height format to top-left/bottom-right coordinates."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        device = self.device
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        
        # Handle empty targets case
        if nt == 0:
            return [torch.zeros(0, device=device) for _ in range(4)]
        
        # Ensure targets are on correct device
        targets = targets.to(device)
        
        try:
            ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
            
            # Memory efficient approach to avoid OOM
            # Split targets processing if too large
            chunk_size = 1000  # Adjust based on available memory
            if nt * na > chunk_size:
                tcls_temp, tbox_temp, indices_temp, anch_temp = [], [], [], []
                
                for start_idx in range(0, nt, chunk_size // na):
                    end_idx = min(start_idx + chunk_size // na, nt)
                    chunk_targets = targets[start_idx:end_idx]
                    chunk_nt = end_idx - start_idx
                    chunk_ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, chunk_nt)
                    chunk_targets = torch.cat((chunk_targets.repeat(na, 1, 1), chunk_ai[..., None]), 2)
                    
                    # Process this chunk
                    c_tcls, c_tbox, c_indices, c_anch = self._process_targets_chunk(p, chunk_targets)
                    
                    # Append results
                    tcls_temp.extend(c_tcls)
                    tbox_temp.extend(c_tbox)
                    indices_temp.extend(c_indices)
                    anch_temp.extend(c_anch)
                    
                return tcls_temp, tbox_temp, indices_temp, anch_temp
            else:
                # Standard processing for smaller batches
                targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
                return self._process_targets_chunk(p, targets)
                
        except RuntimeError as e:
            print(f"Runtime error in build_targets: {e}")
            # Return empty lists as fallback
            return [torch.zeros(0, device=device) for _ in range(4)]
    
    def _process_targets_chunk(self, p, targets):
        """Process a chunk of targets to build target tensors."""
        device = self.device
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # normalized to gridspace gain
        
        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if t.shape[0]:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Boundary checking for indices
            valid_indices = (
                (b >= 0) & (b < shape[0]) &
                (gi >= 0) & (gi < shape[3]) &
                (gj >= 0) & (gj < shape[2])
            )
            
            if not valid_indices.all():
                # Filter out invalid indices
                b = b[valid_indices]
                a = a[valid_indices]
                gj = gj[valid_indices]
                gi = gi[valid_indices]
                c = c[valid_indices]
                gxy = gxy[valid_indices]
                gwh = gwh[valid_indices]
                
            if len(b) == 0:  # No valid indices
                tcls.append(torch.zeros(0, dtype=torch.long, device=device))
                tbox.append(torch.zeros(0, 4, device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device),
                               torch.zeros(0, dtype=torch.long, device=device),
                               torch.zeros(0, dtype=torch.long, device=device),
                               torch.zeros(0, dtype=torch.long, device=device)))
                anch.append(torch.zeros(0, 2, device=device))
                continue

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch