import torch
from models.yolo import Detect
from models.tood import TOODHead

def test_detect_head():
    # YOLOv5 configuration
    nc = 80  # COCO classes
    anchors = [
        [10,13, 16,30, 33,23],    # P3/8
        [30,61, 62,45, 59,119],   # P4/16
        [116,90, 156,198, 373,326] # P5/32
    ]
    ch = [256, 512, 1024]  # Input channels
    
    # Initialize Detect head
    detect = Detect(nc=nc, anchors=anchors, ch=ch)
    
    # Set stride (critical for inference)
    detect.stride = torch.tensor([8., 16., 32.])  # Must match feature map scales
    
    # Create FRESH input tensors for each mode
    batch_size = 4
    train_input = [
        torch.randn(batch_size, ch[0], 80, 80),  # P3
        torch.randn(batch_size, ch[1], 40, 40),  # P4
        torch.randn(batch_size, ch[2], 20, 20)   # P5
    ]
    
    # 1. Test in training mode
    print("Testing in TRAINING mode:")
    detect.train()
    with torch.no_grad():
        train_output = detect(train_input)
        print("\nTraining Output Shapes:")
        for i, out in enumerate(train_output):
            print(f"Layer {i}: {out.shape}")  # Should be [4, 3, grid_h, grid_w, 85]
    
    # 2. Test in inference mode (with NEW inputs)
    print("\nTesting in INFERENCE mode:")
    detect.eval()
    # Create fresh inputs (don't reuse training outputs)
    infer_input = [
        torch.randn(batch_size, ch[0], 80, 80),
        torch.randn(batch_size, ch[1], 40, 40),
        torch.randn(batch_size, ch[2], 20, 20)
    ]
    with torch.no_grad():
        infer_output = detect(infer_input)
    
    print("\nInference Output:")
    if isinstance(infer_output, tuple):
        print(f"Predictions shape: {infer_output[0].shape}")  # [4, num_detections, 85]
        if len(infer_output) > 1:
            print(f"Feature maps: {[x.shape for x in infer_output[1]]}")
    else:
        print(f"Output shape: {infer_output.shape}")

def test_tood_head():
    # YOLOv5 configuration
    nc = 80  # COCO classes
    anchors = [
        [10,13, 16,30, 33,23],    # P3/8
        [30,61, 62,45, 59,119],    # P4/16
        [116,90, 156,198, 373,326] # P5/32
    ]
    ch = [256, 512, 1024]  # Input channels
    
    # Initialize TOOD head
    tood = TOODHead(nc=nc, anchors=anchors, ch=ch)
    tood.stride = torch.tensor([8., 16., 32.])  # Set stride like YOLOv5
    
    # Create test input (batch_size=4, 640x640 img)
    # Note: Use raw feature maps, not processed outputs
    x = [
        torch.randn(4, ch[0], 80, 80),  # P3
        torch.randn(4, ch[1], 40, 40),  # P4
        torch.randn(4, ch[2], 20, 20)   # P5
    ]
    
    # Test in training mode first
    print("Testing in TRAINING mode:")
    tood.train()
    with torch.no_grad():
        train_out = tood(x)
        print("\nTraining Output Shapes:")
        if isinstance(train_out, (list, tuple)):
            for i, out in enumerate(train_out):
                print(f"Layer {i}: {out.shape}")  # Should be [4,3,80,80,85]
        else:
            print(f"Output: {train_out.shape}")
    
    # Test in inference mode
    print("\nTesting in INFERENCE mode:")
    tood.eval()
    with torch.no_grad():
        infer_out = tood(x)
    
    print("\nInference Output:")
    if isinstance(infer_out, tuple):
        print(f"Predictions shape: {infer_out[0].shape}")  # [4,num_detections,85]
        if len(infer_out) > 1:
            print(f"Feature maps: {[x.shape for x in infer_out[1]]}")
    else:
        print(f"Output: {infer_out.shape}")

    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    print("Testing one pass of the original YOLOv5 Head")
    test_detect_head()
    print("Testing one pass of our TOOD Head")
    test_tood_head()