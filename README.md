CE597 - Final Project

Task-Aligned One-Stage Object Detector Re-Implementation into the YOLOv5 framework.

This repository is forked from the official YOLOv5 github, and changes were made with
- models/tood.py (my tood head structure)
- utils/loss.py (attempt at TAL loss function)
- debug.py (verify YOLOv5's Detect head module shape and my TOODHead shape)
- models/yolov5s.yaml (swapped out Detect head with TOODHead)
- models/yolov5 (included code to setup TOOD in parse_model)
- train.py / val.py (account for TOOD head)

## Installation

1. Clone this repository:

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Train
!python train.py --data coco.yaml --cfg yolov5s.yaml --weights yolov5s.pt --batch-size 16 --epochs 50

Validation
!python data/scripts/get_coco.py --val (download MS COCO 2017 validation dataset)
!python val.py --weights yolov5s.pt --data data/coco.yaml --img 640 --task val
!python models/yolo.py --cfg models/yolov5s.yaml (to output some additional metrics such as FLOPs)
