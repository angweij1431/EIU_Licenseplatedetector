"""
=====================================================================
EXAMPLE: FINE-TUNING YOLOV8 ON CUSTOM LICENSE PLATE DATASET
=====================================================================

This example demonstrates how to fine-tune a pre-trained YOLOv8 nano model 
(yolov8n.pt) on a custom dataset annotated for License Plates.

Dataset Directory Structure Required:
  dataset/
    ├── dataset.yaml        # Dataset configuration file
    ├── train/
    │   ├── images/         # Training images (.jpg / .png)
    │   └── labels/         # YOLO format label files (.txt)
    └── val/
        ├── images/         # Validation images
        └── labels/         # Validation labels

Example dataset.yaml content:
-----------------------------
path: ./dataset
train: train/images
val: val/images

names:
  0: license_plate
-----------------------------
"""

import os
from ultralytics import YOLO

def fine_tune_yolo_license_plate():
    # Step 1: Load pre-trained base model weights
    print("[1] Loading base model 'yolov8n.pt'...")
    model = YOLO("yolov8n.pt")

    # Step 2: Define dataset config path
    data_yaml_path = "dataset/dataset.yaml"

    # Verify if dataset config exists before training
    if not os.path.exists(data_yaml_path):
        print(f"  ❌ Config '{data_yaml_path}' not found.")
        print("     To run training, place your images and labels in 'dataset/' directory.")
        print("     Example dataset format: Roboflow or Kaggle Vietnamese License Plate dataset.")
        return

    # Step 3: Train / Fine-tune the model
    print("[2] Starting Fine-Tuning Training Pass...")
    results = model.train(
        data=data_yaml_path,
        epochs=50,             # Number of training epochs (e.g. 30-100)
        imgsz=640,             # Input image resolution
        batch=16,              # Batch size (adjust based on CPU/GPU RAM)
        name="vietnam_lpr",    # Training run output folder name
        project="runs/detect", # Project directory
        patience=10,           # Early stopping if no improvement for 10 epochs
        save=True,             # Save best.pt and last.pt weights
        verbose=True
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Fine-tuned best weights saved to: 'runs/detect/vietnam_lpr/weights/best.pt'")
    print("You can load 'best.pt' directly into YOLOv8Detector('runs/detect/vietnam_lpr/weights/best.pt')!")
    print("=" * 60)

if __name__ == "__main__":
    fine_tune_yolo_license_plate()

