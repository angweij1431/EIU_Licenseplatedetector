import os
import argparse
from ultralytics import YOLO

# Standalone YOLO11 Training Script for Remote Dual-GPU / High-Performance Servers
def main():
    parser = argparse.ArgumentParser(description="Training YOLO11 on Vietnamese License Plate Dataset")
    parser.add_argument("--api-key", default=None, help="Roboflow API key to download dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (default: 32 for RTX 3080)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training (default: 640)")
    parser.add_argument("--device", default="0,1", help="GPU device IDs (e.g. '0,1' for dual 3080, or '0')")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model weights (default: yolo11n.pt)")
    args = parser.parse_args()

    # Step 1: Locate or download dataset
    data_yaml_path = None
    expected_local_dataset = "Vietnamese-Car-License-Plate--1/data.yaml"
    
    if os.path.exists(expected_local_dataset):
        print(f"Using existing local dataset: {expected_local_dataset}")
        data_yaml_path = expected_local_dataset
    elif args.api_key:
        print("Downloading dataset from Roboflow...")
        from roboflow import Roboflow
        rf = Roboflow(api_key=args.api_key)
        project = rf.workspace("alphajack2006-gmail-com").project("vietnamese-car-license-plate-vlbzz")
        dataset = project.version(1).download("yolov8")
        data_yaml_path = f"{dataset.location}/data.yaml"
    else:
        print("Notice: No local dataset found and no --api-key provided.")
        print("If you already downloaded the dataset, pass the path or place 'Vietnamese-Car-License-Plate--1' in this directory.")
        return

    # Step 2: Initialize YOLO11 base weights
    print(f"Loading base YOLO11 weights: {args.model}")
    model = YOLO(args.model)

    # Step 3: Launch multi-GPU / high-throughput training
    print(f"Starting training for {args.epochs} epochs on device [{args.device}] with batch size {args.batch}...")
    results = model.train(
        data=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=10,
        name="vietnam_plate_yolo11",
        save=True
    )

    print("\nTraining complete!")
    print("Exported weights location:")
    print("  runs/detect/vietnam_plate_yolo11/weights/best.pt")
    print("Copy this best.pt into your local Models_yolo/best.pt directory.")

if __name__ == "__main__":
    main()

