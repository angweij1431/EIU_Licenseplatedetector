"""
=================================================================================
GOOGLE COLAB TRAINING TEMPLATE: VIETNAM VEHICLE LICENSE PLATE DETECTOR
=================================================================================

IMPORTANT NOTE ON DATASETS:
---------------------------
❌ Standard COCO dataset (coco.yaml / coco8.yaml) DOES NOT CONTAIN LICENSE PLATES!
   COCO only contains 80 general categories ('car', 'bus', 'person', 'dog', etc.).

✅ YOU MUST USE A LICENSE PLATE DATASET (e.g. Roboflow Universe or Kaggle).
   Roboflow has free, pre-annotated Vietnamese License Plate datasets in YOLOv8 format.

Instructions for Google Colab:
1. Open Google Colab (https://colab.research.google.com/).
2. Change Runtime to GPU: 'Runtime' -> 'Change runtime type' -> Select 'T4 GPU'.
3. Run this script cells in order.
=================================================================================
"""

# Step 1: Install Ultralytics YOLO & Roboflow SDK in Colab
!pip install -q ultralytics roboflow

# ===============================================================================
# Step 2: Download a Real Vietnamese License Plate Dataset in YOLOv8 Format
# ===============================================================================
# Option A: Download from Roboflow Universe (Recommended & Free)
# 1. Sign up at https://universe.roboflow.com/
# 2. Search for 'Vietnamese License Plate' or 'Vietnam LPR'
# 3. Click 'Export Dataset' -> Select 'YOLOv8' format -> Copy code snippet below:

from roboflow import Roboflow

# Replace 'YOUR_API_KEY' with your free Roboflow API key if using Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("vietnam-lpr").project("vietnamese-license-plate")
version = project.version(1)
dataset = version.download("yolov8")

# Option B: Direct Zip URL Download (Alternative)
# !wget https://your-direct-dataset-link.com/vietnam_lpr.zip
# !unzip -q vietnam_lpr.zip -d dataset/

# ===============================================================================
# Step 3: Run YOLOv8 Training on GPU
# ===============================================================================
from ultralytics import YOLO

print("\n[1/3] Initializing YOLOv8 Nano base model on GPU...")
model = YOLO("yolov8n.pt") # Loads base pre-trained weights

print("[2/3] Fine-Tuning model on License Plate Dataset...")
# Point data= to the dataset.yaml file inside your downloaded dataset folder
results = model.train(
    data=f"{dataset.location}/data.yaml", # Path to downloaded dataset.yaml
    epochs=50,                             # 50 Epochs (~5-10 mins on T4 GPU)
    imgsz=640,                             # Image resolution
    batch=16,                              # Batch size
    name="vietnam_plate_lpr",              # Output folder
    save=True,                             # Save best.pt weights
    verbose=True
)

# ===============================================================================
# Step 4: Download Fine-Tuned 'best.pt' to Your Computer
# ===============================================================================
print("\n[3/3] Training Complete! Preparing 'best.pt' for download...")
best_weights_path = "runs/detect/vietnam_plate_lpr/weights/best.pt"

from google.colab import files
import os

if os.path.exists(best_weights_path):
    print(f"Downloading '{best_weights_path}' to your local computer...")
    files.download(best_weights_path)
else:
    print(f"Weights saved at: {best_weights_path}")
