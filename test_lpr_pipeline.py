import time
import os
import sys
import psutil
import cv2
import numpy as np

# Ensure UTF-8 stdout
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 60)
print("VIETNAM VEHICLE LPR 2-STAGE PIPELINE TEST")
print("Stage 1: YOLO Object Detection (Vehicle / Plate ROI)")
print("Stage 2: PaddleOCR Text Recognition")
print("=" * 60)

# Import models
import torch
import paddle
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Load models
yolo_model = YOLO("yolov8n.pt")
ocr_model = PaddleOCR(enable_mkldnn=False)

# Create a more realistic synthetic vehicle + license plate image
# Vehicle body (Blue SUV)
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:] = (240, 240, 240) # White background

# Car body
cv2.rectangle(img, (100, 100), (500, 320), (180, 50, 50), -1) # Blue vehicle
cv2.rectangle(img, (150, 120), (450, 200), (220, 220, 220), -1) # Windshield/windows
cv2.circle(img, (180, 320), 40, (30, 30, 30), -1) # Wheels
cv2.circle(img, (420, 320), 40, (30, 30, 30), -1) # Wheels

# License Plate location: (x1=230, y1=240, x2=370, y2=290)
cv2.rectangle(img, (230, 240), (370, 290), (255, 255, 255), -1) # White plate background
cv2.rectangle(img, (230, 240), (370, 290), (0, 0, 0), 3) # Black border

# Vietnamese License Plate Text "30A-888.88"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "30A-888.88", (240, 275), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

sample_img_path = "sample_car.jpg"
cv2.imwrite(sample_img_path, img)
print(f"\n[1] Generated sample car image: '{sample_img_path}'")

# --- Step 1: Run YOLO for Vehicle Detection ---
print("\n[2] Running YOLO Vehicle Detection...")
yolo_results = yolo_model(sample_img_path, verbose=False)

for r in yolo_results:
    boxes = r.boxes
    print(f"    - YOLO detected {len(boxes)} object(s):")
    for b in boxes:
        cls_id = int(b.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf = float(b.conf[0])
        box_coords = [round(x, 1) for x in b.xyxy[0].tolist()]
        print(f"      * Detected Class: '{cls_name}' | Confidence: {conf:.2f} | BBox: {box_coords}")

# --- Step 2: Crop License Plate ROI & Run PaddleOCR ---
print("\n[3] Running PaddleOCR on the License Plate Region...")
# Crop the plate ROI (230, 240, 370, 290)
plate_crop = img[240:290, 230:370]
plate_crop_path = "cropped_license_plate.jpg"
cv2.imwrite(plate_crop_path, plate_crop)

ocr_res = list(ocr_model.predict(plate_crop_path))

print(f"    - OCR Recognition Results on Cropped License Plate:")
for item in ocr_res:
    if isinstance(item, dict):
        texts = item.get('rec_texts', [])
        scores = item.get('rec_scores', [])
        for txt, score in zip(texts, scores):
            print(f"      * RECOGNIZED LICENSE PLATE: '{txt}' | Confidence: {score*100:.2f}%")

print("\n" + "=" * 60)

