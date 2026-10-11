import time
import os
import sys
import psutil

# Ensure stdout handles UTF-8 on Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 60)
print("BENCHMARK & COMPATIBILITY TEST: PyTorch (YOLO) + PaddlePaddle (PaddleOCR)")
print(f"Python Version: {sys.version.split()[0]}")
print("=" * 60)

start_time = time.time()
mem_initial = get_process_memory_mb()
print(f"[0.00s] Initial RAM Usage: {mem_initial:.2f} MB")

# --- Step 1: Import Libraries ---
# IMPORTANT: On Windows, PyTorch MUST be imported before PaddlePaddle to prevent C++ DLL order conflicts
print("\n[Step 1] Importing PyTorch & PaddlePaddle (PyTorch first for DLL safety)...")
import_start = time.time()

try:
    import torch
    print(f"  - PyTorch version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
except Exception as e:
    print(f"  [ERROR] Failed to import PyTorch: {e}")
    sys.exit(1)

try:
    import paddle
    print(f"  - PaddlePaddle version: {paddle.__version__} (Device: {paddle.get_device()})")
except Exception as e:
    print(f"  [ERROR] Failed to import PaddlePaddle: {e}")
    sys.exit(1)

import_time = time.time() - import_start
mem_after_import = get_process_memory_mb()
print("  [OK] PyTorch and PaddlePaddle loaded simultaneously without crashing!")
print(f"  [TIME] Import Time: {import_time:.3f}s | RAM: {mem_after_import:.2f} MB (+{mem_after_import - mem_initial:.2f} MB)")

# --- Step 2: Import Ultralytics YOLO & PaddleOCR ---
print("\n[Step 2] Importing Ultralytics YOLO and PaddleOCR...")
pkg_import_start = time.time()

try:
    from ultralytics import YOLO
    print("  - Ultralytics YOLO imported successfully")
except Exception as e:
    print(f"  [ERROR] Failed to import YOLO: {e}")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
    print("  - PaddleOCR imported successfully")
except Exception as e:
    print(f"  [ERROR] Failed to import PaddleOCR: {e}")
    sys.exit(1)

pkg_import_time = time.time() - pkg_import_start
print(f"  [TIME] Package Import Time: {pkg_import_time:.3f}s")

# --- Step 3: Create Dummy Synthetic Image with Text ---
print("\n[Step 3] Creating dummy test image (simulated VN License Plate)...")
import numpy as np
import cv2

# Create a 400x300 canvas with simulated vehicle and license plate box
img = np.ones((300, 400, 3), dtype=np.uint8) * 200 # Grey background
cv2.rectangle(img, (50, 50), (350, 250), (50, 50, 50), -1) # Vehicle body
cv2.rectangle(img, (120, 140), (280, 200), (255, 255, 255), -1) # White plate
cv2.rectangle(img, (120, 140), (280, 200), (0, 0, 0), 2) # Black border

# Draw simulated license plate text "51G-123.45"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "51G-123.45", (130, 180), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

dummy_img_path = "dummy_test_image.jpg"
cv2.imwrite(dummy_img_path, img)
print(f"  [OK] Saved dummy image to '{dummy_img_path}'")

# --- Step 4: Initialize Models ---
print("\n[Step 4] Initializing YOLOv8 nano and PaddleOCR models...")

yolo_load_start = time.time()
try:
    yolo_model = YOLO("yolov8n.pt") # Downloads lightweight nano model (~6MB) if needed
    yolo_load_time = time.time() - yolo_load_start
    print(f"  [OK] YOLOv8n initialized in {yolo_load_time:.3f}s")
except Exception as e:
    print(f"  [ERROR] YOLO model initialization failed: {e}")
    sys.exit(1)

ocr_load_start = time.time()
try:
    # enable_mkldnn=False avoids Windows oneDNN PIR instruction bug on CPU
    ocr_model = PaddleOCR(enable_mkldnn=False)
    ocr_load_time = time.time() - ocr_load_start
    print(f"  [OK] PaddleOCR initialized in {ocr_load_time:.3f}s")
except Exception as e:
    print(f"  [ERROR] PaddleOCR initialization failed: {e}")
    sys.exit(1)

mem_after_model_load = get_process_memory_mb()
print(f"  [TIME] Total Models Load RAM: {mem_after_model_load:.2f} MB (+{mem_after_model_load - mem_after_import:.2f} MB)")

# --- Step 5: Inference - YOLO ---
print("\n[Step 5] Running YOLO Inference on dummy image...")
yolo_inf_start = time.time()
yolo_results = yolo_model(dummy_img_path, verbose=False)
yolo_inf_time = time.time() - yolo_inf_start

print(f"  [TIME] YOLO Inference Time: {yolo_inf_time*1000:.2f} ms")
for r in yolo_results:
    boxes = r.boxes
    print(f"  - Detected {len(boxes)} object(s):")
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        print(f"    * Class: '{cls_name}' | Confidence: {conf:.4f} | Box: {[round(x,1) for x in xyxy]}")

# --- Step 6: Inference - PaddleOCR ---
print("\n[Step 6] Running PaddleOCR Inference on dummy image...")
ocr_inf_start = time.time()
ocr_predict_res = list(ocr_model.predict(dummy_img_path))
ocr_inf_time = time.time() - ocr_inf_start

print(f"  [TIME] PaddleOCR Inference Time: {ocr_inf_time*1000:.2f} ms")
print("  - Detected Text & Confidence Scores:")
found_text = False
for item in ocr_predict_res:
    if isinstance(item, dict):
        rec_texts = item.get('rec_texts', [])
        rec_scores = item.get('rec_scores', [])
        for txt, score in zip(rec_texts, rec_scores):
            print(f"    * Recognized Text: '{txt}' | Confidence: {score:.4f} ({score*100:.2f}%)")
            found_text = True
    elif isinstance(item, list):
        for line in item:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                text, conf = line[1][0], line[1][1]
                print(f"    * Recognized Text: '{text}' | Confidence: {conf:.4f} ({conf*100:.2f}%)")
                found_text = True

if not found_text:
    print("    * No text recognized on dummy image.")

# --- Summary ---
total_time = time.time() - start_time
mem_final = get_process_memory_mb()

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY & REPORT")
print("=" * 60)
print(f"  * PyTorch + PaddlePaddle Coexistence : SUCCESS (No segfaults/conflicts)")
print(f"  * Total Script Execution Time       : {total_time:.2f}s")
print(f"  * Final RAM Usage (Both Models)     : {mem_final:.2f} MB")
print(f"  * YOLOv8 Inference Latency (CPU)    : {yolo_inf_time*1000:.2f} ms")
print(f"  * PaddleOCR Inference Latency (CPU) : {ocr_inf_time*1000:.2f} ms")
print("=" * 60)
