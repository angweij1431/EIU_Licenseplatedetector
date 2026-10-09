import os
import sys
import argparse
import numpy as np
import cv2

# Import modular adapters from lpr_engine.py
from lpr_engine import (
    YOLOv8Detector, 
    PaddleOCRAdapter,
    EasyOCRAdapter,
    LPRPipeline
)

# Parse command line argument for any image file path at will
parser = argparse.ArgumentParser(description="Run LPR Pipeline Benchmark on Any Image")
parser.add_argument("image", nargs="?", default="car_test_2.jpg", help="Path to input vehicle image (default: car_test_2.jpg)")
args = parser.parse_args()

test_img_path = args.image

if not os.path.exists(test_img_path):
    print(f"❌ Error: Image '{test_img_path}' not found.")
    sys.exit(1)

print("=" * 60)
print(f"MODULAR LPR BENCHMARK RUNNER ON IMAGE: '{test_img_path}'")
print("Detector: YOLOv8 | OCR: PaddleOCR | Preprocessing: CLAHE+Bilateral")
print("=" * 60)

# Build Pipeline with PaddleOCR as default
detector = YOLOv8Detector("yolov8n.pt")
ocr = PaddleOCRAdapter(lang='en')

pipeline = LPRPipeline(detector=detector, ocr=ocr, enable_preprocessing=True)

# Process target image and save visualization output
summary = pipeline.process(test_img_path, save_annotated_name="output_annotated.jpg", primary_only=True)

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"  * Image Tested          : {test_img_path}")
print(f"  * Detector Model        : {summary['detector']}")
print(f"  * OCR Model             : {summary['ocr']}")
print(f"  * Detection Latency     : {summary['detection_latency_ms']:.2f} ms")
print(f"  * OCR Latency           : {summary['ocr_latency_ms']:.2f} ms")
print(f"  * Process RAM Usage     : {summary['ram_usage_mb']:.2f} MB")
print(f"  * Annotated BBox Output : {summary['annotated_path']}")
print("=" * 60)
