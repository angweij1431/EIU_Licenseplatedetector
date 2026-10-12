import os
import sys
import time
import argparse
import glob
from datetime import datetime

# Import modular pipeline adapters from lpr_engine
from lpr_engine import (
    YOLODetector, 
    PaddleOCRAdapter, 
    LPRPipeline
)
from lpr_core.validator import VietnamPlateValidator

# Parse command line argument for vehicle test image or directory path
parser = argparse.ArgumentParser(description="Run LPR Pipeline Benchmark on an Image or Directory of Images")
parser.add_argument("input_path", nargs="?", default="test_images/car_test_3.jpg", help="Path to input image or directory (default: test_images/car_test_3.jpg)")
args = parser.parse_args()

target_path = args.input_path
if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target path not found: {target_path}")

# Initialize primary plate detector (fine-tuned YOLO11 model)
model_path = "Models_yolo/best.pt" if os.path.exists("Models_yolo/best.pt") else "Models_yolo/yolov8n.pt"
detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.25)
ocr = PaddleOCRAdapter(lang='en')
pipeline = LPRPipeline(detector=detector, ocr=ocr, enable_preprocessing=True)

# Determine whether input is a single file or a directory
if os.path.isdir(target_path):
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(target_path, ext)))
        image_files.extend(glob.glob(os.path.join(target_path, ext.upper())))
    image_files = sorted(list(set(image_files)))
    if not image_files:
        print(f"No image files found in directory: {target_path}")
        sys.exit(0)
    print(f"Found {len(image_files)} test images in directory: {target_path}")
else:
    image_files = [target_path]

# Process images and collect results
results_list = []
for idx, img_path in enumerate(image_files):
    print(f"Processing [{idx+1}/{len(image_files)}]: {img_path}")
    annotated_name = f"output_{os.path.basename(img_path)}"
    summary = pipeline.process(
        img_path, 
        save_annotated_name=annotated_name, 
        sort_by='front', 
        primary_only=True, 
        padding_percent=0.20
    )
    
    extracted_text = "NONE"
    is_valid = False
    raw_lines = []
    if summary['ocr_results']:
        raw_lines = [r['text'] for r in summary['ocr_results']]
        val_lines = [r['validated_text'] for r in summary['ocr_results']]
        joined_text = "-".join(val_lines) if len(val_lines) > 1 else val_lines[0]
        final_val = VietnamPlateValidator.clean_and_correct(joined_text)
        extracted_text = final_val['text']
        is_valid = final_val['is_valid']
        
    results_list.append({
        'image': os.path.basename(img_path),
        'try_level': summary.get('try_level', '1st try'),
        'resolution': summary.get('resolution', '640px'),
        'boxes': len(summary['detections']),
        'plate': extracted_text,
        'valid': is_valid,
        'det_ms': summary['detection_latency_ms'],
        'ocr_ms': summary['ocr_latency_ms'],
        'raw': raw_lines
    })

# Print aggregate summary report
print("\nBenchmark Summary Report:")
print(f"{'Image':<18} | {'Try Level':<10} | {'Resolution':<10} | {'Boxes':<5} | {'Plate':<12} | {'Valid':<6} | {'Det (ms)':<8} | {'OCR (ms)':<8}")
for r in results_list:
    valid_str = "YES" if r['valid'] else "NO"
    print(f"{r['image']:<18} | {r['try_level']:<10} | {r['resolution']:<10} | {r['boxes']:<5} | {r['plate']:<12} | {valid_str:<6} | {r['det_ms']:<8.1f} | {r['ocr_ms']:<8.1f}")

# Dynamically record latest run into BENCHMARK_SUMMARY.md
summary_md_path = "BENCHMARK_SUMMARY.md"
if os.path.exists(summary_md_path):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    history_header = "\n## 7. Dynamic Evaluation History\n\n| Timestamp | Test Image | Try Level | Resolution | Boxes | Plate Text | Valid Format | Det (ms) | OCR (ms) |\n| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |\n"
    if "## 7. Dynamic Evaluation History" not in content:
        content += history_header
        
    rows_to_add = ""
    for r in results_list:
        valid_icon = "YES" if r['valid'] else "NO"
        rows_to_add += f"| {timestamp_str} | `{r['image']}` | {r['try_level']} | {r['resolution']} | {r['boxes']} | `{r['plate']}` | {valid_icon} | {r['det_ms']:.1f} | {r['ocr_ms']:.1f} |\n"
        
    content += rows_to_add
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nEvaluation history logged to: {summary_md_path}")
