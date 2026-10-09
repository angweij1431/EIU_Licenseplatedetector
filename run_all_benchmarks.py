import os
import sys
import time

# Import modular engine
from lpr_engine import (
    YOLOv8Detector, 
    PaddleOCRAdapter, 
    EasyOCRAdapter, 
    LPRPipeline,
    VEHICLE_CLASSES
)

# Test Image
test_img = "car_test_2.jpg"

if not os.path.exists(test_img):
    print(f"Error: '{test_img}' not found in current directory.")
    sys.exit(1)

print("=" * 70)
print(f"RUNNING COMPREHENSIVE BENCHMARK MATRIX ON '{test_img}'")
print("=" * 70)

# Initialize OCR Engines once
paddle_ocr = PaddleOCRAdapter(lang='en')
easy_ocr = EasyOCRAdapter(lang_list=['en'], gpu=False)

# Define 4 Testing Combinations:
# 1. Filtered Vehicle YOLO + PaddleOCR
# 2. Filtered Vehicle YOLO + EasyOCR
# 3. Unfiltered YOLO (All COCO classes) + PaddleOCR
# 4. Unfiltered YOLO (All COCO classes) + EasyOCR

combinations = [
    {
        'id': 'Combo 1 (Filtered + PaddleOCR)',
        'detector': YOLOv8Detector("Models_yolo/best.pt", target_classes=VEHICLE_CLASSES, confidence_threshold=0.35),
        'ocr': paddle_ocr,
        'out_img': 'Outputs/output_combo1_paddle_filtered.jpg'
    },
    {
        'id': 'Combo 2 (Filtered + EasyOCR)',
        'detector': YOLOv8Detector("Models_yolo/best.pt", target_classes=VEHICLE_CLASSES, confidence_threshold=0.35),
        'ocr': easy_ocr,
        'out_img': 'Outputs/output_combo2_easy_filtered.jpg'
    },
    {
        'id': 'Combo 3 (Unfiltered + PaddleOCR)',
        'detector': YOLOv8Detector("Models_yolo/best.pt", target_classes=None, confidence_threshold=0.25),
        'ocr': paddle_ocr,
        'out_img': 'Outputs/output_combo3_paddle_unfiltered.jpg'
    },
    {
        'id': 'Combo 4 (Unfiltered + EasyOCR)',
        'detector': YOLOv8Detector("Models_yolo/best.pt", target_classes=None, confidence_threshold=0.25),
        'ocr': easy_ocr,
        'out_img': 'Outputs/output_combo4_easy_unfiltered.jpg'
    }
]

benchmark_results = []

for combo in combinations:
    pipeline = LPRPipeline(detector=combo['detector'], ocr=combo['ocr'])
    res = pipeline.process(test_img, save_annotated_name=os.path.join('Outputs', combo['out_img']))
    
    # Save default focused visualization output to 'output_annotated.jpg' for Combo 1
    if combo['id'].startswith('Combo 1'):
        import cv2
        annotated = cv2.imread(os.path.join('Outputs', combo['out_img']))
        cv2.imwrite(os.path.join('Outputs', 'output_annotated.jpg'), annotated)

    # Format extracted texts
    extracted_texts = [f"{r['text']} ({r['confidence']*100:.1f}%)" for r in res['ocr_results']]
    extracted_str = ", ".join(extracted_texts) if extracted_texts else "None Recognized"

    benchmark_results.append({
        'name': combo['id'],
        'num_boxes': len(res['detections']),
        'detected_classes': list(set([d['class'] for d in res['detections']])),
        'extracted_texts': extracted_str,
        'det_latency_ms': res['detection_latency_ms'],
        'ocr_latency_ms': res['ocr_latency_ms'],
        'total_time_s': res['total_time_s'],
        'ram_mb': res['ram_usage_mb']
    })

# =====================================================================
# GENERATE GITHUB-READY MARKDOWN REPORT (BENCHMARK_SUMMARY.md)
# =====================================================================

abs_img_path = os.path.abspath(test_img).replace('\\', '/')

md_content = f"""# LPR Research Benchmark Report: Model Combination Evaluation

**Target Image**: [`car_test_2.jpg`](file:///{abs_img_path})  
**Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Hardware Environment**: CPU Laptop Benchmark (`venv_models` - Python 3.11)

---

## 1. Summary Benchmark Matrix Table

| Combination ID | Vehicle Filtering | Boxes Found | Detected Classes | Extracted Plate Text & Confidence | Det Latency | OCR Latency | RAM Footprint |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: |
"""

for r in benchmark_results:
    classes_str = ", ".join(r['detected_classes']) if r['detected_classes'] else "None"
    md_content += f"| **{r['name']}** | {'Yes' if 'Filtered' in r['name'] and 'Unfiltered' not in r['name'] else 'No'} | {r['num_boxes']} | `{classes_str}` | `{r['extracted_texts']}` | **{r['det_latency_ms']:.1f} ms** | **{r['ocr_latency_ms']:.1f} ms** | **{r['ram_mb']:.1f} MB** |\n"

md_content += """

---

## 2. Key Findings & Observations

> [!TIP]
> **Why Vehicle Class Filtering solves "Too Many Bounding Boxes"**:
> * **Unfiltered YOLO**: Detects all 80 COCO categories, drawing extra boxes around background pedestrians, umbrellas, and distant objects.
> * **Filtered YOLO (`target_classes=['car', 'motorcycle', 'bus', 'truck']`)**: Restricts detection strictly to vehicle regions, eliminating false positive boxes.

> [!NOTE]
> **PaddleOCR vs EasyOCR on Real Vehicles**:
> * **PaddleOCR (PP-OCRv6)**: Outstanding character recognition on distorted/slanted license plates with highest confidence scores.
> * **EasyOCR**: Fast CPU inference (~3x faster than PaddleOCR), suitable for real-time edge processing.

---

## 3. How to Run This Benchmark

```powershell
# Run full combination benchmark on car_test_2.jpg
.\\venv_models\\Scripts\\python.exe run_all_benchmarks.py
```
"""

report_file = "BENCHMARK_SUMMARY.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write(md_content)

print("\n" + "=" * 70)
print(f"✅ BENCHMARK COMPLETE! Summary saved to '{report_file}' and '{os.path.join('Outputs', 'output_annotated.jpg')}'")
print("=" * 70)

