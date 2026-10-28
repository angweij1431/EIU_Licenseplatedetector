import os
import sys
import time
import cv2

# Import modular engine adapters
from lpr_engine import (
    YOLOv8Detector, 
    PaddleOCRAdapter, 
    EasyOCRAdapter, 
    LPRPipeline,
    VEHICLE_CLASSES
)

# Test image path
test_img = "car_test_2.jpg"

if not os.path.exists(test_img):
    raise FileNotFoundError(f"Benchmark test image not found: {test_img}")

print(f"Running comprehensive benchmark matrix on image: {test_img}")

# Initialize OCR adapters once
paddle_ocr = PaddleOCRAdapter(lang='en')
easy_ocr = EasyOCRAdapter(lang_list=['en'], gpu=False)

# Check model weights path
model_path = "Models_yolo/best.pt" if os.path.exists("Models_yolo/best.pt") else "yolov8n.pt"

# Define 4 testing combinations
combinations = [
    {
        'id': 'Combo 1 (Filtered + PaddleOCR)',
        'detector': YOLOv8Detector(model_path, target_classes=VEHICLE_CLASSES, confidence_threshold=0.35),
        'ocr': paddle_ocr,
        'out_img': 'Outputs/output_combo1_paddle_filtered.jpg'
    },
    {
        'id': 'Combo 2 (Filtered + EasyOCR)',
        'detector': YOLOv8Detector(model_path, target_classes=VEHICLE_CLASSES, confidence_threshold=0.35),
        'ocr': easy_ocr,
        'out_img': 'Outputs/output_combo2_easy_filtered.jpg'
    },
    {
        'id': 'Combo 3 (Unfiltered + PaddleOCR)',
        'detector': YOLOv8Detector(model_path, target_classes=None, confidence_threshold=0.25),
        'ocr': paddle_ocr,
        'out_img': 'Outputs/output_combo3_paddle_unfiltered.jpg'
    },
    {
        'id': 'Combo 4 (Unfiltered + EasyOCR)',
        'detector': YOLOv8Detector(model_path, target_classes=None, confidence_threshold=0.25),
        'ocr': easy_ocr,
        'out_img': 'Outputs/output_combo4_easy_unfiltered.jpg'
    }
]

benchmark_results = []

for combo in combinations:
    pipeline = LPRPipeline(detector=combo['detector'], ocr=combo['ocr'])
    res = pipeline.process(
        test_img, 
        save_annotated_name=combo['out_img'],
        padding_percent=0.20
    )
    
    # Save focused output image for Combo 1
    if combo['id'].startswith('Combo 1'):
        annotated = cv2.imread(combo['out_img'])
        if annotated is not None:
            cv2.imwrite(os.path.join('Outputs', 'output_annotated.jpg'), annotated)

    # Format extracted OCR strings
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

# Write markdown benchmark summary report
report_file = "BENCHMARK_SUMMARY.md"
with open(report_file, "w", encoding="utf-8") as f:
    f.write("# License Plate Recognition Benchmark Evaluation\n\n")
    f.write(f"Target image: {test_img}\n")
    f.write(f"Execution date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("| Combination | Vehicle Filter | Boxes | Classes | Extracted Text | Det Latency | OCR Latency | RAM |\n")
    f.write("| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: |\n")
    for r in benchmark_results:
        classes_str = ", ".join(r['detected_classes']) if r['detected_classes'] else "None"
        is_filtered = "Yes" if "Filtered" in r['name'] and "Unfiltered" not in r['name'] else "No"
        f.write(f"| {r['name']} | {is_filtered} | {r['num_boxes']} | {classes_str} | {r['extracted_texts']} | {r['det_latency_ms']:.1f} ms | {r['ocr_latency_ms']:.1f} ms | {r['ram_mb']:.1f} MB |\n")

print(f"Benchmark completed successfully. Report saved to: {report_file}")
