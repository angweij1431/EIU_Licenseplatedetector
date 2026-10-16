import os
import sys
import time
import glob
from datetime import datetime

# Allow importing lpr_engine and lpr_core from root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lpr_engine import YOLODetector, PaddleOCRAdapter, LPRPipeline
from lpr_core.validator import VietnamPlateValidator

MODELS_DIR = os.path.join(PROJECT_ROOT, "Models_yolo")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "test_images")
SUMMARY_MD_PATH = os.path.join(os.path.dirname(__file__), "BENCHMARK_SUMMARY.md")


def main():
    print("=========================================================================================================================")
    print("EMPIRICAL LPR PIPELINE BENCHMARK (REAL MODEL OUTPUTS & CONFIDENCE SCORES ONLY)")
    print("=========================================================================================================================")

    # Discover target .onnx models in Models_yolo
    onnx_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.onnx")))
    # Exclude best.onnx and best_11s.onnx as requested
    # onnx_files = [f for f in onnx_files if os.path.basename(f) not in ["best_11.onnx"]]

    if not onnx_files:
        print("No eligible .onnx models found in Models_yolo.")
        return

    # Collect test images
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print("No test images found in test_images/ directory.")
        return

    print(f"Evaluating {len(onnx_files)} models across {len(image_files)} test images.\n")

    # Initialize baseline PaddleOCR once
    ocr = PaddleOCRAdapter(lang='en', mode='full')

    all_image_records = []
    model_summary_records = []

    for model_path in onnx_files:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating: {model_name}...")

        detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.25)
        pipeline = LPRPipeline(detector=detector, ocr=ocr, enable_preprocessing=True)

        current_model_rows = []

        for img_path in image_files:
            img_basename = os.path.basename(img_path)
            t_start = time.time()
            summary = pipeline.process(
                img_path,
                sort_by='front',
                primary_only=True,
                padding_percent=0.20
            )
            total_dt_ms = (time.time() - t_start) * 1000.0

            # 1. Plate Detection Confidence
            plate_found = len(summary['detections']) > 0
            det_conf = (summary['detections'][0]['confidence'] * 100.0) if plate_found else 0.0

            # 2. OCR Text and Confidence
            ocr_conf = 0.0
            extracted_text = "NONE"
            is_valid = False

            if summary['ocr_results']:
                ocr_confs = [r.get('confidence', 0.0) for r in summary['ocr_results']]
                ocr_conf = (sum(ocr_confs) / len(ocr_confs)) * 100.0 if ocr_confs else 0.0

                val_lines = [r.get('validated_text', r.get('text', '')) for r in summary['ocr_results']]
                joined = "".join(val_lines)
                final_val = VietnamPlateValidator.clean_and_correct(joined)
                extracted_text = final_val['text']
                is_valid = final_val['is_valid']

            # 3. Gate Status / Try Level
            try_str = summary.get('try_level', '1st try')
            if is_valid:
                gate_status = "Try 1" if "1st" in try_str else ("Try 2" if "2nd" in try_str else "Try 3")
            else:
                gate_status = "Gate Closed"

            row = {
                'model': model_name,
                'image': img_basename,
                'plate_found': plate_found,
                'det_conf': det_conf,
                'ocr_valid': is_valid,
                'ocr_conf': ocr_conf,
                'gate_status': gate_status,
                'plate_text': extracted_text,
                'det_ms': summary['detection_latency_ms'],
                'ocr_ms': summary['ocr_latency_ms'],
                'total_ms': total_dt_ms
            }
            current_model_rows.append(row)
            all_image_records.append(row)

        # Aggregate Model Level Metrics
        n = len(current_model_rows)
        plates_detected_count = sum(1 for r in current_model_rows if r['plate_found'])
        gates_opened_count = sum(1 for r in current_model_rows if r['ocr_valid'])
        try1_count = sum(1 for r in current_model_rows if r['gate_status'] == "Try 1")
        try2_count = sum(1 for r in current_model_rows if r['gate_status'] == "Try 2")
        closed_count = sum(1 for r in current_model_rows if r['gate_status'] == "Gate Closed")

        # Compute average confidences across detected plates and non-zero OCR outputs
        detected_confs = [r['det_conf'] for r in current_model_rows if r['plate_found']]
        avg_det_conf = (sum(detected_confs) / len(detected_confs)) if detected_confs else 0.0

        valid_ocr_confs = [r['ocr_conf'] for r in current_model_rows if r['ocr_conf'] > 0.0]
        avg_ocr_conf = (sum(valid_ocr_confs) / len(valid_ocr_confs)) if valid_ocr_confs else 0.0

        # Segment cold-start (image 0) from steady-state (images 1..n)
        cold_det_ms = current_model_rows[0]['det_ms']
        steady_rows = current_model_rows[1:] if n > 1 else current_model_rows
        avg_steady_det_ms = sum(r['det_ms'] for r in steady_rows) / len(steady_rows)
        avg_steady_ocr_ms = sum(r['ocr_ms'] for r in steady_rows) / len(steady_rows)
        avg_steady_total_ms = sum(r['total_ms'] for r in steady_rows) / len(steady_rows)

        model_summary_records.append({
            'model': model_name,
            'recall_rate': f"{plates_detected_count}/{n} ({(plates_detected_count/n)*100:.1f}%)",
            'avg_det_conf': avg_det_conf,
            'gate_open_rate': f"{gates_opened_count}/{n} ({(gates_opened_count/n)*100:.1f}%)",
            'avg_ocr_conf': avg_ocr_conf,
            'try1_passes': try1_count,
            'try2_passes': try2_count,
            'gate_closed': closed_count,
            'cold_det_ms': cold_det_ms,
            'avg_steady_det_ms': avg_steady_det_ms,
            'avg_steady_ocr_ms': avg_steady_ocr_ms,
            'avg_steady_total_ms': avg_steady_total_ms
        })

    # Print Image-Level Details Table
    print("\n=========================================================================================================================")
    print("DETAILED PER-IMAGE REAL OUTPUTS & CONFIDENCE SCORES")
    print("=========================================================================================================================")
    print(f"{'Model':<19} | {'Image':<14} | {'Det (Plate Conf %)':<20} | {'OCR (Conf %)':<16} | {'Status':<11} | {'Plate Text':<11} | {'Det (ms)':<9} | {'OCR (ms)':<9} | {'Total (ms)':<10}")
    print("-" * 128)
    for r in all_image_records:
        det_disp = f"{'YES' if r['plate_found'] else 'NO'} ({r['det_conf']:.1f}%)"
        ocr_disp = f"{'YES' if r['ocr_valid'] else 'NO'} ({r['ocr_conf']:.1f}%)"
        print(f"{r['model']:<19} | {r['image']:<14} | {det_disp:<20} | {ocr_disp:<16} | {r['gate_status']:<11} | {r['plate_text']:<11} | {r['det_ms']:>7.1f} ms | {r['ocr_ms']:>7.1f} ms | {r['total_ms']:>8.1f} ms")
    print("=========================================================================================================================\n")

    # Print Aggregate Model Summary Table
    print("\n=========================================================================================================================")
    print("MODEL AGGREGATE SUMMARY (REAL MEASURED SCORES & LATENCY)")
    print("=========================================================================================================================")
    print(f"{'ONNX Model':<19} | {'Plate Recall':<15} | {'Avg Det Conf':<12} | {'Gate Open':<11} | {'Cold Det ms':<12} | {'Steady Det ms':<14} | {'Steady Total ms':<16}")
    print("-" * 128)
    for m in model_summary_records:
        print(f"{m['model']:<19} | {m['recall_rate']:<15} | {m['avg_det_conf']:>10.1f}% | {m['gate_open_rate']:<11} | {m['cold_det_ms']:>10.1f} ms | {m['avg_steady_det_ms']:>12.1f} ms | {m['avg_steady_total_ms']:>14.1f} ms")
    print("=========================================================================================================================\n")

    # Log clean Markdown table into BENCHMARK_SUMMARY.md
    if os.path.exists(SUMMARY_MD_PATH):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        table_md = f"\n## 9. Real Empirical Output & Confidence Matrix\n\n"
        table_md += f"*Evaluated at: `{timestamp_str}` across `{len(image_files)}` test images (Measured outputs and model confidence scores only)*\n\n"
        
        table_md += "### Model Summary\n\n"
        table_md += "| ONNX Model | Plate Recall | Avg Det Conf % | Gate Open Rate | Avg OCR Conf % | Try 1 | Try 2 | Gate Closed | Cold Det (ms) | Steady Det (ms) | Steady Total (ms) |\n"
        table_md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for m in model_summary_records:
            table_md += f"| `{m['model']}` | {m['recall_rate']} | {m['avg_det_conf']:.1f}% | **{m['gate_open_rate']}** | {m['avg_ocr_conf']:.1f}% | {m['try1_passes']} | {m['try2_passes']} | {m['gate_closed']} | {m['cold_det_ms']:.1f} ms | {m['avg_steady_det_ms']:.1f} ms | **{m['avg_steady_total_ms']:.1f} ms** |\n"

        table_md += "\n### Per-Image Real Outputs\n\n"
        table_md += "| Model | Image | Plate Detected (Conf %) | Valid Format (OCR Conf %) | Gate Status | Extracted Plate | Det (ms) | OCR (ms) | Total (ms) |\n"
        table_md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for r in all_image_records:
            det_disp = f"{'YES' if r['plate_found'] else 'NO'} ({r['det_conf']:.1f}%)"
            ocr_disp = f"{'YES' if r['ocr_valid'] else 'NO'} ({r['ocr_conf']:.1f}%)"
            table_md += f"| `{r['model']}` | `{r['image']}` | {det_disp} | {ocr_disp} | **{r['gate_status']}** | `{r['plate_text']}` | {r['det_ms']:.1f} | {r['ocr_ms']:.1f} | {r['total_ms']:.1f} |\n"
        table_md += "\n"

        with open(SUMMARY_MD_PATH, "a", encoding="utf-8") as f:
            f.write(table_md)
        print(f"Real output matrix appended directly to: {SUMMARY_MD_PATH}")


if __name__ == "__main__":
    main()
