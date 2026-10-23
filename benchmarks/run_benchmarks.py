import os
import sys
import argparse
import glob
from datetime import datetime

# Allow importing lpr_engine and lpr_core from the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import modular pipeline adapters from lpr_engine
from lpr_engine import (
    YOLODetector, 
    LPRPipeline,
    get_ocr_adapter
)
from lpr_core.validator import VietnamPlateValidator

# Parse command line argument for vehicle test image or directory path
parser = argparse.ArgumentParser(description="Run cct-xs Pipeline Benchmark on an Image or Directory of Images")
parser.add_argument("input_path", nargs="?", default=None, help="Path to input image or directory (default: test_images)")
parser.add_argument("--image_path", default=None, help="Path to input image or directory (flag alternative)")
parser.add_argument("--model", default="Models_yolo/balanced_best_yolo11n.onnx", help="Path to detector model (default: Models_yolo/balanced_best_yolo11n.onnx)")
parser.add_argument("--ocr", default="cct_xs", help="OCR engine mode: paddleocr, paddle_no_dbnet, paddle_split, crnn, lprnet, cct_xs (default: cct_xs)")
parser.add_argument("--output_dir", default=None, help="Directory to save output annotated images (optional)")
args = parser.parse_args()

target_path = args.image_path or args.input_path or "test_images"
if not os.path.exists(target_path):
    raise FileNotFoundError(f"Target path not found: {target_path}")

# Initialize primary plate detector
model_path = args.model
if not os.path.exists(model_path):
    if model_path == "Models_yolo/best.onnx" and os.path.exists("Models_yolo/best.pt"):
        print(f"Notice: 'Models_yolo/best.onnx' not found. Using 'Models_yolo/best.pt'.")
        model_path = "Models_yolo/best.pt"
    else:
        raise FileNotFoundError(f"Detector model not found: {model_path}. Please place exported weights in Models_yolo/.")

detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.45)
ocr = get_ocr_adapter(args.ocr)
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
        padding_percent=0.20,
        output_dir=args.output_dir,
        save_crops=True # to show crops and augmentations
    )


    
    extracted_text = "NONE"
    is_valid = False
    raw_lines = []
    ocr_conf = 0.0
    min_conf = 0.0
    region_str = "N/A"
    
    # Extract bounding box detection confidence
    box_conf = 0.0
    if summary['detections']:
        box_conf = float(summary['detections'][0]['confidence'])

    if summary['ocr_results']:
        raw_lines = [r['text'] for r in summary['ocr_results']]
        val_lines = [r['validated_text'] for r in summary['ocr_results']]
        joined_text = "".join(val_lines)
        final_val = VietnamPlateValidator.clean_and_correct(joined_text)
        
        r0 = summary['ocr_results'][0]
        ocr_conf = float(r0.get('confidence', 0.0))
        min_conf = float(r0.get('min_confidence', ocr_conf))
        reg_name = r0.get('region', '')
        reg_prob = float(r0.get('region_prob', 0.0))
        
        if reg_name and reg_prob > 0:
            region_str = f"{reg_name} ({reg_prob*100:.1f}%)"
        elif reg_name:
            region_str = reg_name

        # Strict validation: Vietnamese syntax AND Vietnam regional head
        is_region_valid = (reg_name == 'Vietnam')
        is_valid = bool(final_val['is_valid'] and is_region_valid)
        extracted_text = final_val['text'] if is_valid else (f"{final_val['text']} (REJECTED)" if final_val['text'] else "NONE")

    results_list.append({
        'image': os.path.basename(img_path),
        'try_level': summary.get('try_level', '1st try'),
        'resolution': summary.get('resolution', '640px'),
        'boxes': len(summary['detections']),
        'box_conf': box_conf, # added box conf
        'plate': extracted_text,
        'valid': is_valid,
        'ocr_conf': ocr_conf,
        'min_conf': min_conf,
        'region': region_str,
        'det_ms': summary['detection_latency_ms'],
        'ocr_ms': summary['ocr_latency_ms'],
        'raw': raw_lines
    })

## Print aggregate summary report
print("\n" + "=" * 144)
print("BENCHMARK SUMMARY REPORT (CCT-XS v2 Global)")
print("=" * 144)
print(f"{'Image':<22} | {'Try Level':<20} | {'Boxes':<5} | {'Box Conf':<8} | {'Plate':<15} | {'Valid':<5} | {'Mean Conf':<9} | {'Min Char':<8} | {'Region (Prob)':<18} | {'Det (ms)':<9} | {'OCR (ms)':<8}")
print("-" * 144)
for r in results_list:
    valid_str = "YES" if r['valid'] else "NO"
    box_conf_str = f"{r['box_conf']*100:.1f}%" if r['box_conf'] > 0 else "-"
    conf_str = f"{r['ocr_conf']*100:.1f}%" if r['ocr_conf'] > 0 else "-"
    min_str = f"{r['min_conf']*100:.1f}%" if r['min_conf'] > 0 else "-"
    print(f"{r['image']:<22} | {r['try_level']:<20} | {r['boxes']:<5} | {box_conf_str:<8} | {r['plate']:<15} | {valid_str:<5} | {conf_str:<9} | {min_str:<8} | {r['region']:<18} | {r['det_ms']:<9.1f} | {r['ocr_ms']:<8.1f}")
print("-" * 144)

# Compute aggregate evaluation metrics
total_imgs = len(results_list)
detected_count = sum(1 for r in results_list if r['boxes'] > 0)
valid_count = sum(1 for r in results_list if r['valid'])
first_try_count = sum(1 for r in results_list if '1st' in r['try_level'])
second_try_count = sum(1 for r in results_list if '2nd' in r['try_level'])

# Calculate steady-state latency (excluding cold-start first image if batch > 1)
steady_det_list = [r['det_ms'] for r in results_list[1:]] if total_imgs > 1 else [r['det_ms'] for r in results_list]
steady_ocr_list = [r['ocr_ms'] for r in results_list[1:]] if total_imgs > 1 else [r['ocr_ms'] for r in results_list]
avg_steady_det = sum(steady_det_list) / len(steady_det_list) if steady_det_list else 0.0
avg_steady_ocr = sum(steady_ocr_list) / len(steady_ocr_list) if steady_ocr_list else 0.0

print("\nAggregate Evaluation Metrics:")
print(f"  Total Images Processed      : {total_imgs}")
print(f"  Plate Detection Recall      : {(detected_count / total_imgs) * 100:.1f}% ({detected_count}/{total_imgs})")
print(f"  Syntax Validation Pass Rate : {(valid_count / total_imgs) * 100:.1f}% ({valid_count}/{total_imgs})")
print(f"  1st-Try Resolution Rate     : {(first_try_count / total_imgs) * 100:.1f}% ({first_try_count}/{total_imgs})")
print(f"  2nd-Try Escalation Rate     : {(second_try_count / total_imgs) * 100:.1f}% ({second_try_count}/{total_imgs})")
if total_imgs > 1:
    print(f"  Cold-Start First Run Det ms : {results_list[0]['det_ms']:.1f} ms")
    print(f"  Average Steady-State Det ms : {avg_steady_det:.1f} ms (without cold start)")
    print(f"  Average Steady-State OCR ms : {avg_steady_ocr:.1f} ms")
else:
    print(f"  Detection Latency           : {avg_steady_det:.1f} ms")
    print(f"  OCR Latency                 : {avg_steady_ocr:.1f} ms")


# record latest run into BENCHMARK_SUMMARY.md
summary_md_path = os.path.join(os.path.dirname(__file__), "BENCHMARK_SUMMARY.md")
if os.path.exists(summary_md_path):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_md_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # history logger.
    history_header = "\n\n## 8. Fast Plate OCR (CCT-XS) Evaluation History\n\n| Timestamp | Test Image | Try Level | Resolution | Boxes | Box Conf | Plate Text | Valid Format | Mean Conf | Min Char | Region (Prob) | Det (ms) | OCR (ms) |\n| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: |\n"
    if "## 8. Fast Plate OCR (CCT-XS) Evaluation History" not in content:
        content += history_header
        
    rows_to_add = ""
    for r in results_list:
        valid_icon = "YES" if r['valid'] else "NO"
        box_conf_str = f"{r['box_conf']*100:.1f}%" if r['box_conf'] > 0 else "-"
        conf_str = f"{r['ocr_conf']*100:.1f}%" if r['ocr_conf'] > 0 else "-"
        min_str = f"{r['min_conf']*100:.1f}%" if r['min_conf'] > 0 else "-"
        rows_to_add += f"| {timestamp_str} | `{r['image']}` | {r['try_level']} | {r['resolution']} | {r['boxes']} | {box_conf_str} | `{r['plate']}` | {valid_icon} | {conf_str} | {min_str} | {r['region']} | {r['det_ms']:.1f} | {r['ocr_ms']:.1f} |\n"

  
  
    content += rows_to_add
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nEvaluation history logged to: {summary_md_path}")
