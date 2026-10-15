import os
import sys
import time
import argparse
from datetime import datetime

# Import modular pipeline components from lpr_engine
from lpr_engine import (
    YOLODetector, 
    PaddleOCRAdapter, 
    LPRPipeline
)
from lpr_core.validator import VietnamPlateValidator

def run_inference(image_path: str, model_path: str = "Models_yolo/best.onnx", conf_thresh: float = 0.25, 
                  force_try2: bool = False, force_try3: bool = False, measure_warm: bool = True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}. Please place exported ONNX weights in Models_yolo/.")

    print("======================================================================")
    print(f"LPR MODEL INFERENCE: {os.path.basename(image_path)}")
    print("======================================================================")
    print(f"[1] Loading AI Models (YOLO ONNX Plate Detector & PaddleOCR)")
    detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=conf_thresh)
    ocr = PaddleOCRAdapter(lang='en')
    pipeline = LPRPipeline(detector=detector, ocr=ocr, enable_preprocessing=True)

    output_filename = f"output_{os.path.basename(image_path)}"
    
    print("\n[2] Running Inference Pipeline (Plate Detection -> Crop -> OCR -> Syntax Check)")
    start_time = time.time()
    summary = pipeline.process(
        image_path,
        save_annotated_name=output_filename,
        sort_by='front', # this one determines the order of the detected plates if multiple are found
        primary_only=True,
        padding_percent=0.20,
        force_second_try=force_try2,
        force_third_try=force_try3,
        measure_warm=measure_warm
    )
    total_time = time.time() - start_time

    raw_lines = []
    extracted_text = "nil"
    is_valid = False
    format_similarity = 0.0

    if summary['ocr_results']:
        raw_lines = [r['text'] for r in summary['ocr_results']]
        val_lines = [r['validated_text'] for r in summary['ocr_results']]
        joined_text = "".join(val_lines)
        final_val = VietnamPlateValidator.clean_and_correct(joined_text)
        extracted_text = final_val['text']
        is_valid = final_val['is_valid']
        format_similarity = final_val.get('format_similarity', 0.0)

    print("\n======================================================================")
    print("FINAL RESULTS")
    print("======================================================================")
    print(f"Input Image             : {image_path}")
    print(f"Plate Detection Model   : {model_path}")
    print(f"Text Recognition Model  : PaddleOCR")
    print(f"Pipeline Escalation Path: {summary['try_level']} ({summary['resolution']})")
    print("----------------------------------------------------------------------")
    print("1. Plate detection results:")
    print(f"  Plates Found          : {len(summary['detections'])}")
    if summary['detections']:
        box = [int(v) for v in summary['detections'][0]['box']]
        conf = summary['detections'][0]['confidence']
        print(f"  Bounding Box [xyxy]   : {box} (x1, y1, x2, y2 pixels)")
        print(f"  Detection Confidence  : {conf * 100:.1f}%")
    else:
        print("  Bounding Box [xyxy]   : None")
        print("  Detection Confidence  : 0.0%")
    print("----------------------------------------------------------------------")
    print("2. Text recognition & validation results:")
    print(f"  Raw OCR Segments      : {raw_lines}")
    print(f"  Cleaned License Plate : {extracted_text}")
    print(f"  Vietnam Syntax Valid  : {'correct' if is_valid else 'NO'}")
    print(f"  Template Similarity   : {format_similarity * 100:.1f}%")
    print("----------------------------------------------------------------------")
    print("3. Speed of execution:")
    if summary.get('cold_detection_latency_ms') is not None and summary.get('warm_detection_latency_ms') is not None:
        cold_tot = summary['cold_total_latency_ms']
        warm_tot = summary['warm_total_latency_ms']
        print("  [Cold-Start - First Session Load]")
        print(f"    YOLO Detection Time : {summary['cold_detection_latency_ms']:.1f} ms")
        print(f"    PaddleOCR Text Time : {summary['cold_ocr_latency_ms']:.1f} ms")
        print(f"    Total Pipeline Time : {cold_tot:.1f} ms ({cold_tot / 1000.0:.2f} seconds)")
        print("  [Steady-State - Operational Gate Speed (Without Cold Start)]")
        print(f"    YOLO Detection Time : {summary['warm_detection_latency_ms']:.1f} ms")
        print(f"    PaddleOCR Text Time : {summary['warm_ocr_latency_ms']:.1f} ms")
        print(f"    Total Pipeline Time : {warm_tot:.1f} ms ({warm_tot / 1000.0:.2f} seconds)")
    else:
        print(f"  YOLO Detection Time   : {summary['detection_latency_ms']:.1f} ms")
        print(f"  PaddleOCR Text Time   : {summary['ocr_latency_ms']:.1f} ms")
        print(f"  Total Inference Time  : {total_time * 1000:.1f} ms ({total_time:.2f} seconds)")
    print(f"  Saved Image with Box  : {summary['annotated_path']}")
    print("======================================================================\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone LPR Model Inference Tool")
    parser.add_argument("--image_path", nargs="?", default="test_images/car_test_2.jpg", help="Path to input test image")
    parser.add_argument("--model", default="Models_yolo/best.onnx", help="Path to ONNX detector model (default: Models_yolo/best.onnx)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--force_try2", action="store_true", help="Force 2nd try adaptive image enhancement (CLAHE + upscale)")
    parser.add_argument("--force_try3", action="store_true", help="Force 3rd try deep zoom (1024px detector pass)")
    parser.add_argument("--no_warm", action="store_true", help="Disable steady-state warm pass measurement (cold-start only)")
    args = parser.parse_args()

    run_inference(args.image_path, model_path=args.model, conf_thresh=args.conf, 
                  force_try2=args.force_try2, force_try3=args.force_try3, 
                  measure_warm=not args.no_warm)
