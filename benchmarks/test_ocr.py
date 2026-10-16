import os
import sys
import time
import argparse
import cv2
import numpy as np

# Allow importing lpr_engine and lpr_core from root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lpr_engine import YOLODetector, get_ocr_adapter
from lpr_core.validator import VietnamPlateValidator

# Ensure Outputs directory exists
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "Outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def run_ocr_test(image_path: str, model_path: str = "Models_yolo/best.onnx"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Fallback to .pt if .onnx not yet exported
    if not os.path.exists(model_path):
        if model_path == "Models_yolo/best.onnx" and os.path.exists(os.path.join(PROJECT_ROOT, "Models_yolo/best.pt")):
            model_path = os.path.join(PROJECT_ROOT, "Models_yolo/best.pt")
        else:
            raise FileNotFoundError(f"Detector weights not found: {model_path}")

    print("======================================================================")
    print(f"OCR DIAGNOSTIC & GEOMETRIC LINE SPLIT TEST: {os.path.basename(image_path)}")
    print("======================================================================")

    # Step 1: Detect plate using YOLO
    detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.25)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    h_img, w_img = img_bgr.shape[:2]
    detections = detector.detect(image_path)
    if not detections:
        print("No license plate detected in input image.")
        return

    # Select primary plate detection (highest confidence)
    best_det = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
    box = [int(v) for v in best_det['box']]
    x1, y1, x2, y2 = box

    # Apply 20% padding floor
    pad_w = max(20, int((x2 - x1) * 0.20))
    pad_h = max(15, int((y2 - y1) * 0.20))
    x1_p = max(0, x1 - pad_w)
    y1_p = max(0, y1 - pad_h)
    x2_p = min(w_img, x2 + pad_w)
    y2_p = min(h_img, y2 + pad_h)

    crop = img_bgr[y1_p:y2_p, x1_p:x2_p]
    h_crop, w_crop = crop.shape[:2]
    aspect_ratio = w_crop / float(h_crop) if h_crop > 0 else 1.0

    print(f"Plate Crop Dimensions   : {w_crop}x{h_crop} px (Aspect Ratio: {aspect_ratio:.2f})")

    # Step 2: Perform OpenCV Geometric Line Slicing for Visual Inspection
    slice1_path = os.path.join(OUTPUTS_DIR, "debug_slice_line1.jpg")
    slice2_path = os.path.join(OUTPUTS_DIR, "debug_slice_line2.jpg")

    if aspect_ratio <= 2.5:
        # 2-line square plate: 10% vertical overlap
        top_h = int(h_crop * 0.55)
        bot_h = int(h_crop * 0.45)
        line1_crop = crop[0:top_h, :]
        line2_crop = crop[bot_h:h_crop, :]

        cv2.imwrite(slice1_path, line1_crop)
        cv2.imwrite(slice2_path, line2_crop)
        print("Plate Format Detected   : 2-Line (Square Plate)")
        print(f"Saved Line 1 Crop       : {slice1_path}")
        print(f"Saved Line 2 Crop       : {slice2_path}")
    else:
        # 1-line long plate
        cv2.imwrite(slice1_path, crop)  
        print("Plate Format Detected   : 1-Line (Long Plate)")
        print(f"Saved Line Crop         : {slice1_path}")

    # Import preprocessor for enhanced crop comparison
    from lpr_core.preprocessor import PlatePreprocessor
    enhanced_crop = PlatePreprocessor.enhance_for_ocr(crop)

    # Step 3: Compare OCR Modes on Raw vs Enhanced crops
    test_runs = [
        ("PaddleOCR Full (Raw Crop)",        "paddleocr",    crop),
        ("PaddleOCR Line-Split (Raw Crop)",   "paddle_split", crop),
        ("PaddleOCR Full (Enhanced Crop)",     "paddleocr",    enhanced_crop),
        ("PaddleOCR Line-Split (Enhanced Crop)", "paddle_split", enhanced_crop)
    ]

    print("\n----------------------------------------------------------------------")
    print("Running OCR Diagnostic Tests (Raw vs Enhanced Crops)...")
    print("----------------------------------------------------------------------")

    results = []
    for mode_label, engine_key, input_crop in test_runs:
        ocr_adapter = get_ocr_adapter(engine_key)

        # Warm up engine on first run
        ocr_adapter.recognize(input_crop)

        # Measure steady-state OCR latency
        t0 = time.time()
        ocr_res = ocr_adapter.recognize(input_crop)
        latency_ms = (time.time() - t0) * 1000.0

        raw_texts = [r['text'] for r in ocr_res]
        joined = "-".join(raw_texts) if len(raw_texts) > 1 else (raw_texts[0] if raw_texts else "")
        val = VietnamPlateValidator.clean_and_correct(joined)

        results.append({
            'mode': mode_label,
            'raw': raw_texts,
            'plate': val['text'],
            'valid': val['is_valid'],
            'latency': latency_ms
        })

    # Step 4: Display Diagnostic Comparison Table
    print("\n======================================================================")
    print("OCR DIAGNOSTIC SUMMARY")
    print("======================================================================")
    print(f"{'OCR Engine / Crop Mode':<38} | {'Raw Lines':<16} | {'Plate':<10} | {'Valid':<6} | {'Latency':<9}")
    print("-" * 88)
    for r in results:
        valid_str = "YES" if r['valid'] else "NO"
        raw_str = str(r['raw'])[:15]
        print(f"{r['mode']:<38} | {raw_str:<16} | {r['plate']:<10} | {valid_str:<6} | {r['latency']:>6.1f} ms")
    print("======================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone OCR Diagnostic & Visual Slice Test")
    parser.add_argument("image_path", nargs="?", default="test_images/car_test_3.jpg", help="Path to input vehicle image")
    parser.add_argument("--model", default="Models_yolo/best.onnx", help="Path to detector model (default: Models_yolo/best.onnx)")
    args = parser.parse_args()

    run_ocr_test(args.image_path, model_path=args.model)

