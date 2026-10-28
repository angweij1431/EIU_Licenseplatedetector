# Simple interactive demo to test eiu_lpr and view formatted JSON output
import json
import os
import sys
import cv2

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

from eiu_lpr import LPRPipeline

def main():
    print("=" * 65)
    print("EIU LPR Production Vision Engine - Demo")
    print("=" * 65)

    # Initialize pipeline
    print("\n1. Initializing LPRPipeline with ONNX Runtime...")
    pipeline = LPRPipeline(
        detector_path="models/detector.onnx",
        ocr_path="models/cct_xs.onnx",
        ocr_config_path="models/cct_xs_plate_config.yaml",
        detection_threshold=0.20,
    )
    print("   [OK] Pipeline loaded and ready.")

    # Optional warmup
    print("\n2. Priming ONNX sessions (warmup)...")
    pipeline.warmup(iterations=2)
    print("   [OK] Sessions warmed up.")

    # Test an image
    test_image_path = "test_images/car_test_2.jpg"
    print(f"\n3. Reading test image: {test_image_path}")
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"   [ERROR] Could not load image from {test_image_path}")
        return

    # Process frame
    print("4. Running pipeline.process(image, color_format='BGR')...")
    result = pipeline.process(image, color_format="BGR", request_id="req-001")

    # Display clean JSON output matching reference schema exactly
    print("\n" + "=" * 65)
    print("EXACT REFERENCE SCHEMA (result.to_dict(minimal=True)):")
    print("=" * 65)
    print(json.dumps(result.to_dict(minimal=True), indent=2))

    print("\n" + "=" * 65)
    print("FULL EVIDENCE TELEMETRY (result.to_dict(minimal=False)):")
    print("=" * 65)
    print(json.dumps(result.to_dict(minimal=False), indent=2))
    print("=" * 65)

    print(f"\nProcessing Time: {result.processing_time_ms:.1f} ms")
    if result.candidates:
        top = result.candidates[0]
        print(f"Plate Detected : {top.normalized_text}")
        print(f"Valid Syntax   : {'YES (Circular 24/2023)' if top.is_valid else 'NO'}")
        print(f"Mean Confidence: {top.ocr_confidence * 100:.1f}% (Average)")
        print(f"Min Confidence : {top.ocr_min_confidence * 100:.1f}% (Lowest char)")
        print(f"Detector Conf  : {top.detection_score * 100:.1f}%")
        print(f"Bounding Box   : {top.bbox}")
    print("\n[SUCCESS] Demo completed with zero disk writes and zero errors.\n")

if __name__ == "__main__":
    main()
