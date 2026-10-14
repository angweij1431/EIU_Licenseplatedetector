import os
import sys
import time
import argparse
import cv2

# Allow importing lpr_engine and lpr_core from the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import core modules
from lpr_engine import YOLODetector, PaddleOCRAdapter
from lpr_core.preprocessor import PlatePreprocessor
from lpr_core.validator import VietnamPlateValidator

# Ensure Outputs directory exists
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Parse command line argument for test image path
parser = argparse.ArgumentParser(description="LPR Preprocessing Comparison (Raw vs Enhanced)")
parser.add_argument("image", nargs="?", default="test_images/car_test_3.jpg", help="Path to input test image (default: test_images/car_test_3.jpg)")
parser.add_argument("--model", default="Models_yolo/best.onnx", help="Path to ONNX detector model (default: Models_yolo/best.onnx)")
args = parser.parse_args()

image_path = args.image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Input image not found: {image_path}")

print(f"Running Preprocessing Comparison on: {image_path}")

# Initialize detector and OCR model
model_path = args.model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"ONNX model not found: {model_path}. Please place exported ONNX weights in Models_yolo/.")

detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.25)
ocr = PaddleOCRAdapter(lang='en')

# Load original image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Failed to load image: {image_path}")

# Run detection
det_start = time.time()
detections = detector.detect(image_path)
det_latency = (time.time() - det_start) * 1000

if not detections:
    print("No license plate bounding box detected. Cannot perform ablation.")
    sys.exit(1)

# Select primary detection target (highest confidence)
target = sorted(detections, key=lambda x: x['confidence'], reverse=True)[0]
x1, y1, x2, y2 = map(int, target['box'])
print(f"Detected target box: [{x1}, {y1}, {x2}, {y2}] | Confidence: {target['confidence']:.2f}")

# Apply padding with minimum pixel boundary
h, w, _ = img.shape
pad_w = max(20, int((x2 - x1) * 0.20))
pad_h = max(15, int((y2 - y1) * 0.20))

crop_x1 = max(0, x1 - pad_w)
crop_y1 = max(0, y1 - pad_h)
crop_x2 = min(w, x2 + pad_w)
crop_y2 = min(h, y2 + pad_h)

raw_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

# Produce enhanced crop using PlatePreprocessor pipeline
prep_start = time.time()
enhanced_crop = PlatePreprocessor.enhance_for_ocr(raw_crop)
prep_latency = (time.time() - prep_start) * 1000

# Save comparison crops to Outputs
base_name = os.path.splitext(os.path.basename(image_path))[0]
raw_crop_path = os.path.join(OUTPUTS_DIR, f"ablation_raw_{base_name}.jpg")
enhanced_crop_path = os.path.join(OUTPUTS_DIR, f"ablation_enhanced_{base_name}.jpg")
cv2.imwrite(raw_crop_path, raw_crop)
cv2.imwrite(enhanced_crop_path, enhanced_crop)
print(f"Raw crop saved: {raw_crop_path}")
print(f"Enhanced crop saved: {enhanced_crop_path}")

# Helper function to process crop with OCR and validator
def run_ocr_and_validate(crop_img):
    t0 = time.time()
    results = ocr.recognize(crop_img)
    latency = (time.time() - t0) * 1000
    if not results:
        return {'text': 'NONE', 'confidence': 0.0, 'is_valid': False, 'latency_ms': latency, 'raw_lines': []}
    
    raw_lines = [r['text'] for r in results]
    val_lines = [VietnamPlateValidator.clean_and_correct(r['text'])['text'] for r in results]
    joined_text = "-".join(val_lines) if len(val_lines) > 1 else val_lines[0]
    final_val = VietnamPlateValidator.clean_and_correct(joined_text)
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    return {
        'text': final_val['text'],
        'confidence': avg_conf,
        'is_valid': final_val['is_valid'],
        'latency_ms': latency,
        'raw_lines': raw_lines
    }

print("Running OCR on Raw Crop...")
res_raw = run_ocr_and_validate(raw_crop)

print("Running OCR on Enhanced Crop...")
res_enhanced = run_ocr_and_validate(enhanced_crop)

# Print comparison report
print("\nPreprocessing Comparison Results (Raw vs Enhanced):")
print(f"Image evaluated: {image_path}")
print(f"Target Bounding Box: [{x1}, {y1}, {x2}, {y2}] | Initial Box Size: {x2-x1}x{y2-y1} px")
print(f"Preprocessing overhead: {prep_latency:.2f} ms")
print("")
print("Configuration A (Raw Crop without Preprocessing):")
print(f"  Raw OCR lines: {res_raw['raw_lines']}")
print(f"  Parsed plate text: {res_raw['text']}")
print(f"  Average OCR confidence: {res_raw['confidence']:.4f}")
print(f"  Valid Vietnamese format: {res_raw['is_valid']}")
print(f"  OCR latency: {res_raw['latency_ms']:.2f} ms")
print("")
print("Configuration B (Enhanced Crop with CLAHE + Bilateral + Upscaling):")
print(f"  Raw OCR lines: {res_enhanced['raw_lines']}")
print(f"  Parsed plate text: {res_enhanced['text']}")
print(f"  Average OCR confidence: {res_enhanced['confidence']:.4f}")
print(f"  Valid Vietnamese format: {res_enhanced['is_valid']}")
print(f"  OCR latency: {res_enhanced['latency_ms']:.2f} ms")
print("")
print("Summary Delta:")
print(f"  Confidence difference: {res_enhanced['confidence'] - res_raw['confidence']:+.4f}")
print(f"  Validation status shift: Raw ({res_raw['is_valid']}) -> Enhanced ({res_enhanced['is_valid']})")

