# CCT-XS (Fast Plate OCR) Integration Walkthrough

**Date:** 2026-10-21  
**Scope:** Integration of Compact Convolutional Transformer (CCT-XS) for Optical Character Recognition (OCR), deprecation path for LPRNet, and two-factor plate verification.

---

## 1. Overview and Rationale

The previous OCR (Optical Character Recognition) engine, LPRNet (License Plate Recognition Network), exhibited structural character confusion on Vietnamese license plates (`D` versus `0`, `H` versus `1` or `S`). Squeezing crops to a 24-pixel height destroyed vertical stroke features, and training on synthetic wireframe fonts degraded real-world recognition. Furthermore, reading motorcycle plates required 300 lines of heuristic preprocessing (4-point perspective unwarping, Canny edge detection, Otsu thresholding, and horizontal line slicing).

CCT-XS (Compact Convolutional Transformer - Extra Small) replaces LPRNet via the `fast-plate-ocr` library. It uses a hybrid convolutional backbone coupled with self-attention transformer layers, allowing native recognition of both single-line (automobile) and dual-line (motorcycle) license plates without manual geometric slicing or edge-detection heuristics.

---

## 2. Model File Inventory

The model and configuration files were bundled locally into the repository to guarantee offline execution:

* **`Models_ocr/cct_xs.onnx`** (3.34 MB):  
  Exported Open Neural Network Exchange (ONNX) weights for CCT-XS v2 Global. Runs via ONNX Runtime using the `CPUExecutionProvider`.
* **`Models_ocr/cct_xs_plate_config.yaml`** (1.72 KB):  
  Configuration file defining character token vocabulary, image normalization parameters, and regional classification heads.

---

## 3. Code Modifications in `lpr_engine.py`

### A. Implementation of `CCTXSRecognizerAdapter`
**Lines 614 to 642:**
The previous placeholder implementation returned a mock `'LINE'` string. It was replaced with an active adapter wrapping `fast_plate_ocr.LicensePlateRecognizer`:

```python
# CCT-XS compact convolutional transformer adapter (Fast Plate OCR)
class CCTXSRecognizerAdapter(BaseOCR):
    def __init__(self, model_path: str = "Models_ocr/cct_xs.onnx", config_path: str = "Models_ocr/cct_xs_plate_config.yaml"):
        from fast_plate_ocr import LicensePlateRecognizer
        if os.path.exists(model_path) and os.path.exists(config_path):
            self.recognizer = LicensePlateRecognizer(
                onnx_model_path=model_path,
                plate_config_path=config_path,
                device='cpu'
            )
        else:
            self.recognizer = LicensePlateRecognizer(
                hub_ocr_model='cct-xs-v2-global-model',
                device='cpu'
            )

    def recognize(self, image_crop: np.ndarray, debug_prefix: str = None):
        if image_crop is None or image_crop.size == 0:
            return [{'text': '', 'confidence': 0.0}]

        preds = self.recognizer.run(image_crop, return_confidence=True)
        if not preds or not preds[0].plate:
            return [{'text': '', 'confidence': 0.0}]

        pred = preds[0]
        char_probs = pred.char_probs
        conf = float(np.mean(char_probs)) if char_probs is not None else float(getattr(pred, 'region_prob', 0.95))
        return [{'text': pred.plate, 'confidence': round(conf, 4)}]
```

* **Line 618-623**: Verifies if local ONNX and YAML files exist in `Models_ocr/`. If found, loads them locally without making network calls.
* **Line 634**: Executes inference directly on the plate image array (`image_crop`) with `return_confidence=True`.
* **Line 639-641**: Extracts individual character probabilities (`char_probs`), computes the arithmetic mean character confidence, and returns standardized OCR dictionary entries (`{'text': ..., 'confidence': ...}`).

### B. Factory Loader Routing
**Lines 645 to 660:**
Updated `get_ocr_adapter()` to register CCT-XS aliases and make it the primary default OCR option:

```python
def get_ocr_adapter(name: str = "cct_xs", **kwargs) -> BaseOCR:
    name_clean = name.lower().strip()
    if name_clean in ["cct_xs", "cct", "fast_plate", "fast_plate_ocr"]:
        return CCTXSRecognizerAdapter(**kwargs)
    ...
```

### C. Crop Margin Adjustment
**Lines 732 to 743:**
Inside `_crop_and_ocr()`, bounding box padding logic was updated to prevent surrounding vehicle bumper and mudguard noise from entering the transformer:

```python
is_lprnet = isinstance(self.ocr, LPRNetAdapter)
is_cct_xs = isinstance(self.ocr, CCTXSRecognizerAdapter)
for idx, d in enumerate(detections):
    x1, y1, x2, y2 = map(int, d['box'])
    h, w, _ = img.shape
    if is_lprnet or is_cct_xs:
        pad_w = 4
        pad_h = 4
    else:
        pad_w = max(20, int((x2 - x1) * padding_percent))
        pad_h = max(15, int((y2 - y1) * padding_percent))
```

* Traditional PaddleOCR required wide margins (`padding_percent=0.20`, 15-20 pixels) so DBNet could find text borders.
* CCT-XS requires a tight crop around the license plate border. Padding was fixed to 4 pixels (`pad_w = 4, pad_h = 4`).

### D. Bypass of Morphological Enhancement
**Line 761:**
Inside `_crop_and_ocr()`, the adaptive enhancement check was updated:

```python
if is_blurry and self.enable_preprocessing and not (is_lprnet or is_cct_xs):
```

* Bypasses CLAHE (Contrast Limited Adaptive Histogram Equalization), bilateral filtering, and morphological erosion for CCT-XS. Transformer attention heads natively resolve low contrast without destructive artificial pixel erosion.

---

## 4. Code Modifications in `main.py`

### A. Model Startup Configuration
**Lines 24 to 47:**
Updated backend model loading from baseline `best.onnx` to production `balanced_best_yolo11n.onnx` and instantiated CCT-XS:

```python
# Primary detector and Fast Plate OCR initialization
model_path = "Models_yolo/balanced_best_yolo11n.onnx"
if not os.path.exists(model_path):
    model_path = "Models_yolo/best.onnx"
primary_detector = YOLODetector(model_path, target_classes=['plate', 'license_plate'], confidence_threshold=0.45)

from lpr_engine import get_ocr_adapter
ocr = get_ocr_adapter("cct_xs")
pipeline = LPRPipeline(
    detector=primary_detector, 
    ocr=ocr,
    enable_preprocessing=False
)

# Registered campus vehicle database (maps student RFID card ID to authorized license plate)
REGISTERED_VEHICLES = {
    "STU_1001": "51G74372",
    "STU_1002": "59D242069",
    "STU_1003": "43D199999",
    "STU_1004": "29H133979",
    "STU_1005": "61H76854",
    "STU_1006": "51G77777",
    "STU_1007": "30A69770",
}
```

### B. Enforcing Two-Factor Plate Matching
**Lines 160 to 198:**
The previous logic opened the barrier whenever OCR text satisfied generic Vietnamese regular expression syntax, allowing any vehicle to enter with any valid card. Replaced with two-factor verification:

```python
# Two-factor verification: match OCR text against student registered plate
if registered_plate and ocr_text != "NONE_RECOGNIZED":
    match_res = VietnamPlateValidator.match_plate(ocr_text, registered_plate, max_distance=1)
    if match_res['match']:
        gate_status = "GATE_OPEN"
        mode = pipe_try_level
        try_level = pipe_try_level
        audit_flag = False
        flag_reason = None if match_res['match_type'] == 'exact' else f"Fuzzy match (distance: {match_res['distance']}) against registered plate {registered_plate}."
    else:
        gate_status = "ACCESS_DENIED"
        mode = "MISMATCH"
        try_level = pipe_try_level
        audit_flag = True
        flag_reason = f"Vehicle mismatch: card registered to {registered_plate}, detected {ocr_text}."
```

* **Exact Match (distance = 0)**: Barrier opens normally (`GATE_OPEN`).
* **Fuzzy Match (distance = 1)**: Barrier opens with an audit log record for security tracking.
* **Mismatch (distance > 1)**: Card is valid, but the vehicle is not registered to that card. Barrier stays closed (`ACCESS_DENIED`).
* **Unreadable Plate**: Barrier opens under student card fallback with an audit warning (`GATE_OPEN_FLAGGED`).

---

## 5. Documentation in `MINI_REPORT_LPR_RESEARCH.md`

Appended **Entry #6** (lines 499 to 585) following established project reporting standards:
* Documented root-cause failure points of LPRNet (24px spatial squeeze, wireframe synthetic font).
* Documented comparative benchmark metrics across all 16 test images.
* Recorded steady-state OCR CPU latency reduction from 264.2 ms (LPRNet) and 2,400 ms (PaddleOCR) down to 31.8 ms (CCT-XS).

---

## 6. How to Run and Verify

Execute commands using the project virtual environment:

### Batch Benchmark Across 16 Test Images:
```powershell
.\venv_models\Scripts\python.exe benchmarks/run_benchmarks.py test_images --model Models_yolo/balanced_best_yolo11n.onnx --ocr cct_xs --output_dir Outputs/cct_xs_output
```

### Single Image Inference:
```powershell
.\venv_models\Scripts\python.exe infer.py test_images/bike_test_10.jpg --model Models_yolo/balanced_best_yolo11n.onnx --ocr cct_xs
```

### Start FastAPI Gate Access Service:
```powershell
.\venv_models\Scripts\python.exe main.py
```

