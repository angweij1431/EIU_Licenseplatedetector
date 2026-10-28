# Phase 1 Release Notes: Production Vision Library (`eiu_lpr`)

**Date**: 2026-10-28  
**Branch**: `feature/production`  
**Package**: `eiu_lpr-0.1.0-py3-none-any.whl`  
**Target Environment**: Linux Edge Worker (CerberusEdge), Python 3.12, Edge CPU Inference  

---

## 1. Executive Summary

Phase 1 refactors the research pipeline into an installable, in-memory Python wheel fulfilling all blocking integration requirements for CerberusEdge.

### Core Architectural Changes
1. **100% In-Memory API**:
   - Accepts frames directly in RAM as NumPy arrays (`np.ndarray`, `uint8`, shape `[H, W, 3]`).
   - Default color space is **RGB** (native to camera streams). Optional `color_format="BGR"` converts automatically for OpenCV workflows.
   - Zero filesystem I/O (`cv2.imwrite`, `cv2.imread`, and automatic `Outputs/` directory generation removed).
   - Zero console output (`print()` calls eliminated to prevent Personally Identifiable Information / PII leakage).
2. **Decoupled Functional APIs**:
   - `detect(image, detection_threshold=None)`: Fast 640px bounding box detection for temporal multi-frame tracking.
   - `read_plate(image, bbox)`: Proportional 4% padding (clamped 3–12px), mild adaptive enhancement, CCT-XS OCR, and Circular 24/2023 syntax validation.
   - `recognize(image)`: Combined detection and plate reading convenience method.
   - `warmup(iterations=2)`: Session priming for predictable latency.
3. **Pure ONNX Runtime Inference**:
   - The `ultralytics` package is eliminated from runtime execution.
   - Pure NumPy implementation for letterbox resizing, normalization, coordinate un-letterboxing, and Non-Maximum Suppression (NMS).
   - Retains the exact same trained YOLO11n weights with identical accuracy (< 0.1 pixel coordinate variance).
4. **Structured JSON Telemetry**:
   - Returns typed dataclasses (`BoundingBox`, `PlateCandidate`, `PipelineResult`).
   - Outputs JSON-serializable dictionaries matching the CerberusEdge reference schema (`minimal=True` or full evidence mode).
5. **Smart Path Resolution**:
   - Model paths can be passed as absolute paths or relative paths. The library automatically resolves relative paths against the project root if run from subdirectories.

---

## 2. Package Directory Layout

```
EIU_YOLO_PaddleOCR_Licenseplatedetection/
├── pyproject.toml                 # Package configuration (opencv-python-headless)
├── run_tests.py                   # Automated test suite (10/10 tests)
├── demo.py                        # Interactive verification demo
├── dist/
│   └── eiu_lpr-0.1.0-py3-none-any.whl  # Installable distribution wheel
│
├── src/eiu_lpr/                   # THE FACTORY: Clean Production Package
│   ├── __init__.py                # Public API exports and version constants
│   ├── pipeline.py                # LPRPipeline orchestrator
│   ├── detector.py                # Pure ONNX Runtime YOLO detector
│   ├── recognizer.py              # In-memory CCT-XS OCR adapter
│   ├── preprocessor.py            # Crop padding, Retinex glare attenuation, CLAHE
│   ├── validator.py               # Circular 24/2023 normalization & regex
│   ├── models.py                  # Dataclasses & JSON serialization
│   └── exceptions.py              # Typed custom exceptions
│
├── models/                        # THE ASSETS: Frozen release weights
│   ├── detector.onnx              # balanced_skew_robust_yolo11n.onnx (10.9 MB)
│   ├── cct_xs.onnx                # Compact Convolutional Transformer (1.8 MB)
│   ├── cct_xs_plate_config.yaml   # OCR alphabet and geometry configuration
│   └── checksums.sha256           # Cryptographic SHA-256 hashes
│
└── docs/
    └── PHASE1_RELEASE_NOTES.md    # Master Phase 1 documentation
```

---

## 3. Cryptographic Checksums (SHA-256)

Release model assets in `models/` verified against `models/checksums.sha256`:
* `detector.onnx`: `6aa3fa4ce22d711d3c39216568ebbc45c7843b57fcfcfae774627fa962bb28a5`
* `cct_xs.onnx`: `8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44`
* `cct_xs_plate_config.yaml`: `0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6`

---

## 4. Telemetry Schema Specification

### Minimal Reference Schema (`result.to_dict(minimal=True)`)
Matches the Edge backend reference schema exactly:
```json
{
  "schema_version": "1.0",
  "request_id": "req-001",
  "status": "OK",
  "candidates": [
    {
      "plate": "30A69770",
      "raw_plate": "30A69770",
      "confidence": 0.9969,
      "detection_confidence": 0.7569,
      "bbox": [178, 234, 242, 266]
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 179.33
}
```

### Full Evidence Schema (`result.to_dict(minimal=False)`)
Includes character breakdown and regulatory compliance:
```json
{
  "schema_version": "1.0",
  "request_id": "req-001",
  "status": "OK",
  "candidates": [
    {
      "plate": "30A69770",
      "raw_plate": "30A69770",
      "confidence": 0.9969,
      "detection_confidence": 0.7569,
      "bbox": [178, 234, 242, 266],
      "is_valid": true,
      "ocr_min_confidence": 0.9805,
      "character_details": [
        {"char": "3", "confidence": 0.9996},
        {"char": "0", "confidence": 0.9805},
        {"char": "A", "confidence": 1.0},
        {"char": "6", "confidence": 0.9999},
        {"char": "9", "confidence": 0.9999},
        {"char": "7", "confidence": 0.9999},
        {"char": "7", "confidence": 0.9955},
        {"char": "0", "confidence": 0.9999}
      ],
      "region": "Vietnam",
      "region_prob": 1.0,
      "pass_level": "enhanced_crop"
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 179.33
}
```

---

## 5. Automated Test Verification Summary

Ran `run_tests.py` using `unittest` (all 10 tests passed in 4.5s):

| Test ID | Description | Status |
| :--- | :--- | :--- |
| `test_01` | In-memory RGB array inference (reads `30A69770`, valid=True, confidence > 0.90) | **PASSED** |
| `test_02` | BGR color format support (automatic conversion from OpenCV frames) | **PASSED** |
| `test_03` | Zero disk I/O verification (0 new files created on disk during inference) | **PASSED** |
| `test_04` | Split APIs consistency (`detect()` + `read_plate()` produces identical results to `recognize()`) | **PASSED** |
| `test_05` | Pure ONNX runtime verification (`ultralytics` not imported in `sys.modules`) | **PASSED** |
| `test_06` | Error handling (invalid array dimensions, types, or shapes raise `InvalidInputError`) | **PASSED** |
| `test_07` | Configuration safety (missing model files or invalid thresholds raise typed exceptions) | **PASSED** |
| `test_08` | Session warmup (primes execution providers to guarantee steady-state latency) | **PASSED** |
| `test_09` | Telemetry versioning (custom `model_version` and `schema_version` forwarded cleanly) | **PASSED** |
| `test_10` | Dynamic threshold override (per-call `detection_threshold` parameter works on the fly) | **PASSED** |

Sample Fleet Verification:
* `car_test_2.jpg`: `30A69770` (Single-line automobile, valid)
* `bike_test_10.jpg`: `43D199999` (Two-line motorcycle, valid)
* `bike_test_11.jpg`: `29H133979` (Two-line motorcycle, valid)
* `car_test_5.jpg`: `51G77777` (Single-line automobile, valid)
* `neg_sample_test_1.jpg`: `[]` (0 false positives on negative scenes)
