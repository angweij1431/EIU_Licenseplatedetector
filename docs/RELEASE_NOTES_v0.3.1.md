# Release Notes: v0.3.1

Release Date: November 26, 2026
Model Version: `alpr-v0.3` (unchanged)
Schema Version: `1.0` (unchanged)
Package Version: `0.3.1`

## Overview

Release 0.3.1 is a backward-compatible patch release for the `eiu_lpr` Automated License Plate Recognition (ALPR) library. It fixes a crop preprocessing regression observed on sharp, downscaled overhead motorcycle feeds and documents future architecture options for vehicle class attribution, lane Region of Interest (ROI) filtering, and motion-burst frame selection without altering the production JavaScript Object Notation (JSON) output schema or model weights.

## Key Changes

### 1. Preprocessor Trigger and Color-Channel Fix
- **Blur-only adaptive enhancement trigger** (`src/eiu_lpr/pipeline.py`): Narrowed `enable_adaptive_enhancement` to run `PlatePreprocessor.mild_enhance_for_transformer()` only when a plate crop is genuinely blurred (`blur_score < 150.0`), removing the previous small-crop dimension trigger (`crop_w < 110 or crop_h < 60`).
  - Field testing on sharp downscaled overhead motorcycle feeds (for example, `1024x576` frames with `blur_score > 7000`) showed that the 2x2 morphological erosion step distorted tight dual-line characters (for example, turning `67F77811` into `61F77811` and `94E117720` into `94G137720`).
  - Sharp crops now bypass morphological erosion and retain native Optical Character Recognition (OCR) accuracy at both native (`2880x1620`) and downscaled (`1024x576`) resolutions.
- chore fix: **Blue-Green-Red (BGR) to Red-Green-Blue (RGB) normalization before enhancement** (`src/eiu_lpr/pipeline.py`): When `enable_adaptive_enhancement=True` and `color_format="BGR"` are used, `LPRPipeline.read_plate()` now converts the crop from BGR (Blue-Green-Red) to RGB (Red-Green-Blue) before computing Laplacian blur scores and applying Retinex and Contrast Limited Adaptive Histogram Equalization (CLAHE) enhancement.

### 2. Documented Future Options (On Hold — Not Active in Schema `1.0`)
- Added Section 5 in `docs/INTEGRATION_HANDBOOK.md` recording copy-paste templates and evaluation notes for future extensions without complicating the core library:
  - **Vehicle class attribution (on hold)**: Recorded that the current `models/detector.onnx` (`alpr-v0.3`) is a single-class plate detector (`[1, 8400, 6]`, class `plate` only), and that off-the-shelf Common Objects in Context (COCO) nano detectors lose motorcycle recall from steep overhead camera angles (`0.005–0.097` on 3 of 4 overhead field shots). Documented both future options: zero-latency plate-inferred classification (Circular 24/2023 syntax plus bounding-box aspect ratio) or fine-tuning a lightweight overhead vehicle detector on target camera frames if retaining plate-less vehicle images is required.
  - **Lane ROI (Region of Interest) filtering and motion-burst selection (on hold)**: Added drop-in helper snippets for filtering detections inside a pixel ROI (Region of Interest) and ranking multi-shot motion-burst frames by `is_valid` and `ocr_min_confidence`.

### 3. Tests and Repository Hygiene
- Added regression tests in `tests/test_in_memory_api.py` verifying that JSON (JavaScript Object Notation) schema `1.0` keys remain unchanged and that `enable_adaptive_enhancement` skips sharp crops while triggering `enhanced_crop` on genuinely blurred crops.
- Updated `.gitignore` to exclude local field deployment photos (`test_images/actual*`) containing Personally Identifiable Information (PII) and Windows zone metadata files (`*:Zone.Identifier`).

## Upgrade Notes
- **Drop-in upgrade**: No changes to model files (`models/detector.onnx`, `models/cct_xs.onnx`, `models/cct_xs_plate_config.yaml`), JSON (JavaScript Object Notation) schema `1.0`, or public method signatures.
- Install the built wheel:
  ```bash
  pip install --upgrade dist/eiu_lpr-0.3.1-py3-none-any.whl
  ```

## Verification
- Automated Test Suite: 15 of 15 tests passing (`pytest tests/ -v`), including `tests/test_golden_dataset.py`.
- Model Checksums: Verified unchanged (`cd models && sha256sum -c checksums.sha256` -> `detector.onnx: OK`).
