# Engineering Change Log and Architectural Evolution

All notable engineering milestones, deprecations, and architectural evolutions for the Vietnamese Automated License Plate Recognition (ALPR) system are documented in this file.

---

## [2026-10-26] - Multi-Fleet Benchmark & Vietnamese Legal Alignment

### Added Code
- `benchmarks/run_benchmarks.py`:
  - `infer_prefix(path_str)`: Auto-maps target folders to standard fleet prefixes (`C1`, `C1D`, `C2`, `C2D`, `M1`, `M1D`).
  - 4-bucket directory creator and routing logic saving outputs to `passed/`, `rejected/`, `no_detection/`, and `crops/`.
  - Sequential ID tagging (`C1-01`, `M1-46`, etc.) prepended to console logs, output image filenames, and `BENCHMARK_SUMMARY.md`.
- `lpr_core/validator.py`:
  - Circular 24/2023/TT-BCA motorcycle syntax support: added `[A-Z]{2}` series to `VietnamPlateValidator.MOTO_PATTERN`.
  - Expanded `check_format_similarity` templates to include `DDLLDDDD`, `DDLLDDDDD`, and `DDLLDDDDDD`.
- `lpr_engine.py`:
  - Resolution-adaptive bounding box thickness: `box_thickness = max(1, int(round(min(w, h) / 320.0)))`.
  - Resolution-adaptive tag font scale: `tag_font_scale = max(0.24, min(1.0, min(w, h) / 1000.0))`.
  - Single-line ultra-compact status bar for micro-thumbnails (`< 380x260 px`): `[PLATE] | [STATUS] ([RES])`.
  - Bounding box tag repositioning: flips label below box if top edge conflicts with HUD banner.

### Adjusted Code
- `infer.py`:
  - Simplified CLI argument parser to require a direct positional `image_path` argument.
  - Eliminated silent model weight fallbacks; directly raises `FileNotFoundError` if image or model path does not exist.
- `lpr_engine.py`:
  - Cleaned all informal comments and slang into concise, single-line `#` comments.
  - Converted deprecated adapter error handlers in `get_ocr_adapter` to raise formal `NotImplementedError`.
- `lpr_core/preprocessor.py`:
  - Cleaned docstrings and standardized single-line `#` comments across all image filtering functions.
- `main.py`:
  - Removed unused `PaddleOCRAdapter` import; consolidated `get_ocr_adapter` import at top level.

### Removed Code
- `lpr_engine.py`:
  - Removed ~110 lines of dead commented-out `CRNNAdapter` and `LPRNetAdapter` class implementations.
  - Removed silent `.pt` fallback check inside `YOLODetector.__init__`.
- `benchmarks/run_benchmarks.py`:
  - Removed silent fallback from `best.onnx` to `best.pt`.
- `ColabNotebookTemplate/`:
  - Replaced ambiguous filenames (`15oct_benchmark_YOLOs_copy.ipynb`, `benchmark_YOLOs.ipynb`, etc.) with semantic model-parameter workflow notebooks (`01_train_yolo_balanced_1to1_25_40ep.ipynb`, `02_finetune_cct_xs_vietnam.ipynb`, etc.).

---

## [2026-10-23] - Compact Convolutional Transformer (CCT-XS) Integration

### Added Code
- `lpr_engine.py`:
  - `CCTXSRecognizerAdapter`: Wrapper for `fast_plate_ocr.LicensePlateRecognizer` operating on `Models_ocr/cct_xs.onnx` (1.8 MB).
  - Per-character softmax confidence extraction: `plate_char_probs = pred.char_probs[:plate_len]`, computing `confidence` (mean) and `min_confidence` (min).
  - Scale-adaptive bounding box padding: `pad_w = max(3, min(12, int((x2 - x1) * 0.04)))` and `pad_h = max(3, min(12, int((y2 - y1) * 0.04)))` (clamped between 3 px and 12 px).
  - Multi-vehicle perspective selection (`LPRPipeline._select_primary()`): implements `sort_by='front'` (bottom-most $y_2$ coordinate for boom gates), `sort_by='area'` (closest vehicle for side toll booths), and `sort_by='confidence'`.
  - Regional gating (`LPRPipeline._is_strictly_valid()`): hard blocks candidates predicting non-Vietnam origin with $\ge 50\%$ probability.
- `lpr_core/preprocessor.py`:
  - `PlatePreprocessor.attenuate_glare()`: YCrCb luminance isolation, localized Gaussian illumination field estimation, and Retinex illumination division for specular glare hot spots ($Y \ge 225$) covering 0.5% to 40% of crop area.
  - `PlatePreprocessor.mild_enhance_for_transformer()`: Pass 2 enhancement chaining glare attenuation, bicubic scaling to $\ge 128 \times 64$, mild CLAHE (`clipLimit=1.5`), soft $2 \times 2$ morphological stroke repair, and unsharp masking ($1.3 \times \text{sharp} - 0.3 \times \text{blur}$).

### Adjusted Code
- `lpr_engine.py`:
  - `get_ocr_adapter()`: Set `cct_xs` as the default production OCR adapter.
  - `LPRPipeline.process()`: Added character confidence threshold checks (`ocr_conf >= 0.70` and `ocr_min_conf >= 0.60`) to Pass 1 verification gate before escalating.

### Removed Code
- `lpr_engine.py`:
  - Removed legacy fixed 20% bounding box padding for CCT-XS crops.
  - Removed 20 px artificial white border padding (`cv2.copyMakeBorder`) from transformer enhancement pipeline to prevent attention token distortion.
  - Removed geometric dual-line horizontal splitting heuristics for CCT-XS passes.

---

## [2026-10-21] - Perspective Homography & Contour Analysis

### Added Code
- `lpr_engine.py`:
  - `order_quad_points(pts)`: Deterministically orders quadrilateral vertices clockwise into Top-Left, Top-Right, Bottom-Right, and Bottom-Left using $(x + y)$ sum and $(y - x)$ difference.
  - `compute_plate_quad_and_confidence(image_crop)`: Otsu and Canny contour extraction computing composite score:
    $$C_{\text{contour}} = 0.25 S_{\text{fill}} + 0.30 S_{\text{AR}} + 0.30 S_{\text{angle}} + 0.15 S_{\text{lum}}$$
  - `canonical_perspective_unwarp(image_crop, pts)`: 4-point projective homography transformation via `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`.
  - `deskew_and_slice_plate()`: Row luminance thresholding (`row_means >= plate_thresh`) to isolate bright license plate surfaces from dark motorcycle rear mudguards prior to slicing.

### Adjusted Code
- `lpr_engine.py`:
  - Raised default YOLO detector confidence threshold to `0.45`, eliminating background false alarms.
  - Motorbike horizontal slice ratio updated to 0-52% top slice and 48-100% bottom slice.

### Removed Code
- `lpr_engine.py`:
  - Removed duplicate `h, w = image_crop.shape[:2]` extractions in `_recognize_no_dbnet()` and `_recognize_split_lines()`.
  - Removed unused variable bindings in `CRNNAdapter` and `LPRNetAdapter`.

---

## [2026-10-15] - Dataset Balancing & Latency Decomposition

### Added Code
- `lpr_core/validator.py`:
  - `VietnamPlateValidator.clean_and_correct()`: Strips non-alphanumeric symbols (hyphens, dots, rivets) before regex evaluation.
  - Automated Levenshtein edit distance and structural template similarity scoring (`check_format_similarity()`).
- `lpr_core/preprocessor.py`:
  - Adaptive resolution recovery in `enhance_for_ocr()`: automatically triggers $2.5\times$ bicubic upscaling and morphological erosion ($2 \times 2$ kernel) when crop width $< 110\text{ px}$, height $< 60\text{ px}$, or blur score $< 150.0$.
- `lpr_engine.py`:
  - `get_unique_filepath()`: Non-overwriting file naming using clean integer suffix increments (`_1.jpg`, `_2.jpg`).
  - Translucent 2-line Heads-Up Display (HUD) banner with dynamic text scaling loop preventing horizontal clipping.
- `ColabNotebookTemplate/`:
  - Balanced 1:1 dataset merger: downsampled car training dataset to 1,400 images matching 1,397 motorcycle images.
  - Negative sample injection: integrated 120 empty campus backgrounds with 0-byte label files.

### Adjusted Code
- `ColabNotebookTemplate/01_train_yolo_balanced_1to1_25_40ep.ipynb`:
  - Fixed variable scoping bug (`if s == "train":` corrected to `if split == "train":`), enabling dataset sample limiting.

### Removed Code
- `lpr_core/validator.py`:
  - Removed hardcoded character replacement dictionaries in compliance with project rules.
- `lpr_engine.py`:
  - Removed faulty regex `^(.*)_(\d+)$` from `get_unique_filepath()` that was corrupting image base stems.

---

## [2026-10-14] - Full ONNX Runtime Migration

### Added Code
- `lpr_engine.py`:
  - `YOLODetector`: ONNX Runtime CPU execution provider wrapper.
  - `PaddleOCRAdapter`: Modular PP-OCRv6 adapter supporting `full`, `no_dbnet`, and `split_lines` modes.
  - 3-tier escalation coordinator: Pass 1 (640px) $\to$ Pass 2 (Enhancement) $\to$ Pass 3 (1024px Deep Zoom).
- `benchmarks/`:
  - `benchmark_yolo_onnx.py`: Automated full-pipeline ONNX detector benchmark matrix.
  - `test_ocr.py`: Isolated OCR crop visual slicing tool.
  - `compare_preprocessing.py`: Side-by-side diagnostic script.

---

## Architectural Deprecation Stubs

### 1. PaddleOCR (PP-OCRv6 Full Pipeline)
- **Status**: Deprecated for production edge deployment; retained only as an offline high-latency baseline reference.
- **Reason for Deprecation**:
  - Computational Latency: Dual-stage DBNet text detection + SVTR recognition requires `1,200 - 2,850 ms` per crop on edge CPU, exceeding the 2.0-second real-time access control threshold.
  - Resource Footprint: Requires `1.8 - 2.5 GB` RAM and extensive C++ runtime dependencies.
  - Slicing Dependency: Cannot natively decode two-line motorcycle plates without heuristic horizontal splitting.

### 2. LPRNet (Lightweight Spatial Convolutional Network)
- **Status**: Permanently Deprecated.
- **Reason for Deprecation**:
  - Complete Failure on Dual-Row Plates: Structurally limited to 1-line horizontal sequences; collapsed completely on Vietnamese square motorcycle plates.
  - CTC Loss Collapse: Prone to character collapse and hallucinations under minor perspective tilt (consistently confusing `D` with `0`, and `H` with `1` or `S`).
  - Rigid Input Dimensions: Squashing plates to fixed 94x24 pixels degraded vertical aspect ratios.

### 3. CRNN (Convolutional Recurrent Neural Network)
- **Status**: Deprecated.
- **Reason for Deprecation**:
  - High Error Rate on 2-Line Plates: Required manual pre-slicing and lacked spatial attention mechanisms needed for varying Vietnamese license fonts. Replaced by CCT-XS vision transformer.
