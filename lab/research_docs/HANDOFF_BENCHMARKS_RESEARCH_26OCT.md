# Comprehensive Project Handoff & Research Transition Document

**Date:** 2026-10-26  
**Branch:** `main`  
**Execution Environment:** `.\venv_models\Scripts\python.exe`  
**Target Scope:** Vietnamese Automated License Plate Recognition (ALPR) & Campus Boom Barrier Access Control at Eastern International University (EIU)  
**Primary Focus for Next Session:** Repository Documentation Touch-Up, Visual & Structural Polish, Architecture Diagrams, and Production Readme Facelift

---

## 1. Executive Summary & Core Milestones Completed

### A. Compact Convolutional Transformer (CCT-XS) Integration
* **Architectural Replacement**: Permanently replaced legacy PaddleOCR (PP-OCRv6) and LPRNet with `cct-xs-v2-global-model` (`1.8 MB` ONNX model via `fast-plate-ocr`).
* **Performance Gain**: OCR inference latency dropped from `1,200 - 2,850 ms` (PaddleOCR) down to **`12.5 - 51.7 ms`** (mean **`31.8 ms`**) on edge CPU without discrete GPU hardware.
* **Unified Dual-Row Reading**: Completely eliminated fragile geometric horizontal slicing heuristics; CCT-XS natively reads both 1-line automobile and 2-line square motorcycle plates in a single forward pass.
* **Geographical Security Head**: Utilizes the native regional classification head ($P(\text{Vietnam})$) to block foreign plates (e.g. Brazil plate `EXK066M` rejected with 99.8% Brazil probability).

### B. Adaptive Visual Scaling & HUD Overhaul
* **The Problem**: Fixed pixel font/line constants caused Heads-Up Display (HUD) banners and bounding box tags to cover up to 70% of low-resolution vehicle thumbnails ($< 380 \times 260\text{ px}$), while high-resolution captures ($> 1500\text{ px}$) had hairline-thin outlines.
* **Code Implementation in `lpr_engine.py`**:
  * **Dynamic Box Thickness**: $\text{box\_thickness} = \max(1, \text{int}(\text{round}(\min(w, h) / 320.0)))$.
  * **Dynamic Font Scale**: $\text{tag\_font\_scale} = \max(0.24, \min(1.0, \min(w, h) / 1000.0))$.
  * **Micro-Thumbnail Status Bar**: For images smaller than $380 \times 260\text{ px}$, collapses into a single-line ultra-compact $\le 20\text{ px}$ status bar (`[PLATE] | [STATUS] ([RES])`) instead of the two-line banner.
  * **Adaptive Tag Flip**: Bounding box labels flip below the plate box if the top edge is pressed against the HUD.

### C. Multi-Fleet Benchmark Suite (151 Real-World Images)
* **Execution across 6 Distinct Fleets**:
  * `Car / car1` (35 images)
  * `Car / car1_defective` (2 images)
  * `Car / car2` (29 images)
  * `Car / car2_defective` (4 images)
  * `Motor / motor1` (72 images)
  * `Motor / motor1_defective` (9 images)
* **Comprehensive 151-Image Scorecard**:
  * **Passed (Gate Opened)**: 81 images (syntax valid, high confidence, regional match).
  * **Rejected (Gate Closed)**: 19 images (defective, unverified syntax, foreign plates).
  * **No Detection ($0$ Boxes)**: 51 images (micro-thumbnails $< 160\text{ px}$ from Facebook CDN downloads or extreme camera distance).
  * **Crops Generated**: 143 cropped plate snippets.
  * **Defective Rejection Safety**: `car1_defective` achieved **100% safe rejection** (0 false gate triggers).

### D. Automated ID Tagging & 4-Bucket Folder Routing
* **Updated `benchmarks/run_benchmarks.py`**:
  * Added `--prefix` auto-inference mapping folders to clean prefixes: `C1`, `C1D`, `C2`, `C2D`, `M1`, `M1D`.
  * Every image gets an ID (`C1-01`, `C1-02`, etc.) prepended in the console summary, output image filenames, and markdown reports.
  * Auto-routes annotated output images into 4 organized sub-buckets under `Outputs/khai_test/<category>/`:
    * `passed/`: Syntax-verified Vietnamese plates that opened the gate.
    * `rejected/`: Damaged, defective, foreign, or unverified plates safely closed.
    * `no_detection/`: Frames where 0 boxes were found.
    * `crops/`: Cropped license plate snippets prefixed with the matching image ID.

### E. Markdown Table Realignment (`benchmarks/BENCHMARK_SUMMARY.md`)
* Resolved table rendering corruption caused by mismatched historical column counts:
  * **Section 8**: Preserved early 18-image tests (12 columns).
  * **Section 8.1**: Intermediate adaptive enhancement tests (13 columns).
  * **Section 10**: Brand new dedicated table for the 151-image Multi-Fleet Benchmark (14 columns: `Timestamp | ID | Test Image | Try Level | Resolution | Boxes | Box Conf | Plate Text | Valid Format | Mean Conf | Min Char | Region (Prob) | Det (ms) | OCR (ms)`).
* Verified 100% column alignment in VS Code Markdown Preview (`Ctrl + Shift + V`).

### F. Vietnamese Regulatory Insight (Circular 24/2023 Two-Letter Plates)
* **Observation**: Image `M1-46` (`59VA24003`) was recognized by CCT-XS with 99.3% mean confidence and 99.1% Vietnam regional probability, but was flagged as `(REJECTED)`.
* **Root Cause**: `VietnamPlateValidator.MOTO_PATTERN` strictly required `[A-Z][0-9]` (e.g. `59D2...`).
* **Legal Alignment**: Under Circular 24/2023/TT-BCA (and Circular 58/2020 for $<50\text{cc}$ vehicles), Vietnamese motorcycle identification plates (*biển số định danh*) officially use **two consecutive letters** (`[A-Z]{2}`, such as `59-VA` in Gò Vấp, HCMC).
* **Ready Patch**: Update `MOTO_PATTERN` to `r'^[1-9][0-9]([A-Z][0-9]|[A-Z]{2})[0-9]{4,6}$'`.

---

## 2. Directory Code Inventory & Architecture Map

### A. Root Directory (`./`)
* **`lpr_engine.py`**:
  * Core pipeline orchestrator with modular adapters (`YOLODetector`, `PaddleOCRAdapter`, `CRNNAdapter`, `LPRNetAdapter`, `CCTXSRecognizerAdapter`, `LPRPipeline`).
  * Implements dynamic visual scaling for bounding boxes, tags, and micro-thumbnails.
  * Contains 3-tier escalation engine: Pass 1 (Fast 640px) $\to$ Pass 2 (Adaptive Mild Enhancement) $\to$ Pass 3 (Deep Zoom 1024px).
* **`infer.py`**:
  * Single-image command-line inference script with dual-pass measurement (cold-start vs steady-state).
* **`main.py`**:
  * FastAPI REST service for boom barrier access control (`/process_plate`, `/health`, `/stats`).
* **`simulate_rfid.py`**:
  * Multi-factor verification simulator combining RFID student card scans with camera plate recognition.
* **`requirements.txt`**:
  * Pinned dependencies (`ultralytics`, `fast-plate-ocr`, `onnxruntime`, `opencv-python`, `fastapi`, `uvicorn`).

### B. Core Preprocessing & Validation (`lpr_core/`)
* **`lpr_core/preprocessor.py`**:
  * `PlatePreprocessor.attenuate_glare()`: Retinex specular glare attenuation in YCrCb color space for tropical sunlight and headlight glare ($Y \ge 225$).
  * `PlatePreprocessor.mild_enhance_for_transformer()`: CLAHE contrast balancing (clipLimit 1.5), 2x2 morphological stroke reinforcement, and unsharp masking without harsh binarization.
* **`lpr_core/validator.py`**:
  * `VietnamPlateValidator`: Syntax verification against `AUTO_PATTERN` and `MOTO_PATTERN`.
  * Levenshtein string distance scoring and structural template matching (`check_format_similarity`).

### C. Benchmarks & Testing (`benchmarks/`)
* **`benchmarks/run_benchmarks.py`**:
  * Batch evaluation runner supporting `--model`, `--ocr`, `--output_dir`, and `--prefix`.
  * Automatically categorizes outputs into `passed/`, `rejected/`, `no_detection/`, and `crops/`.
  * Appends structured evaluation telemetry to Section 10 of `BENCHMARK_SUMMARY.md`.
* **`benchmarks/BENCHMARK_SUMMARY.md`**:
  * Master performance ledger recording historical and multi-fleet benchmark metrics.

### D. Documentation & Training Assets
* **`MINI_REPORT_CCT_XS_RESEARCH.md`**:
  * 8-section technical research document covering executive summaries, architectural comparisons, production flowcharts, code implementation blocks, edge hardware specs, edge case failures, and the Google Colab fine-tuning plan.
* **`README.md`**:
  * Repository entry-point guide with workflow descriptions and benchmark results.
* **`ColabNotebookTemplate/finetune_cct_xs.ipynb`**:
  * Turnkey 15-cell Google Colab notebook for GPU fine-tuning CCT-XS on Vietnamese datasets.

---

## 3. Comprehensive Analysis of CCT-XS Drawbacks

For documentation, defense presentations, and academic reporting, the limitations of CCT-XS must be clearly understood:

1. **Fixed Input Geometry ($128 \times 64\text{ px}$)**:
   * Long rectangular single-line car plates ($4.7:1$ aspect ratio) are squashed into a $2:1$ tensor, compressing character spacing and increasing confusion between narrow letters (`1`, `I`, `T`).
2. **Rigid Output Length Head (`max_plate_slots = 10`)**:
   * Uses fixed positional classification heads; cannot decode plates with $>10$ characters without model retraining.
3. **No Spatial Transformer Network (STN)**:
   * Lacks projective homography / rectification layers; fails under severe perspective yaw angles ($> 45^\circ$, as in `car_very_angled.jpg`).
4. **Limited Parameter Capacity (~1.8 MB)**:
   * As an extra-small model, it cannot hallucinate or reconstruct sub-pixel strokes on micro-thumbnails ($< 40 \times 20\text{ px}$, as in `car_test_3.jpg`).
5. **Pre-Trained International Font Bias**:
   * Pretrained on global distributions; exhibits minor confusion on subtle Vietnamese font ambiguities (`0` vs `D`, `8` vs `B`, `5` vs `S`).
6. **Sensitivity to Hard Binarization**:
   * Convolutional patch embeddings rely on smooth pixel gradients; degraded by harsh Otsu black-and-white thresholding.
7. **Complete Upstream Dependency**:
   * Recognizer only; cannot locate text in a full frame and relies 100% on YOLO bounding box accuracy.

---

## 4. Immediate Roadmap for the Next Conversation

When starting the new conversation, the primary focus is **touching up documentation to make it sleek, clean, modern, and publication-ready**:

### Priority 1: `README.md` Overhaul
* Reorganize into a clean, modern GitHub repository layout with high-density technical sections.
* Add a visual architecture flowchart (Mermaid diagram) showing the 3-tier escalation engine and 4-bucket directory routing.
* Integrate the new 151-image Multi-Fleet Benchmark Scorecard.
* Document the camera physical positioning constraints (1.2–1.5m height, 6–12mm lens, CPL optical filter).

### Priority 2: Legal Syntax Upgrade (`lpr_core/validator.py`)
* Discuss and apply the Circular 24/2023 update to `MOTO_PATTERN`:
  ```python
  MOTO_PATTERN = re.compile(r'^[1-9][0-9]([A-Z][0-9]|[A-Z]{2})[0-9]{4,6}$')
  ```
* Re-verify image `M1-46` (`59VA24003`) to demonstrate transition from `REJECTED` to `PASSED`.

### Priority 3: `MINI_REPORT_CCT_XS_RESEARCH.md` Final Polish
* Ensure formatting adheres strictly to `/document-general` (concise, high-density, no fluff, plain English first with bracketed technical details).
* Add Section 9 summarizing the 151-image real-world campus fleet evaluation and the Circular 24/2023 discovery.

---

## 5. Quick Verification Commands for the Next Session

```powershell
# 1. Verify working environment and dependencies
.\venv_models\Scripts\python.exe -c "import ultralytics, onnxruntime, fast_plate_ocr, cv2; print('All core libraries functional!')"

# 2. Test run single-image inference
.\venv_models\Scripts\python.exe infer.py "test_images/car_test_2.jpg" --model Models_yolo/balanced_40eps_yolo11n.onnx --ocr cct_xs

# 3. Test multi-fleet batch runner (e.g. car1_defective)
.\venv_models\Scripts\python.exe benchmarks/run_benchmarks.py "test_images/CAR MOTOR/Car/car1_defective" --model Models_yolo/balanced_40eps_yolo11n.onnx --ocr cct_xs --output_dir "Outputs/khai_test/car1_defective"
```

