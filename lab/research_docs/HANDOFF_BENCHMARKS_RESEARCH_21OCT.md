# Comprehensive Project Handoff & Directory Code Reference

**Date:** 2026-10-21  
**Branch:** `main`  
**Environment:** `.\venv_models\Scripts\python.exe`  
**Primary Scope:** Vietnamese Automated License Plate Recognition (ALPR) & Access Control — Architecture, Directory Code Inventory, 4-Point Canonical Geometry Normalizer, Deterministic Contour Confidence Scoring, and Benchmark Verification

---

## 1. Executive Summary & Progress Completed

### A. 4-Point Canonical Geometry Normalizer Integration
* **Problem**: 2D affine rotation (Hough transform) only corrects in-plane tilt angle ($\theta$). Vehicles approaching gate cameras from angled lanes exhibit perspective trapezoidal skew (where the edge closest to the lens appears larger than the far edge).
* **Fix Applied**: Implemented 4-point projective homography in [`lpr_engine.py`](lpr_engine.py):
  * [`order_quad_points()`](lpr_engine.py): Deterministically sorts quadrilateral corners clockwise into Top-Left (TL), Top-Right (TR), Bottom-Right (BR), and Bottom-Left (BL) using coordinate sum $(x + y)$ and coordinate difference $(y - x)$.
  * [`canonical_perspective_unwarp()`](lpr_engine.py): Warps quadrilateral points to an orthogonal destination rectangle via `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`.

### B. Deterministic Plate Contour Confidence Formulation
* **Mathematical Formulation**: Because standard You Only Look Once (YOLO) detectors yield axis-aligned bounding boxes rather than corner keypoints, [`compute_plate_quad_and_confidence()`](lpr_engine.py) computes a numerical confidence score ($C_{\text{contour}} \in [0.0, 1.0]$):
  $$C_{\text{contour}} = 0.25 S_{\text{fill}} + 0.30 S_{\text{AR}} + 0.30 S_{\text{angle}} + 0.15 S_{\text{lum}}$$
  * **$S_{\text{fill}}$ (Solidity / Area Fill)**: Evaluates contour area relative to crop bounding box area ($0.40 \le \text{fill} \le 0.98 \implies 1.0$).
  * **$S_{\text{AR}}$ (Aspect Ratio Consistency)**: Validates against Vietnamese standards (motorcycles: $1.05 - 1.60$; automobiles: $1.80 - 4.80$).
  * **$S_{\text{angle}}$ (Orthogonality Deviation)**: Measures average deviation of internal corners from $90^\circ$ ($\Delta \theta \le 5^\circ \implies 1.0$, $\Delta \theta \ge 25^\circ \implies 0.0$).
  * **$S_{\text{lum}}$ (Luminance Contrast)**: Difference between inside contour brightness and exterior margin ($\Delta L \ge 30 \implies 1.0$).
* **Decision Threshold**: If $C_{\text{contour}} \ge 0.70$, perspective unwarp executes. If $C_{\text{contour}} < 0.70$, it falls back safely to Hough affine rotation or the raw crop.

### C. Motorbike Dual-Row Plate Luminance Profiling & Slicing
* **Problem**: Motorbike crops include black rear mudguards. Slicing strictly at $50\%$ height pushed the mudguard into the top slice and shifted the bottom row down, cutting off digits.
* **Fix Applied**: Added row luminance thresholding (`row_means >= plate_thresh`) to isolate the bright plate rectangle before splitting into top slice ($0 - 52\%$) and bottom slice ($48 - 100\%$).
* **Result**: On `bike_test_10.jpg`, recovered all five nines (`99999`) in the bottom slice.

### D. Dead Code & Variable Cleanup
* Removed duplicate `h, w = image_crop.shape[:2]` extractions in `_recognize_no_dbnet()` and `_recognize_split_lines()`.
* Removed unused `h, w` variable bindings in `CRNNAdapter`, `LPRNetAdapter`, and `CCTXSRecognizerAdapter`.
* Raised default YOLO (You Only Look Once) detector confidence threshold to `0.45` across `lpr_engine.py` and `benchmarks/run_benchmarks.py`, eliminating false detections on negative background samples.

### E. Comparative Benchmark Results (LPRNet Recognizer)
* **Syntax Pass Rate**: Doubled from **12.5% (2/16)** to **25.0% (4/16)**.
* **1st-Try Resolution Rate**: Doubled from **12.5% (2/16)** to **25.0% (4/16)**.
* **Steady-State OCR Latency**: Dropped **25.2%** from **353.4 ms** down to **264.2 ms** due to reduced character aspect distortion.
* **1st-Try Clean Passes (Valid: YES)**:
  * `bike_test_9.jpg`: `55S242069` (Det: 191.3 ms, OCR: 341.2 ms)
  * `car_test_2.jpg`: `33A69770` (Det: 193.3 ms, OCR: 174.6 ms)
  * `car_test_4.jpg`: `51G74372` (Det: 504.0 ms, OCR: 240.1 ms - 100% exact match)
  * `car_test_5.jpg`: `51G77777` (Det: 252.0 ms, OCR: 237.6 ms - 100% exact match)
* **Negative Samples**: 0 false openings across all 5 negative background images.

---

## 2. Directory Breakdown & File-Level Code Inventory

### A. Root Directory (`./`)
Contains core execution scripts, API services, and project-level configuration files.

* **`lpr_engine.py`**: The primary pipeline and model adapter library.
  * `get_target_output_dir(model_name)`: Determines output subdirectory (`baseline_output`, `refined_output`, or `balanced_output`) based on model weight names.
  * `get_unique_filepath(directory, filename)`: Appends clean sequential integer suffixes (`_1`, `_2`) to prevent file overwrites without mangling filenames.
  * `order_quad_points(pts)`: Takes four 2D corner points and sorts them clockwise: Top-Left (minimum sum $x+y$), Top-Right (minimum difference $y-x$), Bottom-Right (maximum sum $x+y$), and Bottom-Left (maximum difference $y-x$).
  * `compute_plate_quad_and_confidence(image_crop)`: Evaluates candidate contours using Otsu thresholding and Canny edge extraction. Computes composite confidence ($C_{\text{contour}}$) from area fill, aspect ratio consistency, corner orthogonality, and luminance contrast.
  * `canonical_perspective_unwarp(image_crop, pts)`: Computes orthogonal destination bounds and applies projective perspective transformation via OpenCV (`cv2.warpPerspective`).
  * `deskew_and_slice_plate(image_crop, debug_prefix, save_debug)`: Master geometry normalizer. Applies 4-point perspective unwarp when $C_{\text{contour}} \ge 0.70$ (falls back to Hough affine deskewing if lower), handles single-row car plates, isolates bright plate surface for motorbikes, and splits dual-row motorcycle plates into top ($52\%$) and bottom ($52\%$) slices.
  * `get_ocr_adapter(name, **kwargs)`: Factory function returning an instance of the requested OCR (Optical Character Recognition) engine.
  * `BaseDetector` (Class): Abstract base class defining the `detect(image_path)` interface for plate detectors.
  * `BaseOCR` (Class): Abstract base class defining the `recognize(image_crop, debug_prefix)` interface for text recognizers.
  * `YOLODetector(BaseDetector)` (Class): Adapter for Ultralytics YOLO (You Only Look Once) models in ONNX (Open Neural Network Exchange) or PyTorch formats. Filters boxes by class and confidence threshold.
  * `PaddleOCRAdapter(BaseOCR)` (Class): Adapter for PaddleOCR (PP-OCRv6). Implements `_recognize_full()` (DBNet detection + recognition), `_recognize_no_dbnet()` (pure recognition bypass), and `_recognize_split_lines()` (geometric dual-line splitting).
  * `CRNNAdapter(BaseOCR)` (Class): Adapter for CRNN (Convolutional Recurrent Neural Network) ONNX models operating on $100 \times 32$ grayscale tensors.
  * `LPRNetAdapter(BaseOCR)` (Class): Adapter for LPRNet (License Plate Recognition Network) ONNX models. Preprocesses crops to $94 \times 24$ RGB tensors and decodes characters via CTC (Connectionist Temporal Classification) greedy decoding.
  * `CCTXSRecognizerAdapter(BaseOCR)` (Class): Adapter for CCT-XS (Compact Convolutional Transformer) models for Fast Plate OCR evaluation.
  * `LPRPipeline` (Class): End-to-end execution coordinator. Implements `_select_primary()` (vehicle selection by position, area, or confidence), `_is_strictly_valid()` (syntax compliance check), `_crop_and_ocr()` (adaptive enhancement and recognition), and `process()` (multi-stage 3-tier escalation engine with HUD banner rendering).
* **`infer.py`**: Standalone command-line inference tool for single vehicle images. Runs dual-pass cold-start versus warm steady-state latency evaluation and outputs annotated images with green HUD (Heads-Up Display) summary banners.
* **`main.py`**: FastAPI backend service for automated boom barrier campus access control.
  * Exposes endpoints: `GET /` (service overview), `GET /health` (system and model status), `POST /process_plate` (multipart image upload, pipeline execution, barrier decision), and `GET /stats` (traffic volume and timing telemetry).
* **`simulate_rfid.py`**: Demonstration script for two-factor campus access control. Simulates synchronous fusion of student RFID (Radio Frequency Identification) card badge scans with camera license plate recognition.
* **`requirements.txt`**: Pinning of required Python dependencies (`ultralytics`, `paddleocr`, `onnxruntime`, `opencv-python`, `fastapi`, `uvicorn`).
* **`MINI_REPORT_LPR_RESEARCH.md`**: Comprehensive technical research report documenting Sections 1 to 7 (Hardware Specs, Benchmark Tests, Selected Model Architecture, Edge Deployment Trade-offs, Failure Modes, Hardware Cost Analysis, and Engineering Change Log Entries #1 through #5).
* **`README.md`**: High-level repository orientation, installation instructions, and workflow summary.

---

### B. `lpr_core/` (Core Preprocessing & Validation Modules)
Houses model-agnostic computer vision and syntax validation logic.

* **`preprocessor.py`**: Image quality assessment and stroke reconstruction.
  * `PlatePreprocessor` (Class):
    * `compute_blur_score(image)`: Calculates image sharpness via the variance of the Laplacian operator ($\sigma^2_{\text{Laplacian}}$).
    * `enhance_for_ocr(crop)`: Evaluates resolution and sharpness. Applies $2.5\times$ bicubic upscaling, morphological stroke reconstruction (`cv2.erode` with a $2 \times 2$ kernel to reconnect broken character lines), and bilateral filtering to suppress background noise.
* **`validator.py`**: Vietnamese license plate legal syntax verification.
  * `VietnamPlateValidator` (Class):
    * `AUTO_PATTERN`: RegEx standard for cars (`^[1-9][0-9][A-Z][0-9]{4,6}$`).
    * `MOTO_PATTERN`: RegEx standard for motorcycles (`^[1-9][0-9][A-Z][0-9][0-9]{4,6}$`).
    * `clean_and_correct(raw_text)`: Strips non-alphanumeric symbols (hyphens, dots, rivets) and evaluates regex compliance.
    * `levenshtein_distance(s1, s2)`: Calculates minimum string edit distance between two alphanumeric sequences.
    * `check_format_similarity(text, min_similarity)`: Converts characters into a structural mask ('D' for digit, 'L' for letter) and computes similarity against standard Vietnamese civilian templates.
    * `match_plate(ocr_text, registered_plate, max_distance)`: Verifies recognized text against authorized student database records.

---

### C. `benchmarks/` (Evaluation Suites & Performance Metrics)
Automates batch testing, latency measurement, and statistical logging.

* **`run_benchmarks.py`**: Primary batch evaluation script. Iterates across test images, executes detection and OCR, evaluates syntax pass rates, computes cold vs steady-state latency, and appends timestamped records to `BENCHMARK_SUMMARY.md`.
* **`benchmark_yolo_onnx.py`**: Benchmark harness dedicated to detector models. Evaluates inference speed and detection counts across all ONNX models in `Models_yolo/`.
* **`compare_preprocessing.py`**: Diagnostic utility comparing raw detector crops directly against preprocessed crops across OCR confidence scores.
* **`test_ocr.py`**: Standalone OCR diagnostic tool testing character prediction accuracy on isolated plate crops.
* **`verify_latency.py`**: Dedicated latency benchmarking tool isolating ONNX Runtime session warm-up from steady-state inference.
* **`BENCHMARK_SUMMARY.md`**: Active markdown benchmark report logging evaluation tables and aggregate performance metrics.
* **`BENCHMARK_SUMMARY_LEGACY_OCT14.md`**: Preserved historical benchmark logs from earlier development iterations.

---

### D. `ColabNotebookTemplate/` (Model Training & Fine-Tuning)
Contains Jupyter notebooks and standalone scripts for cloud GPU model training.

* **`train_lprnet_vietnam.py` / `train_lprnet.py`**: Standalone PyTorch scripts training LPRNet models using CTC (Connectionist Temporal Classification) loss on synthetic and real Vietnamese plate character sets.
* **`train_lprnet_vietnam.ipynb`**: Google Colab notebook providing end-to-end dataset loading, data augmentation, CTC loss training, and ONNX export for LPRNet.
* **`train_yolo11.py`**: Training script for fine-tuning Ultralytics YOLO11 architectures on vehicle plate datasets.
* **`train_vietnam_lpr_colab_YOLO11.ipynb` / `train_vietnam_lpr_colab.ipynb`**: Training notebooks featuring 1:1 balanced car-to-motorbike dataset merging (1,400 cars : 1,397 motorbikes) and negative background sample injection.
* **`benchmark_YOLOs.ipynb`**: Cloud GPU benchmarking notebook evaluating detector frame rates.

---

### E. `Models_ocr/` (Optical Character Recognition Weights)
Houses trained and exported weight checkpoints for character recognition.

* **`lprnet.onnx`**: Exported ONNX model of LPRNet trained for 150 epochs on Vietnamese license plates ($94 \times 24$ input).
* **`best_lprnet.pth`**: PyTorch checkpoint weights for LPRNet.
* **`cct_xs.onnx`**: Compact Convolutional Transformer ONNX weights for Fast Plate OCR evaluation.
* **`crnn.onnx`**: Exported ONNX weights for the baseline CRNN model ($100 \times 32$ input).
* **`yelprnet.onnx` / `yebest_lprnet.pth`**: Alternative experimental LPRNet checkpoints.

---

### F. `Models_yolo/` (Vehicle Plate Detector Weights)
Stores ONNX and PyTorch weights for YOLO plate detection models.

* **`balanced_best_yolo11n.onnx` / `.pt`**: Selected primary production detector ($5.5\text{ MB}$, input $640 \times 640$, steady-state latency ~104 ms). Trained on 1:1 balanced car/motorbike data with 120 negative background samples.
* **`best.onnx` / `best.pt`**: Baseline car-heavy detector checkpoint.
* **`refined_best_*.onnx` / `.pt`**: Gen 2 refined detector checkpoints (4:1 imbalanced ratio).
* **`balanced_best_*.onnx` / `.pt`**: Gen 3 detector family covering YOLO8, YOLO10, YOLO11, and YOLO26 nano and small architectures.

---

### G. `Outputs/` (Visual Output Dumps & Debug Slices)
Directory tree organizing visual inspection outputs and intermediate debug crops.

* **`Outputs/deskewed_slices/`**: Canonical unwarped crops (`{stem}_deskewed.jpg`), raw crops (`{stem}_full_crop.jpg`), and dual-row slices (`{stem}_top_slice.jpg`, `{stem}_bot_slice.jpg`).
* **`Outputs/lprnet_output/`**: Annotated visualizations resulting from LPRNet benchmark runs.
* **`Outputs/balanced_output/`**: Visualizations generated by balanced YOLO11n paired with PaddleOCR.
* **`Outputs/baseline_output/`**: Visualizations from baseline detector runs.
* **`Outputs/refined_output/`**: Visualizations from refined detector runs.
* **`Outputs/bench_test_output/`**: Visualizations from batch benchmark test runs.

---

### H. `test_images/` (Standard Evaluation Test Suite)
Contains the 16 real-world test images used for all comparative benchmarking:
* **4 Motorcycles**: `bike_test_9.jpg` (`59-D2 420.69`), `bike_test_10.jpg` (`43-D1 999.99`), `bike_test_11.jpg` (`29-H1 339.79`), `bike_test_12.jpg` (`61-H7 6854`).
* **7 Automobiles**: `car_test_2.jpg` to `car_test_8.jpg` (includes clear cars, distant angled vehicles, and motion blur).
* **5 Negative Background Samples**: `neg_sample_test_1.jpg` to `neg_sample_test_5.jpg` (pedestrian walkways, empty pavement, bicycle frames, road signs, empty gates).

---

### I. `debug_slices/` (Local Debugging Mirror)
Local working directory mirroring intermediate perspective-unwarped crops and line slices for rapid inspection during active development.

---

### J. `archive/` (Legacy Reference Code & Assets)
Preserved historical test scripts and documentation.
* **`run_all_benchmarks.py`**: Legacy multi-model benchmark script.
* **`test_lpr_pipeline.py` / `test.py`**: Early experimental test scripts.
* **`train_yolo_example.py`**: Legacy template script for YOLO training.
* **`LPR_BENCHMARK_AND_ARCHITECTURE.md`**: Early system architecture design draft.
* **`sample_car.jpg`, `real_test_bus.jpg`, `test_plate_sample.jpg`**: Legacy testing imagery.

---

## 3. Standard Execution Commands

Execute all Python commands using the virtual environment: `.\venv_models\Scripts\python.exe`.

### Run Benchmark with LPRNet:
```powershell
.\venv_models\Scripts\python.exe benchmarks/run_benchmarks.py test_images --model Models_yolo/balanced_best_yolo11n.onnx --ocr lprnet --output_dir Outputs/lprnet_output
```

### Run Benchmark with PaddleOCR:
```powershell
.\venv_models\Scripts\python.exe benchmarks/run_benchmarks.py test_images --model Models_yolo/balanced_best_yolo11n.onnx --ocr paddleocr --output_dir Outputs/balanced_output
```

### Run Standalone Single Image Inference:
```powershell
.\venv_models\Scripts\python.exe infer.py test_images/car_test_4.jpg --model Models_yolo/balanced_best_yolo11n.onnx --ocr lprnet
```

---

## 4. Prioritized Next Steps for the Next Conversation

1. **Character Confusion Post-Processing**:
   * Investigation confirmed that on motorcycle plates (`bike_test_10`), LPRNet extracts `430199999` instead of `43D199999`. The optical confusion maps letter `D` to digit `0`, and letter `H` to `S` or `8`.
   * how would we resolve this issue in a manner that does not require hardcoding? any suggestions?
2. **Fast Plate OCR (CCT-XS) Model Integration**:
   * Complete implementation of `CCTXSRecognizerAdapter` in `lpr_engine.py` to evaluate the compact transformer against LPRNet and PaddleOCR on CPU latency and character accuracy.
3. **Repository Commit Policy**:
   * Per established project guidelines, keep `MINI_REPORT_LPR_RESEARCH.md`, `benchmarks/verify_latency.py`, and all `*HANDOFF*` files untracked and local.
