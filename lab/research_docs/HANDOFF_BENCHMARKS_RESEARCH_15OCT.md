# Comprehensive Project Handoff & Milestone Summary

**Date:** 2026-10-15  
**Branch:** `features/benchmarks-research`  
**Environment:** `.\venv_models\Scripts\python.exe`  
**Primary Scope:** Vietnamese License Plate Recognition (LPR) & Access Control — Engine Optimization, HUD Visualization, Cold vs. Warm Latency Decomposition, and Balanced Retraining

---

## 1. Executive Summary & What Was Done Today

### A. Alphanumeric Syntax Standardization & Latency Spike Resolution
- **Root Cause Identified**: Previous pipeline evaluated raw OCR outputs containing hyphens and dots (e.g. `59-D2 420.69`). Strict regex checks failed, causing the engine to escalate through all 3 try levels (Try 1 -> Try 2 -> Try 3). Because PaddleOCR was called 3 separate times on CPU (~4s each), latency ballooned to ~16 seconds!
- **Fix Applied**: Updated `VietnamPlateValidator.clean_and_correct` in `lpr_core/validator.py` to strip all non-alphanumeric characters, and updated vehicle regex standards:
  - Automobile: `^[1-9][0-9][A-Z][0-9]{4,6}$`
  - Motorcycle: `^[1-9][0-9][A-Z][0-9][0-9]{4,6}$`
- **Result**: Valid plates now exit cleanly on **1st Try**, dropping execution time to ~1.8s.

### B. Adaptive Character Stroke Reconstruction (8 vs 9 Resolution)
- **Problem**: On `bike_test_12.jpg`, the detected plate was only 49x37 pixels. Small pixel scale caused PaddleOCR to misread digit `8` as `9` (`59D242069` read as `59D242068`). High-frequency noise gave an artificially high Laplacian sharpness score (> 2600).
- **Fix Applied**: Added an automatic **Adaptive Trigger** in `PlatePreprocessor.enhance_for_ocr`:
  - Triggers if crop width < 110 px OR crop height < 60 px OR blur score < 150.
  - Automatically applies 2.5x bicubic upscaling and morphological stroke reconstruction (`cv2.erode` with a 2x2 rectangular kernel).
- **Result**: Restores stroke connectivity, correctly recognizing `59D242069` on Try 1 with near-zero latency overhead (~0.3 ms).

### C. Destination-Aware Output Routing & Filename Safety
- **Clean Folder Partitioning**:
- **Clean Folder Partitioning**:
  - `Outputs/baseline_output/`: Car baseline (`best.onnx`)
  - `Outputs/refined_output/`: Gen 2 imbalanced model (`refined_best_*.onnx`)
  - `Outputs/balanced_output/`: Gen 3 balanced models (`balanced_*.onnx`)
  - `Outputs/bench_test_output/`: Batch benchmark matrices
- **Filename Collision Fix**: Updated `get_unique_filepath()` in `lpr_engine.py`. Previous regex `^(.*)_(\d+)$` mistook image numbers (`car_test_3`) for run versions and incremented them to `car_test_4`. The fix keeps the base stem intact and cleanly appends version suffixes (`car_test_3_1.jpg`, `car_test_3_2.jpg`).

### D. Visual Heads-Up Display (HUD) Banner & Plate Tags
- **2-Line Translucent Green HUD**: Renders across the top edge of output images:
  - Line 1: `Model: <name> | Run #<n> | Status: <try_level> (<resolution>)`
  - Line 2: `Cold: <cold_total>ms (Det: <cold_det>ms) | Warm: <warm_total>ms (Det: <warm_det>ms)`
- **Dynamic Text Scaling**: Uses `cv2.getTextSize()` in a loop (`while font_scale > 0.28`) to guarantee text never clips horizontally or vertically.
- **Bounding Box Tags**: Renders plate text and confidence directly above the box, colored green if syntax passed, or orange if unverified.

### E. Latency Decomposition: Cold-Start vs. Steady-State
- **Cold-Start Session Boot (~10.1s)**: Initial model weight loading, ONNX Runtime session compilation, CPU memory arena allocation, and thread spawning.
- **Warm Steady-State Gate Speed (~2.88s)**:
  - **YOLO Detection Time**: **61 ms (~2.1% of total time)** — already edge-ready (> 16 FPS).
  - **PaddleOCR Recognition Time**: **2,828 ms (~97.9% of total time)** — the definitive system bottleneck.
- Proves mathematically that future optimization must focus on lightweight OCR (LPRNet/CRNN) rather than detector pruning.

### F. Colab Dataset Merging Bug Discovery & Balanced Retraining
- **Silent Scoping Bug**: In `15oct_benchmark_YOLOs copy.ipynb`, `if s == "train":` checked the lingering outer loop variable `s` (which was `"valid"`), causing `max_train_samples=1600` to be ignored. It loaded 5,778 cars vs. 1,397 motorbikes (a 4.1:1 imbalance).
- **The Fix**: Changed to `if split == "train":`.
- **Clean 1:1 Balance**: Downsampled car training set to **1,400 images** matching the **1,397 motorbike images** (a true 1.002:1 ratio).
- **Negative Sample Injection**: Added ~120 empty background images (empty parking gates, pavement, road surfaces) with empty 0-byte `.txt` files, completely discarding external labels to prevent people or cracks from being mapped to class 0 (`plate`).
- **Training Launch**: 25 epochs on Tesla T4 GPU with `patience=0` to prevent premature early stopping, FP16 AMP, and standard default hyperparameters for a controlled experiment.

---

## 2. Directory Tree & Current File State

```text
c:\Users\User\EIU_OITP\lpr_research\
│
├── ColabNotebookTemplate\
│   ├── 15oct_benchmark_YOLOs copy.ipynb # Active training notebook with fixed 1:1 merge & negative samples
│   ├── benchmark_YOLOs.ipynb
│   ├── train_vietnam_lpr_colab.ipynb
│   └── train_vietnam_lpr_colab_YOLO11.ipynb
│
├── Models_yolo\                         # Active ONNX detector models
│   ├── best.onnx                       # Baseline car-only model
│   ├── best_yolo10n.onnx
│   ├── best_yolo10s.onnx
│   ├── best_yolo11n.onnx
│   ├── best_yolo11s.onnx
│   ├── best_yolo26n.onnx
│   ├── best_yolo26s.onnx
│   ├── best_yolo8.onnx
│   ├── refined_best_yolo11s.onnx       # Gen 2 model (imbalanced 4:1)
│   └── (balanced_yolo26n.onnx)         # Gen 3 model (currently training in Colab)
│
├── Outputs\                            # Organized output visual dumps
│   ├── baseline_output\ (.gitkeep)     # Runs from best.onnx
│   ├── bench_test_output\ (.gitkeep)   # Batch benchmark test runs
│   ├── refined_output\ (.gitkeep)      # Runs from refined_best_*.onnx
│   └── balanced_output\ (.gitkeep)     # Automatically routed runs from balanced_*.onnx
│
├── benchmarks\
│   ├── BENCHMARK_SUMMARY.md            # Clean, active benchmark markdown table
│   ├── BENCHMARK_SUMMARY_LEGACY_OCT14.md # Preserved legacy benchmark history
│   ├── benchmark_yolo_onnx.py          # Automated full matrix evaluation
│   ├── run_benchmarks.py               # Primary pipeline benchmark
│   └── test_ocr.py                     # OCR diagnostic & slicing tool
│
├── lpr_core\
│   ├── preprocessor.py                 # Stroke reconstruction (cv2.erode) & bicubic upscaler
│   └── validator.py                    # Strict alphanumeric regex & template similarity
│
├── test_images\                        # 16 benchmark evaluation test images
│   ├── car_test_2.jpg to 8.jpg
│   ├── bike_test_9.jpg to 12.jpg
│   └── neg_sample_test_1.jpg to 5.jpg
│
├── MINI_REPORT_LPR_RESEARCH.md         # Technical report draft with the 5 core figures
├── HANDOFF_BENCHMARKS_RESEARCH_15OCT.md# This document
├── infer.py                            # Standalone inference with dual cold/warm latency reporting
├── lpr_engine.py                       # Core pipeline with 3-tier escalation, HUD overlay, & routing
└── main.py                             # FastAPI gate access control backend service
```

---

## 3. Core Technical Decisions & Rules Persisted

1. **Pure Alphanumeric Validation**:
   - Always strip hyphens and dots from OCR text before syntax checking.
   - Vietnamese vehicle plates follow: `^[1-9][0-9][A-Z][0-9]{4,6}$` (cars) and `^[1-9][0-9][A-Z][0-9][0-9]{4,6}$` (motorcycles).
2. **Cold-Start vs. Steady-State Distinction**:
   - Never report single-shot inference latency without labeling cold-start vs. warm steady-state.
   - Use `infer.py` with `measure_warm=True` to empirically record both numbers on the same image.
3. **Safe File Versioning**:
   - Preserve original base image filenames (`car_test_3`). Never strip trailing digits with regex. Append `_1`, `_2` cleanly.
4. **Negative Sample Integrity**:
   - Background images must NEVER contain readable license plates.
   - Matching label files must be 0-byte empty `.txt` files. Discard all external classes (pedestrians, cracks, etc.) to prevent class 0 contamination.
5. **Controlled Hyperparameters**:
   - When evaluating dataset balance improvements, keep optimizer and learning rate at standard defaults to isolate data curation as the independent variable.

---

## 4. Immediate Next Steps

1. **Retrieve the Trained Model**:
   - When Colab finishes 25 epochs, export to ONNX:
     ```python
     model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
     ```
   - Download the `.onnx` file and place it in `Models_yolo/` as `balanced_yolo26n.onnx` (or `balanced_yolo11s.onnx`).
2. **Execute Single-Image Verification**:
   ```powershell
   .\venv_models\Scripts\python.exe infer.py test_images/bike_test_11.jpg --model "Models_yolo/balanced_yolo26n.onnx"
   ```
   - Verify that output image and HUD banner are automatically saved into `Outputs/balanced_output/`.
3. **Execute Full Matrix Benchmark**:
   ```powershell
   .\venv_models\Scripts\python.exe benchmarks/benchmark_yolo_onnx.py
   ```
   - Check `benchmarks/BENCHMARK_SUMMARY.md` to record the official Gen 3 balanced scores across all 16 test images.

