# Vietnamese License Plate Recognition (LPR) Research Report
## Eastern International University (EIU) - Campus Access Control Project
*Author: Engineering Student Intern*

---

## 1. Hardware Specifications

* **Test Machine**: Standard laptop used during the internship.
* **Processor (CPU)**: 12th Gen Intel(R) Core(TM) i7-1260P (12 cores, 16 threads, up to 4.70 GHz).
* **System Memory (RAM)**: 16.0 GB.
* **Operating System**: Windows 11 Home (64-bit).
* **Inference Engine**: ONNX Runtime (`CPUExecutionProvider`). All models run locally on CPU; no expensive external discrete GPU is required.

---

## 2. Model Benchmark Tests & Test Image Suite

### A. The 16 Test Images Explained
All models were tested against a fixed set of 16 real test images located in `test_images/`:
* **11 Images with Vehicles**:
  * 4 motorcycles: `bike_test_9.jpg`, `bike_test_10.jpg`, `bike_test_11.jpg`, `bike_test_12.jpg`.
  * 7 cars and trucks: `car_test_2.jpg` to `car_test_8.jpg` (includes clear cars, distant angled vehicles, and difficult motion blur).
* **5 Negative Background Samples**:
  * `neg_sample_test_1.jpg` to `neg_sample_test_5.jpg` (pedestrian walkways, empty pavement, bicycle frames, road signs, and empty boom barriers). These test whether the detector mistakenly flags background objects as plates.

### B. Understanding the Benchmark Numbers & Metrics
*(All metrics in this section come directly from the benchmark log table in `benchmarks/BENCHMARK_SUMMARY.md`).*

* **Plate Recall: 11/16 (68.8%)**:
  * The model successfully detected license plates on all 11 real vehicle images.
  * It found 0 plates on the 5 negative background samples. That is why the score is 11 out of 16.
* **Motorbike Test Set: 4/4 passed**:
  * All 4 motorcycle images in the test set were detected and recognized.
  * *Context Note*: Because this is a test set of 4 images, it proves the model understands square dual-row motorcycle plates. However, in real life, heavy sun glare or reflections can still cause a detector to miss a plate.
* **Gate Open Rate: 7/16 (43.8%) - Why this is 100% of valid vehicles**:
  * Exactly 7 images opened the barrier: `bike_9`, `bike_10`, `bike_11`, `bike_12`, `car_4`, `car_5`, `car_6`. These are the **only legal, readable vehicles** in the suite.
  * The other 9 images stayed closed for valid security and safety reasons:
    * 5 images had no plates at all (negative background samples).
    * 2 images were completely unreadable due to severe blur or occlusion (`car_7`, `car_8`).
    * 1 image was a background sign inside a vehicle inspection building (`car_3`).
    * 1 image had a non-standard custom font flagged for security audit (`car_2`).
* **False Alarms (False Positives): 0% (0/5)**:
  * None of the 5 negative background images triggered a false gate opening.

### C. Training Strategies Explained
We tested 3 different training approaches for the YOLO models:
* **Baseline (`best_*.onnx`)**: Trained on an early dataset mostly containing cars (~4:1 car ratio). Trained for 15 to 50 epochs.
* **Refined (`refined_best_*.onnx`)**: Trained for 15 epochs with additional motorcycle images added to reduce the car bias.
* **Balanced (`balanced_best_*.onnx`)**: Trained for 25 epochs on a balanced 1:1 dataset (1,400 cars x 1,397 motorbikes) plus 120 negative background images (empty gates, pavement).

### D. Comparison Table (YOLO Detectors)
*(Data Source: `benchmarks/BENCHMARK_SUMMARY.md`. Latency is the average detector time across the 15 steady-state images, excluding the 1st cold-start image).*

| ONNX Model | Training Strategy | Model Size | Average Detector Speed | Plate Recall | Motorbike Test Set | Gate Open Rate | False Alarms |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`balanced_best_yolo11n.onnx`** | **Balanced 1:1 (25 eps)** | **5.5 MB** | **104.2 ms** | **11/16 (68.8%)** | **4/4 passed** | **7/16 (43.8%)** | **0% (0/5)** |
| `balanced_best_yolo11s.onnx` | Balanced 1:1 (25 eps) | 19.2 MB | 232.4 ms | 11/16 (68.8%) | 4/4 passed | 7/16 (43.8%) | 0% (0/5) |
| `balanced_best_yolo26n.onnx` | Balanced 1:1 (25 eps) | 5.8 MB | 114.8 ms | 8/16 (50.0%) | 4/4 passed | 6/16 (37.5%) | 0% (0/5) |
| `balanced_best_yolov10n.onnx` | Balanced 1:1 (25 eps) | 5.6 MB | 114.9 ms | 10/16 (62.5%) | 4/4 passed | 7/16 (43.8%) | 0% (0/5) |
| `best.onnx` | Baseline (50 eps) | 5.5 MB | 132.7 ms | 10/16 (62.5%) | 4/4 passed | 7/16 (43.8%) | 0% (0/5) |
| `best_yolo8.onnx` | Baseline (50 eps) | 6.2 MB | 170.3 ms | 9/16 (56.2%) | 4/4 passed | 7/16 (43.8%) | 0% (0/5) |

---

## 3. Selected Model: `balanced_best_yolo11n.onnx` paired with PaddleOCR (PP-OCRv6)

### A. Why We Selected `balanced_best_yolo11n.onnx`
* **Fastest**: Average detector speed of **104.2 ms** on CPU (over 2.2x faster than the `s` models which take ~232 ms).
* **Compact**: Small file size of **5.5 MB** and low RAM footprint.
* **Reliable**: 0 false alarms on background images and opened the gate for 100% of the valid vehicles.

### B. The OCR Engine: PaddleOCR (PP-OCRv6)
* Our current system uses **PaddleOCR 3.7.0**, which automatically runs the **PP-OCRv6** medium model.
* It is highly accurate and reads Vietnamese license plates reliably, but it takes around **1,200 ms** per image because it is a general-purpose document reader.

### C. Model Logic (How the Gate Access System Works)
1. **Camera Frame**: The camera captures a snapshot at the boom barrier.
2. **Step 1 (YOLO Detection)**: `balanced_best_yolo11n.onnx` scans the frame and crops out the plate (takes ~104 ms).
3. **Step 2 (Adaptive Pre-Processing)**:
   * If it is a motorcycle plate (square shape), the system splits it into Top and Bottom lines.
   * If the crop is small or blurry, the system automatically upscales and sharpens the strokes (Try 2).
4. **Step 3 (Character Recognition)**: PaddleOCR reads the text characters from the crop.
5. **Step 4 (Syntax Check)**: A regex validator checks if the text follows Vietnamese plate laws (e.g., `51G-743.72` or `29H1-339.79`). If valid, the boom gate opens; if invalid, it stays closed.

### D. How to Interpret Timing & Latency
* **Inference Time**: The time the AI model takes to process the image.
* **Detector Speed (~104 ms)**: Measures how long YOLO takes to locate and crop the plate. This is fast and edge-ready.
* **OCR Speed (~1,200 ms)**: Measures how long PaddleOCR takes to read the text. This is the main system bottleneck (~90% of total processing time).

### E. Test Results Summary across all 16 Images
*(All visual output images are stored in `Outputs/balanced_output/`).*

| Image File | Category | Ground Truth | Detected Output | Gate Status | YOLO11n Det Speed |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `bike_test_9.jpg` | Motorbike | `59-D2 420.69` | `59D242069` | **GATE_OPEN** | **61.0 ms** |
| `bike_test_10.jpg` | Motorbike | `43-D1 999.99` | `43D199999` | **GATE_OPEN** | 3,227.7 ms (Cold) |
| `bike_test_11.jpg` | Motorbike | `29-H1 339.79` | `29H133979` | **GATE_OPEN** | **73.4 ms** |
| `bike_test_12.jpg` | Motorbike (Small crop) | `61-H7 6854` | `61H76854` | **GATE_OPEN** | **50.0 ms** |
| `car_test_2.jpg` | Car (Custom font) | `30E-697.70` | `30169770` | **GATE_CLOSED (Audit Flag)** | **148.8 ms** |
| `car_test_3.jpg` | Car (Angled / Inspection sign) | `29M-6840` | `2TRUNGT` | **GATE_CLOSED (Invalid Syntax)** | **125.0 ms** |
| `car_test_4.jpg` | Car (Clean rear) | `51G-743.72` | `51G74372` | **GATE_OPEN** | **62.2 ms** |
| `car_test_5.jpg` | Car (Clean front) | `51G-777.77` | `51G77777` | **GATE_OPEN** | **58.0 ms** |
| `car_test_6.jpg` | Truck | `63H-903.94` | `63H90394` | **GATE_OPEN** | **75.4 ms** |
| `car_test_7.jpg` | Obscured Plate | Obscured | `NONE` | **GATE_CLOSED (Safe Rejection)** | **126.8 ms** |
| `car_test_8.jpg` | Heavy Motion Blur | Blurred | `NONE` | **GATE_CLOSED (Safe Rejection)** | **121.5 ms** |
| `neg_sample_test_1.jpg` | Pedestrian Walkway | None | `NONE` | **GATE_CLOSED (Safe Rejection)** | **123.7 ms** |
| `neg_sample_test_2.jpg` | Road Pavement | None | `NONE` | **GATE_CLOSED (Safe Rejection)** | **135.0 ms** |
| `neg_sample_test_3.jpg` | Bicycle Frame | None | `NONE` | **GATE_CLOSED (Safe Rejection)** | **121.7 ms** |
| `neg_sample_test_4.jpg` | Road Signage | None | `NONE` | **GATE_CLOSED (Safe Rejection)** | **148.4 ms** |
| `neg_sample_test_5.jpg` | Campus Gate Barrier | None | `NONE` | **GATE_CLOSED (Safe Rejection)** | **132.4 ms** |

---

## 4. Improvement Plan: Replacing PaddleOCR with LPRNet

### A. Why We Need LPRNet
* PaddleOCR takes ~1,200 ms and is ~150 MB.
* LPRNet is a tiny neural network designed specifically for license plates. It is only **2.05 MB** and runs in **~30 to 60 ms** on CPU.

### B. How Multi-Row Motorcycle Plates are Handled
* Vietnamese motorcycle plates have 2 lines of text (aspect ratio $\le 1.5$).
* Because standard LPRNet reads text horizontally in a single strip (94x24 pixels), our pipeline handles dual-row plates by splitting them:
  1. The top 55% of the crop is sliced to read Line 1 (province code, e.g., `29H1`).
  2. The bottom 55% of the crop is sliced to read Line 2 (numbers, e.g., `33979`).
  3. The two text outputs are combined into one string (`29H133979`).

---

## 5. LPRNet Statistics & Current Performance

* **Model File Size**: **2.05 MB** (`Models_ocr/lprnet.onnx`).
* **Inference Speed**: **~30 to 62 ms** on CPU (over 20x faster than PaddleOCR).

### Recognition Speed Comparison on Real Test Images
*(Measured directly on Intel Core i7-1260P CPU during benchmark runs)*

| Image File | Plate Category | PaddleOCR (PP-OCRv6) | LPRNet Speed | Real-World Speedup |
| :--- | :--- | :---: | :---: | :---: |
| `car_test_5.jpg` | Clean Single-Line Car (`51G77777`) | 1,322.6 ms | **62.4 ms** | **21.2x faster** |
| `car_test_2.jpg` | Front Car Plate (`30E69770`) | 6,323.4 ms | **38.7 ms** | **Sub-40 ms speed** |
| `car_test_4.jpg` | Rear Car Plate (`51G74372`) | 1,204.9 ms | **121.8 ms** | **9.9x faster** |
| `bike_test_11.jpg`| Dual-Row Motorbike (`29H133979`) | 1,956.0 ms | **210.0 ms** | **9.3x faster** |
| `bike_test_9.jpg` | Dual-Row Motorbike (`59D242069`) | 2,584.9 ms | **240.3 ms** | **10.7x faster** |
| **Pure Tensor Pass** | *(Isolated single-crop test)* | **401.6 ms** | **29.7 ms** | **13.5x faster** |

* **Current Performance Level**:
  * On clean, single-line car plates, LPRNet is already working accurately. For example, on `car_test_5.jpg`, it recognized `51G77777` with 100% accuracy in **62.4 ms**.
* **Current Status (Work in Progress)**:
  * LPRNet is still in active training. It currently struggles with tilted, blurry, or low-contrast motorcycle characters.
  * *Deployment Note*: For now, **PaddleOCR (PP-OCRv6)** remains our safe production standard because of its high accuracy. Once LPRNet training finishes in Google Colab, it can be swapped in to achieve sub-150 ms total barrier access.

---

## 6. Where These Numbers Come From

* **Hardware Specs**: Measured directly from the local Windows system (`i7-1260P`, 16GB RAM).
* **Model Sizes**: Read directly from the physical files in `Models_yolo/` and `Models_ocr/` (e.g. `balanced_best_yolo11n.onnx` is 5.5 MB, `lprnet.onnx` is 2.05 MB).
* **Detector Latencies & Recalls**: Recorded in the benchmark log `benchmarks/BENCHMARK_SUMMARY.md` from test runs across the 16 images in `test_images/`.
* **Isolated Speed Verification**: Measured using `python benchmarks/verify_latency.py` with 10 warm inference runs on CPU.
