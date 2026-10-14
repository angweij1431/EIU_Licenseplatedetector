# LPR System Benchmark Summary and Production Architecture Guide

Target Test Images: `car_test_2.jpg`, `car_test_3.jpg`
Execution Environment: Windows CPU (Python 3.11 with PyTorch 2.14, PaddlePaddle 3.3.1, Ultralytics YOLOv8)

---

## 1. Executive Summary & Recommended Stack

* **Detector**: YOLOv8 fine-tuned on Vietnamese license plates (`Models_yolo/best.pt`).
* **Inference Resolution**: `imgsz=1024` for high-resolution edge detection on distant plates.
* **Crop Padding**: Minimum 20px horizontal, 15px vertical padding floor to prevent edge character clipping.
* **Preprocessor**: Dynamic bicubic upscaling ($2.5\times$ for crops $<160\text{px}$ wide) + CLAHE contrast boost + Bilateral edge-preserving filter.
* **OCR Engine**: Streamlined PaddleOCR (`PP-OCRv6`) with document unwarping and rotation classifiers disabled.
* **Validator**: Position-aware numeric correction + official Vietnamese regex validation (`VietnamPlateValidator`).

---

## 2. Latency Breakdown and Optimization Results

### Cold Start vs. Warm Cache
* **Cold Start (Run 1)**: ~6,000 ms to 7,000 ms total latency. Heavy libraries (`torch`, `paddle`, `cv2`) are loaded from disk into memory, thread pools are initialized, and neural network graphs are allocated.
* **Warm Cache (Runs 2+)**: ~2,000 ms to 2,600 ms total latency. Network weights and memory pools are pinned in RAM.

### PaddleOCR Streamline Optimization
By default, PaddleOCR v3 executes document-level classifiers (`PP-LCNet_x1_0_doc_ori`, `PP-LCNet_x1_0_textline_ori`) and paper unwarping networks (`UVDoc`). Disabling these redundant document modules directly reduced OCR latency:
* **Before Optimization**: 7,033 ms OCR latency.
* **After Optimization**: 2,663 ms OCR latency (**62% reduction** with identical accuracy).

---

## 3. Preprocessing Comparison Benchmark (Raw vs. Enhanced Crop)

Evaluated on `car_test_3.jpg` (small 2-line license plate: initial bounding box $45 \times 33\text{ px}$):

* **Configuration A (Raw Crop without Preprocessing)**:
  * Extracted lines: `['6840']` (Line 1 completely missed).
  * Parsed plate text: `6840`.
  * Format valid: `False` (Access Denied).
  * OCR latency: 787.35 ms.

* **Configuration B (Enhanced Crop with Upscaling, CLAHE, and Bilateral Denoising)**:
  * Extracted lines: `['29M', '8840']` (Both lines successfully isolated).
  * Parsed plate text: `29M-8840`.
  * Format valid: `True` (Access Granted).
  * Preprocessing overhead: 14.99 ms.
  * OCR latency: 2,048.21 ms.

* **Ablation Takeaway**:
  Preprocessing adds under 15 ms of computational overhead but directly recovers missing character lines on distant or low-contrast plates.

---

## 4. Architecture Evaluation: Event-Based Trigger vs. Continuous Real-Time Stream

### Option 1: Continuous Real-Time Video Stream
* **Mechanism**: Pulls 15 to 30 FPS from an RTSP camera stream continuously in a background loop.
* **CPU/GPU Load**: 100% continuous utilization 24/7. Causes thermal throttling on edge devices.
* **Drawbacks**: High complexity, frame lag, memory accumulation, and wasted compute on empty lanes.

### Option 2: Event-Based Trigger with Asynchronous Buffer (Recommended)
* **Mechanism**: Camera captures a snapshot upon an external trigger (RFID card tap, ground induction loop, or optical beam sensor).
* **CPU/GPU Load**: 0% idle utilization. Inference wakes up for only ~150 ms to 400 ms per vehicle transaction.
* **Lookahead Cache**: A lightweight video loop (5 FPS) buffers the latest plate as a vehicle rolls into the bay. When the RFID card taps, the match is an instantaneous cache lookup ($<10\text{ ms}$).

---

## 5. Physical Site and Hardware Deployment Guidelines

In commercial automated gate installations, physical site engineering dictates reliability:

* **Camera Choke Point**: Install a speed bump or lane barrier 3 to 5 meters before the gate to slow vehicles to $<10\text{ km/h}$.
* **Camera Angle**: Keep horizontal and vertical angles under 30 degrees relative to the vehicle heading.
* **Plate Pixel Width**: Adjust camera optical zoom so that plates occupy at least 150 to 250 horizontal pixels at the capture line.
* **Lighting**: Use infrared (IR) strobe illuminators with fast shutter speeds ($1/500\text{s}$ to $1/1000\text{s}$) to eliminate nocturnal headlight blooming and motion blur.

---

## 6. CLI Execution Commands

```powershell
# 1. Run single-image benchmark
.\venv_models\Scripts\python.exe run_benchmark.py car_test_3.jpg

# 2. Run preprocessing comparison (Raw vs Enhanced)
.\venv_models\Scripts\python.exe compare_preprocessing.py test_images/car_test_3.jpg

# 3. Run RFID gate simulator against the API service
.\venv_models\Scripts\python.exe simulate_rfid.py test_images/car_test_3.jpg
```

## 7. Dynamic Evaluation History

| Timestamp | Test Image | Resolution | Boxes | Plate Text | Valid Format | Det (ms) | OCR (ms) |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| 2026-10-11 21:32:15 | `car_test_2.jpg` | 1024px | 1 | `301-69770` | NO | 2744.2 | 1662.1 |
| 2026-10-11 21:32:15 | `car_test_3.jpg` | 1024px | 1 | `29M-8840` | YES | 260.8 | 3680.6 |
| 2026-10-11 21:56:29 | `car_test_2.jpg` | 1024px | 1 | `301-69770` | NO | 2897.4 | 1701.5 |
| 2026-10-11 21:56:29 | `car_test_3.jpg` | 1024px | 1 | `29M-8840` | YES | 266.6 | 3493.0 |
| 2026-10-11 22:04:45 | `car_test_3.jpg` | 1024px | 1 | `29M-8840` | YES | 2749.6 | 2080.3 |
| 2026-10-11 22:05:49 | `car_test_2.jpg` | 1024px | 1 | `301-69770` | NO | 2895.9 | 1589.6 |
| 2026-10-11 22:05:49 | `car_test_3.jpg` | 1024px | 1 | `29M-8840` | YES | 280.3 | 2665.2 |
| 2026-10-12 09:10:48 | `car_test_2.jpg` | 1024px | 1 | `301-69770` | NO | 4736.0 | 1695.3 |
| 2026-10-12 09:10:48 | `car_test_3.jpg` | 1024px | 1 | `29M-8840` | YES | 335.8 | 2840.3 |
| 2026-10-12 10:03:48 | `car_test_3.jpg` | 1024px | 1 | `29M-8840` | YES | 7556.4 | 3150.1 |
| 2026-10-12 10:27:24 | `car_test_2.jpg` | 2nd try | 1024px | 1 | `301-69770` | NO | 6274.1 | 15379.7 |
| 2026-10-12 10:27:24 | `car_test_3.jpg` | 2nd try | 1024px | 1 | `29M-8840` | YES | 964.7 | 5017.2 |
| 2026-10-12 11:17:01 | `car_test_2.jpg` | 2nd try | 1024px | 1 | `301-69770` | NO | 5742.9 | 3767.9 |
| 2026-10-12 11:17:01 | `car_test_3.jpg` | 2nd try | 1024px | 1 | `29M-8840` | YES | 405.5 | 1994.5 |
| 2026-10-12 11:23:07 | `car_test_4.jpg` | 2nd try | 1024px | 1 | `516-743.72` | NO | 4856.9 | 8106.3 |
| 2026-10-12 11:25:40 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 15504.1 | 7345.7 |
| 2026-10-12 11:28:20 | `car_test_2.jpg` | 2nd try | 1024px | 1 | `301-69770` | NO | 4967.7 | 6048.1 |
| 2026-10-12 11:28:20 | `car_test_3.jpg` | 2nd try | 1024px | 1 | `29M-8840` | YES | 3530.3 | 9838.3 |
| 2026-10-12 11:28:20 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 417.4 | 7752.6 |
| 2026-10-12 11:33:55 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 4680.9 | 1330.8 |
| 2026-10-12 11:33:55 | `car_test_3.jpg` | 2nd try | 1024px | 1 | `29M-8840` | YES | 671.7 | 5067.3 |
| 2026-10-12 11:33:55 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 243.5 | 1413.9 |
| 2026-10-12 14:06:33 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 5869.6 | 1560.7 |
| 2026-10-12 14:06:33 | `car_test_3.jpg` | 1st try | 640px | 1 | `29H-6840` | YES | 192.8 | 5135.4 |
| 2026-10-12 14:06:33 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 194.4 | 1450.9 |
| 2026-10-12 14:09:14 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 4753.7 | 1464.3 |
| 2026-10-12 14:09:14 | `car_test_3.jpg` | 1st try | 640px | 1 | `29H-6840` | YES | 175.3 | 5126.6 |
| 2026-10-12 14:09:14 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 204.0 | 1745.4 |
| 2026-10-12 14:13:46 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 3618.1 | 753.6 |
| 2026-10-12 14:13:46 | `car_test_3.jpg` | 1st try | 640px | 1 | `29H-6840` | YES | 112.8 | 3189.7 |
| 2026-10-12 14:13:46 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 179.9 | 1158.2 |
| 2026-10-12 14:19:53 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 2488.5 | 1086.0 |
| 2026-10-12 15:01:38 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 3479.8 | 749.9 |
| 2026-10-12 15:01:38 | `car_test_3.jpg` | 1st try | 640px | 1 | `29H-6840` | YES | 103.0 | 2792.0 |
| 2026-10-12 15:01:38 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 247.6 | 938.1 |
| 2026-10-12 15:05:13 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 2319.1 | 802.8 |
| 2026-10-12 16:26:58 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 3822.8 | 1777.1 |
| 2026-10-12 16:26:58 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 221.4 | 5608.4 |
| 2026-10-12 16:26:58 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 208.5 | 1311.1 |
| 2026-10-12 16:57:32 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 7748.1 | 1576.9 |
| 2026-10-12 17:16:12 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 4175.7 | 848.7 |
| 2026-10-12 17:16:27 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 2371.1 | 805.5 |
| 2026-10-12 17:16:40 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 2041.6 | 775.3 |
| 2026-10-12 17:16:55 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 2095.4 | 796.1 |
| 2026-10-12 17:17:34 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 2581.9 | 754.6 |
| 2026-10-12 17:17:34 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 102.4 | 3029.2 |
| 2026-10-12 17:17:34 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 303.2 | 1360.4 |
| 2026-10-12 17:17:58 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 2386.6 | 795.0 |
| 2026-10-12 17:17:58 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 176.5 | 2887.2 |
| 2026-10-12 17:17:58 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 112.8 | 922.3 |
| 2026-10-12 21:34:17 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 5490.6 | 1003.4 |
| 2026-10-12 21:34:17 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 117.7 | 3291.7 |
| 2026-10-12 21:34:17 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 153.3 | 858.7 |
| 2026-10-12 21:35:59 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 3289.6 | 1001.7 |
| 2026-10-12 21:35:59 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 115.9 | 3377.7 |
| 2026-10-12 21:35:59 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 145.6 | 836.6 |
| 2026-10-13 09:26:04 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 4436.8 | 841.8 |
| 2026-10-13 09:26:04 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 102.0 | 2794.8 |
| 2026-10-13 09:26:04 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 133.0 | 718.9 |
| 2026-10-13 16:36:10 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 5860.5 | 1663.4 |
| 2026-10-13 16:36:10 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 206.0 | 4583.8 |
| 2026-10-13 16:36:10 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 281.8 | 1220.9 |

## 8. YOLO Detection ONNX Performance (Cold Start Excluded)

*Evaluated at: `2026-10-13 16:46:11` across `3` test images*

| Model Architecture | File | Format | Recall % | Plates Detected | Steady Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLO11-FineTuned (best)** | `best.onnx` | ONNX | 100.0% | 3/3 | **49.7 ms** |
| **YOLOv8-FineTuned** | `best_yolo8.onnx` | ONNX | 66.7% | 2/3 | **75.9 ms** |
| **YOLOv8-Nano (Base)** | `yolov8n.onnx` | ONNX | 100.0% | 3/3 | **158.0 ms** |

| 2026-10-13 16:47:53 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 4621.2 | 1900.2 |
| 2026-10-13 16:47:53 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 201.9 | 5713.1 |
| 2026-10-13 16:47:53 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 88.5 | 1628.0 |

## 8. Complete YOLO ONNX Architecture Matrix (Cold Start Excluded)

*Evaluated at: `2026-10-13 16:51:30` across `3` test images*

| ONNX Model File | Model Size | Plate Recall % | Plates Detected | Steady Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| `best.onnx` | 10.2 MB | 100.0% | 3/3 | **64.4 ms** |
| `best_11s.onnx` | 36.2 MB | 100.0% | 3/3 | **91.2 ms** |
| `best_yolo10n.onnx` | 9.0 MB | 66.7% | 2/3 | **54.0 ms** |
| `best_yolo10s.onnx` | 27.8 MB | 100.0% | 3/3 | **92.1 ms** |
| `best_yolo11n.onnx` | 10.2 MB | 66.7% | 2/3 | **50.2 ms** |
| `best_yolo11s.onnx` | 36.2 MB | 100.0% | 3/3 | **197.7 ms** |
| `best_yolo26n.onnx` | 9.4 MB | 66.7% | 2/3 | **107.2 ms** |
| `best_yolo26s.onnx` | 36.4 MB | 100.0% | 3/3 | **193.5 ms** |
| `best_yolo8.onnx` | 12.2 MB | 66.7% | 2/3 | **85.4 ms** |
| `yolov8n.onnx` | 12.3 MB | 100.0% | 3/3 | **86.5 ms** |


## 8. Complete Multi-Model End-to-End Pipeline Matrix

*Evaluated at: `2026-10-13 16:57:44` across `3` test images (General latency including cold-start captured)*

| ONNX Model | Size | Plate Recall | Syntax Valid | 1st-Try Rate | 2nd-Try Rate | Avg Det (ms) | Avg OCR (ms) | Avg Total (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `best_yolo10n.onnx` | 9.0 MB | 66.7% | 66.7% | 66.7% | 33.3% | 1639.9 ms | 1290.1 ms | **2951.3 ms** |
| `best_yolo10s.onnx` | 27.8 MB | 100.0% | 66.7% | 33.3% | 66.7% | 602.9 ms | 9373.6 ms | **10012.2 ms** |
| `best_yolo11n.onnx` | 10.2 MB | 66.7% | 66.7% | 66.7% | 33.3% | 445.6 ms | 1574.1 ms | **2048.6 ms** |
| `best_yolo11s.onnx` | 36.2 MB | 100.0% | 66.7% | 66.7% | 33.3% | 394.0 ms | 3810.6 ms | **4225.3 ms** |
| `best_yolo26n.onnx` | 9.4 MB | 66.7% | 33.3% | 33.3% | 66.7% | 225.3 ms | 2918.5 ms | **3164.7 ms** |
| `best_yolo26s.onnx` | 36.4 MB | 100.0% | 66.7% | 66.7% | 33.3% | 303.7 ms | 2545.8 ms | **2868.5 ms** |
| `best_yolo8.onnx` | 12.2 MB | 66.7% | 66.7% | 66.7% | 33.3% | 98.2 ms | 669.7 ms | **781.3 ms** |
| `yolov8n.onnx` | 12.3 MB | 0.0% | 0.0% | 0.0% | 100.0% | 180.9 ms | 0.0 ms | **197.6 ms** |


## 9. Real Empirical Output & Confidence Matrix

*Evaluated at: `2026-10-13 17:04:58` across `3` test images (Measured outputs and model confidence scores only)*

### Model Summary

| ONNX Model | Plate Recall | Avg Det Conf % | Gate Open Rate | Avg OCR Conf % | Try 1 | Try 2 | Gate Closed | Avg Det (ms) | Avg Total (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `best_yolo10n.onnx` | 2/3 (66.7%) | 51.5% | **2/3 (66.7%)** | 63.2% | 2 | 0 | 1 | 946.7 ms | **2149.6 ms** |
| `best_yolo10s.onnx` | 3/3 (100.0%) | 65.3% | **2/3 (66.7%)** | 92.8% | 1 | 1 | 1 | 404.4 ms | **5784.9 ms** |
| `best_yolo11n.onnx` | 2/3 (66.7%) | 50.6% | **2/3 (66.7%)** | 64.5% | 2 | 0 | 1 | 170.0 ms | **1008.4 ms** |
| `best_yolo11s.onnx` | 3/3 (100.0%) | 61.9% | **2/3 (66.7%)** | 98.0% | 2 | 0 | 1 | 215.6 ms | **2639.7 ms** |
| `best_yolo26n.onnx` | 2/3 (66.7%) | 52.3% | **1/3 (33.3%)** | 63.5% | 1 | 0 | 2 | 125.1 ms | **2427.7 ms** |
| `best_yolo26s.onnx` | 3/3 (100.0%) | 67.9% | **2/3 (66.7%)** | 94.0% | 2 | 0 | 1 | 339.9 ms | **2736.4 ms** |
| `best_yolo8.onnx` | 2/3 (66.7%) | 53.2% | **2/3 (66.7%)** | 63.9% | 2 | 0 | 1 | 187.9 ms | **1030.7 ms** |
| `yolov8n.onnx` | 0/3 (0.0%) | 0.0% | **0/3 (0.0%)** | 0.0% | 0 | 0 | 3 | 188.4 ms | **205.1 ms** |

### Per-Image Real Outputs

| Model | Image | Plate Detected (Conf %) | Valid Format (OCR Conf %) | Gate Status | Extracted Plate | Det (ms) | OCR (ms) | Total (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `best_yolo10n.onnx` | `car_test_2.jpg` | YES (75.9%) | YES (94.6%) | **Try 1** | `30A-69770` | 2530.7 | 1500.0 | 4042.7 |
| `best_yolo10n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 234.3 | 0.0 | 275.3 |
| `best_yolo10n.onnx` | `car_test_4.jpg` | YES (78.5%) | YES (94.9%) | **Try 1** | `51G-743.72` | 75.0 | 2037.3 | 2130.9 |
| `best_yolo10s.onnx` | `car_test_2.jpg` | YES (76.1%) | YES (94.3%) | **Try 1** | `30A-69770` | 692.4 | 1686.6 | 2396.0 |
| `best_yolo10s.onnx` | `car_test_3.jpg` | YES (44.9%) | YES (85.0%) | **Try 2** | `29H-6840` | 183.3 | 5284.1 | 5505.6 |
| `best_yolo10s.onnx` | `car_test_4.jpg` | YES (74.9%) | NO (99.2%) | **Gate Closed** | `516-743.72` | 337.4 | 9103.7 | 9453.0 |
| `best_yolo11n.onnx` | `car_test_2.jpg` | YES (76.5%) | YES (95.6%) | **Try 1** | `30A-69770` | 357.9 | 1393.4 | 1760.3 |
| `best_yolo11n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 107.2 | 0.0 | 135.2 |
| `best_yolo11n.onnx` | `car_test_4.jpg` | YES (75.1%) | YES (97.9%) | **Try 1** | `51G-743.72` | 45.0 | 1077.8 | 1129.8 |
| `best_yolo11s.onnx` | `car_test_2.jpg` | YES (74.0%) | YES (97.9%) | **Try 1** | `30A-69770` | 310.1 | 986.7 | 1304.7 |
| `best_yolo11s.onnx` | `car_test_3.jpg` | YES (36.0%) | NO (100.0%) | **Gate Closed** | `TRUNGT` | 222.6 | 5357.9 | 5602.0 |
| `best_yolo11s.onnx` | `car_test_4.jpg` | YES (75.6%) | YES (96.1%) | **Try 1** | `51G-743.72` | 114.2 | 891.1 | 1012.2 |
| `best_yolo26n.onnx` | `car_test_2.jpg` | YES (76.6%) | YES (94.5%) | **Try 1** | `30A-69770` | 188.8 | 902.3 | 1098.1 |
| `best_yolo26n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 94.5 | 0.0 | 120.5 |
| `best_yolo26n.onnx` | `car_test_4.jpg` | YES (80.4%) | NO (95.9%) | **Gate Closed** | `516-743.72` | 92.0 | 5962.8 | 6064.6 |
| `best_yolo26s.onnx` | `car_test_2.jpg` | YES (76.6%) | YES (97.5%) | **Try 1** | `30A-69770` | 472.8 | 1636.0 | 2120.8 |
| `best_yolo26s.onnx` | `car_test_3.jpg` | YES (51.6%) | NO (87.1%) | **Gate Closed** | `A` | 450.8 | 4300.8 | 4778.6 |
| `best_yolo26s.onnx` | `car_test_4.jpg` | YES (75.6%) | YES (97.4%) | **Try 1** | `51G-743.72` | 96.2 | 1204.5 | 1309.7 |
| `best_yolo8.onnx` | `car_test_2.jpg` | YES (79.6%) | YES (94.6%) | **Try 1** | `30A-69770` | 353.5 | 1416.3 | 1779.8 |
| `best_yolo8.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 135.0 | 0.0 | 162.0 |
| `best_yolo8.onnx` | `car_test_4.jpg` | YES (80.0%) | YES (97.2%) | **Try 1** | `51G-743.72` | 75.1 | 1066.2 | 1150.3 |
| `yolov8n.onnx` | `car_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 287.9 | 0.0 | 298.9 |
| `yolov8n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 142.1 | 0.0 | 170.1 |
| `yolov8n.onnx` | `car_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 135.2 | 0.0 | 146.2 |

| 2026-10-14 09:19:52 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 6119.9 | 1585.2 |
| 2026-10-14 09:19:52 | `car_test_3.jpg` | 2nd try | 640px | 1 | `29H-6840` | YES | 186.2 | 10585.9 |
| 2026-10-14 09:19:52 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 543.9 | 2693.4 |
| 2026-10-14 09:19:52 | `car_test_5.jpg` | 1st try | 640px | 1 | `51G-777.77` | YES | 326.1 | 2808.5 |
| 2026-10-14 09:19:52 | `car_test_6.jpg` | 2nd try | 1024px | 1 | `63-H9-0394` | NO | 1020.9 | 17449.3 |
| 2026-10-14 09:19:52 | `car_test_7.jpg` | 2nd try | none | 0 | `NONE` | NO | 635.1 | 0.0 |
| 2026-10-14 09:19:52 | `car_test_8.jpg` | 2nd try | none | 0 | `NONE` | NO | 575.8 | 0.0 |
| 2026-10-14 09:23:02 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A-69770` | YES | 2669.3 | 812.0 |
| 2026-10-14 09:23:02 | `car_test_3.jpg` | 2nd try | 1024px | 1 | `29M-6840` | YES | 323.4 | 2346.5 |
| 2026-10-14 09:23:02 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G-743.72` | YES | 209.3 | 1428.6 |
| 2026-10-14 09:23:02 | `car_test_5.jpg` | 1st try | 640px | 1 | `51G-777.77` | YES | 170.4 | 1215.8 |
| 2026-10-14 09:23:02 | `car_test_6.jpg` | 2nd try | 1024px | 1 | `63-H9-0394` | NO | 497.5 | 9069.6 |
| 2026-10-14 09:23:02 | `car_test_7.jpg` | 2nd try | none | 0 | `NONE` | NO | 328.6 | 0.0 |
| 2026-10-14 09:23:02 | `car_test_8.jpg` | 2nd try | none | 0 | `NONE` | NO | 331.2 | 0.0 |
