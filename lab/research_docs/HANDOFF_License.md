# Project Handoff: License Plate Recognition and Commercial Architecture Migration

Date: 2026-10-27  
Working Directory: `c:\Users\User\EIU_OITP\lpr_research`  
Python Environment: `.\venv_models\Scripts\python.exe`  
Target Scope: Automated License Plate Recognition (ALPR) and Campus Access Control at Eastern International University (EIU)  
Primary Goal for Incoming Session: Replace the Ultralytics YOLO detector with a commercially permissible open-source alternative (Apache 2.0, MIT, or BSD) to eliminate GNU Affero General Public License (AGPL) restrictions.

---

## 1. Project Background and Context

The project is an edge-optimized Automated License Plate Recognition (ALPR) system deployed for campus vehicle access control. The system performs three primary functions:
1. Detects vehicle license plates in video or camera image streams.
2. Reads alphanumeric characters from both single-line automobile plates and dual-line motorcycle plates.
3. Validates license plate syntax against Vietnamese road traffic standards (Circular 24/2023/TT-BCA) to trigger boom barriers or log security exceptions.

The current system has reached high functional accuracy:
- 100% recall on valid Vietnamese test plates.
- Clean separation between valid, defective, and foreign plates.
- End-to-end inference latency under 450 milliseconds on edge Central Processing Unit (CPU) hardware without requiring a graphics card.

---

## 2. Reason for Hand-Off: Licensing Constraint

The object detection component currently relies on YOLO11n trained via the `ultralytics` package.

Ultralytics licenses its software and model weights under the GNU Affero General Public License version 3.0 (AGPL-3.0). Under AGPL-3.0:
- Any software service or network application utilizing AGPL-3.0 components must release its entire source code under the same copyleft terms.
- Commercial closed-source distribution or proprietary campus integration cannot proceed without purchasing commercial licenses from Ultralytics.

To prepare the project for commercial deployment and proprietary distribution, the object detection module must be migrated to a permissive license:
- Acceptable licenses: Apache License 2.0, MIT License, or BSD 3-Clause License.
- Prohibited licenses: GNU General Public License (GPL), GNU Lesser General Public License (LGPL), or GNU Affero General Public License (AGPL).

---

## 3. Current Pipeline Architecture

The system is decoupled in `lpr_engine.py` into separate detector, recognizer, and validation components.

### Component 1: Plate Detector (Needs Replacement - potentially we will do a new obj detection model and train one up separately.)
- Current implementation: `YOLODetector` class wrapping `Models_yolo/balanced_skew_robust_yolo11n.onnx` via Open Neural Network Exchange (ONNX) Runtime.
- Function: Takes an input image frame and outputs bounding boxes `[x1, y1, x2, y2]` with confidence scores for class 0 (`plate`).
- Status: Functions accurately, but subject to AGPL-3.0 restrictions through Ultralytics. Must be replaced.

### Component 2: Plate Recognizer (Permissible - Keep As Is)
- Current implementation: `CCTXSRecognizerAdapter` class wrapping `Models_ocr/cct_xs.onnx` (1.8 MB).
- Library: `fast-plate-ocr`.
- License: MIT License (fully approved for commercial use).
- Function: Directly takes a cropped plate image and outputs plate string, per-character softmax confidences, and country classification ($P(\text{Vietnam})$).
- Performance: 12 to 50 milliseconds per crop on CPU. Dual-line motorcycle plates and single-line automobile plates are recognized in a single forward pass without geometric slicing.

### Component 3: Image Preprocessing (Custom Code - Keep As Is)
- Location: `lpr_core/preprocessor.py`.
- Function: Provides specular glare attenuation for tropical lighting and mild adaptive contrast enhancement (CLAHE) for low-contrast images.
- License: Proprietary project code (no third-party copyleft dependencies).

### Component 4: Syntax Validator (Custom Code - Keep As Is)
- Location: `lpr_core/validator.py`.
- Function: Validates format against Vietnamese regulation Circular 24/2023/TT-BCA, supporting single-line car formats, traditional motorcycle formats, and modern two-letter identification plates.
- License: Proprietary project code.

### Component 5: API and Verification Simulation (Custom Code - Keep As Is)
- Locations: `main.py` (FastAPI service) and `simulate_rfid.py` (multi-image streaming simulator with Radio Frequency Identification (RFID) card matching).
- Function: Evaluates plates against an approved campus registry, writes structured JSON telemetry, and routes images into `passed/`, `rejected/`, and `flagged/` directories.

---

## 4. Key Requirements for the Replacement Detector

The incoming engineer or agent should select or train an object detector meeting these criteria:

1. Permissive License:
   - The training framework, model architecture, and exported weights must be licensed under Apache 2.0, MIT, or BSD 3-Clause.
2. Architecture Scope:
   - Single-class object detection (`plate`).
   - Small footprint suitable for edge CPU deployment (model size under 25 megabytes). However, since moving this to edge hardware- suggest compatible and ways we can simulate.
3. Runtime Interface:
   - Must export cleanly to Open Neural Network Exchange (ONNX) format.
   - Must support dynamic input resolutions (640x640 base resolution, escalatable to 1024x1024 for micro-thumbnails) to support the multi-try escalation logic in `lpr_engine.py`.
4. Inference Speed:
   - Detection latency on Intel Core i5/i7 CPU under 150 milliseconds per 640x640 frame.
5. Accuracy Target:
   - Over 95% recall on standard vehicle plates.
   - Capability to detect rotated or skewed motorcycle plates up to 45 degrees Dutch roll angle.

---

## 5. Candidate Permissive Architectures to Investigate

The following model families do not use AGPL and have permissive commercial licensing:
- if possible please look and suggest more modernised projects
- could you let me know your cut off knowledge and when these appear? - let me know if you require any web search
1. YOLOX (Megvii):
   - License: Apache License 2.0.
   - Highlights: Anchor-free, highly mature, standard ONNX export pipeline, widely used in commercial ALPR systems.
2. YOLOv6 (Meituan):
   - License: Apache License 2.0.
   - Highlights: Purpose-built for industrial edge inspection and CPU/GPU hardware deployment.
3. RT-DETR (Real-Time Detection Transformer via Baidu PaddleDetection):
   - License: Apache License 2.0.
   - Highlights: Fast transformer-based detector without Non-Maximum Suppression (NMS) post-processing bottlenecks.
4. D-FINE (2024 Real-Time Detector):
   - License: MIT License.
   - Highlights: Modern real-time object detector with strong accuracy-to-latency balance and clean MIT licensing.
5. RTMDet (OpenMMLab MMDetection):
   - License: Apache License 2.0.
   - Highlights: Highly competitive accuracy with lightweight variants (RTMDet-Tiny, RTMDet-s).
6. MobileNetV3-SSD / SSDLite (Torchvision):
   - License: BSD 3-Clause.
   - Highlights: Standard PyTorch torchvision library with zero third-party licensing baggage.
7. other modern alternatives
8. If required, potentially to suggest cr

---

## 6. Project Assets and Training Data Inventory

All datasets and notebooks needed to train a replacement detector are available in the repository:

1. Notebook Templates:
   - `ColabNotebookTemplate/01_train_yolo_balanced_1to1_25_40ep.ipynb`: Base training notebook.
   - `ColabNotebookTemplate/07_train_yolo_balanced_40eps_skew_robust.ipynb`: Skew-robust augmentation parameters (rotation degrees=15.0, perspective=0.0005, shear=5.0, scale=0.5).
2. Evaluation Scripts:
   - `benchmarks/run_benchmarks.py`: Evaluates detector accuracy, latency, and syntax pass rate across image batches.
   - `simulate_rfid.py`: Validates end-to-end access control logic, JSON output, and multi-image directory streaming.
3. Test Suites:
   - `test_images/`: 18 representative real-world test images covering cars, motorcycles, rainy/foggy scenes, steep angles, and negative background samples.
   - Multi-fleet test directories: 151 images categorized across car, motorcycle, and defective plates.


---

## 7. Migration Steps for the Next Session

1. Select a candidate architecture with an Apache 2.0 or MIT license.
2. Adapt the training dataset annotations (currently formatted in standard YOLO bounding box text format: `class x_center y_center width height`) into the target format if required. / let me know what should be changed or altered.
3. Train the model using the skew augmentation settings established in Notebook 07.
4. Export the trained model to Open Neural Network Exchange (ONNX) format with dynamic input shapes enabled.
5. In `lpr_engine.py`, implement an adapter class inheriting from `PlateDetector` (e.g., `YOLOXDetector` or `GenericONNXDetector`) to parse output tensors without modifying downstream OCR or validation logic.
6. Run `benchmarks/run_benchmarks.py` and `simulate_rfid.py` to verify that recall, latency, and gate routing match or exceed existing benchmarks.

