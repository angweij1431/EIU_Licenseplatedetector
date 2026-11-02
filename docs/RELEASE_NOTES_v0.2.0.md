# Release Notes: v0.2.0

Release Date: November 2, 2026  
Model Version: `alpr-v0.2`  
Package Version: `0.2.0`  

## Overview
Release 0.2.0 replaces the baseline plate detector with a Megvii YOLOX-Nano model fine-tuned for Vietnamese license plates under Circular 24/2023/TT-BCA. This transition achieves full Apache 2.0 license compliance for commercial embedding inside CerberusEdge smart parking appliances.

## Key Changes

### 1. Licensing and Compliance
- Replaced Ultralytics YOLO11n (AGPL-3.0) with Megvii YOLOX-Nano (Apache 2.0).
- Eliminates copyleft obligations, allowing closed-source proprietary distribution.

### 2. Detection Engine Architecture
- Model Architecture: Megvii YOLOX-Nano (depth factor 0.33, width factor 0.25, approximately 0.91 million parameters).
- Model Format: Open Neural Network Exchange (ONNX) format with in-graph inference decoding (`decode_in_inference = True`).
- Output Tensor Shape: `[1, 8400, 6]` containing bounding box coordinates `(x, y, w, h)`, objectness score, and class confidence.
- Scoring Formula: Detection confidence is calculated as the product of objectness and class score (`obj_score * cls_score`).
- Preprocessing Specifications:
  - Channel ordering: Blue-Green-Red (BGR).
  - Tensor values: Unnormalized `[0.0, 255.0]` float32.
  - Letterbox alignment: Top-left padding with fill value 114.

### 3. Benchmark and Latency Metrics
Evaluated on x86_64 CPU (ONNX Runtime, 50 warm iterations):
- Model File Size: 8.67 MB (17 percent reduction from 10.48 MB).
- Detector CPU Latency: Mean 48.2 ms (compared to 51.6 ms baseline, approximately 7 to 15 percent faster).
- Pipeline End-to-End Latency: Mean 76.7 ms (median p50: 73.9 ms).
- Cold Start Initialization: 278.1 ms.
- Motorcycle Recall: Higher confidence across motorcycle test samples (e.g. `bike_test_9.jpg`: 0.879 to 0.912).
- False Positive Suppression: False detections on non-plate background samples reduced significantly (e.g. `neg_sample_test_4.jpg`: 0.451 down to 0.001).

### 4. Known Edge-Case Trade-offs
- Low-Contrast and Foggy Conditions: Low-contrast scenes (`car_foggy_test.jpg`) yield lower objectness confidence (0.047) compared to coupled heads. Low-light deployments should utilize Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessing or lower the detection threshold to 0.10.
- Plate-Like Distractor Signs: Rectangular rooftop signs ("TẬP LÁI") on driving school vehicles can trigger candidate proposals. The downstream Compact Convolutional Transformer (CCT-XS) Optical Character Recognition (OCR) engine and syntax validator safely flag these candidates as invalid (`is_valid: False`), preventing invalid barrier gate operations.

## Release Checksums (SHA-256)
- `models/detector.onnx`: `8c303c836e1abbcb336e514dda4999fd69c770c4b683be617d9c32998dead8e6`
- `models/cct_xs.onnx`: `8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44`
- `models/cct_xs_plate_config.yaml`: `0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6`

## Verification
- Test Suite: 13 of 13 tests passing (100 percent pass rate via `pytest tests/ -v`).
