# Model Card: EIU LPR Vision Engine

## Model Details
- Model Name: EIU Automated License Plate Recognition (ALPR) Engine
- Version: `alpr-v0.3`
- Detector Architecture: Megvii YOLOX-Nano single-class (`plate`) exported to Open Neural Network Exchange (ONNX) format with in-graph inference decoding (150-epoch retrained weights on dedicated server, peak validation at Epoch 149 with mAP@50:95 of 0.668).
- Recognizer Architecture: Compact Convolutional Transformer (CCT-XS) Optical Character Recognition (OCR) model in ONNX format.
- License: Apache 2.0 (Detector architecture and weights), MIT License (Pipeline framework and OCR model).

## Model Files and Checksums

The release model assets are located in `models/` and verified with SHA-256 hashes:

| File Name | Framework | Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `detector.onnx` | ONNX Runtime | 8.7 MB | `412a566f1bb1187e5f9bd66f33312fcc2fa2fb346ee5ddda5484342f773613ca` |
| `cct_xs.onnx` | ONNX Runtime | 3.2 MB | `8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44` |
| `cct_xs_plate_config.yaml` | YAML Config | 1.7 KB | `0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6` |

## Main Application
The primary application of this engine is:
- Ingesting in-memory camera image frames (`np.ndarray` in RGB or BGR layout).
- Detecting license plate coordinates using a 640x640 base resolution.
- Extracting crops with dynamic padding, glare attenuation, and contrast enhancement.
- Reading plate characters in a single forward pass.
- Validating syntax against Circular 24/2023/TT-BCA standards.
- Emitting structured JSON context data directly in memory.

## Pure ONNX Runtime Setup
- Inference executes directly on ONNX Runtime (`onnxruntime`), eliminating `ultralytics` and PyTorch runtime dependencies.
- Image preprocessing: Native BGR channel layout, unnormalized `[0.0, 255.0]` float32 input tensors, and top-left padding with fill value 114.
- Postprocessing: Decoupled anchor-free output tensor `[1, 8400, 6]` `(x, y, w, h, obj_score, cls_score)`. Final detection confidence is computed via `obj_score * cls_score`, followed by vectorized NumPy Non-Maximum Suppression (NMS).
- This design provides a minimal package size (~20 MB), fast startup, and native CPU performance without external framework overhead.

## Supported Formats (Circular 24/2023/TT-BCA)
- Single-line automobiles: 2 digits + 1 letter + 4 to 6 digits (e.g. `30A69770`, `51G74372`).
- Dual-line motorcycles: 2 digits + (1 letter and 1 digit, or 2 letters) + 4 to 6 digits (e.g. `59D242069`, `59VA24003`). Both lines are recognized together in a single pass without image slicing.

## Performance
- CPU Detection Latency: ~48.2 ms mean latency (tested on x86_64 CPU with ONNX Runtime).
- End-to-End Latency: ~76.7 ms mean latency (Detector + Adaptive Enhancer + OCR + Validator).
- Cold Start Initialization: ~278 ms.

## Limitations
- Extreme low-contrast or heavy fog: Objectness confidence drops on hazy plates; low-light or foggy feeds benefit from contrast preprocessing (CLAHE) or lowering detection threshold to 0.10.
- Vehicle rooftop distractor signs: Rectangular driving school signs ("TẬP LÁI") or advertising banners can produce candidate boxes, but are safely rejected downstream by the character syntax validator (`is_valid: False`).
- Dutch roll tilt angles exceeding 45 degrees.
- Plates heavily occluded by mud, chains, or severe physical damage.
