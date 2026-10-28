# Model Card: EIU LPR Vision Engine

## Model Details
- Model Name: EIU Automated License Plate Recognition (ALPR) Engine
- Version: `alpr-v0.1`
- Detector Architecture: YOLO11n single-class (`plate`) exported to Open Neural Network Exchange (ONNX) format.
- Recognizer Architecture: Compact Convolutional Transformer (CCT-XS) Optical Character Recognition (OCR) model in ONNX format.
- License: MIT License (Pipeline framework and OCR model), Open Research Distribution (Detector weights).

## Model Files and Checksums

The release model assets are located in `models/` and verified with SHA-256 hashes:

| File Name | Framework | Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `detector.onnx` | ONNX Runtime | 10.9 MB | `6aa3fa4ce22d711d3c39216568ebbc45c7843b57fcfcfae774627fa962bb28a5` |
| `cct_xs.onnx` | ONNX Runtime | 3.3 MB | `8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44` |
| `cct_xs_plate_config.yaml` | YAML Config | 1.7 KB | `0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6` |

## Main Application
The primary application of this engine is:
- Ingesting in-memory camera image frames (`np.ndarray` in RGB layout).
- Detecting license plate coordinates using a 640x640 base resolution.
- Extracting crops with dynamic padding, glare attenuation, and contrast enhancement.
- Reading plate characters in a single forward pass.
- Validating syntax against Circular 24/2023/TT-BCA standards.
- Emitting structured JSON context data directly in memory.

## Pure ONNX Runtime Setup
- Inference executes directly on ONNX Runtime (`onnxruntime`), eliminating the `ultralytics` package and PyTorch runtime dependencies.
- Image preprocessing (aspect-preserving letterbox scaling, channel layout conversion, and float normalization) and Non-Maximum Suppression (NMS) duplicate filtering are executed using pure NumPy.
- This design provides a minimal package size (~20 MB), fast startup, and native CPU performance without external framework overhead.

## Supported Formats (Circular 24/2023/TT-BCA)
- Single-line automobiles: 2 digits + 1 letter + 4 to 6 digits (e.g. `30A69770`, `51G74372`).
- Dual-line motorcycles: 2 digits + (1 letter and 1 digit, or 2 letters) + 4 to 6 digits (e.g. `59D242069`, `59VA24003`). Both lines are recognized together in a single pass without image slicing.

## Performance
- Low latencies observed across edge CPU test runs (typically 50 ms to 200 ms end-to-end depending on edge CPU model).
- High recall across test fleets, with reliable rejection of non-vehicle background scenes.

## Limitations
- Dutch roll tilt angles exceeding 45 degrees.
- Plates heavily occluded by mud, chains, or severe physical damage.
