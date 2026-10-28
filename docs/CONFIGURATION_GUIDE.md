# Configuration and API Reference

Tool reference for integrating the `eiu_lpr` vision library.

## 1. Repository Structure

1. **`src/eiu_lpr/`**: Production library source code.
   - `pipeline.py`: Main `LPRPipeline` class exposing `detect()`, `read_plate()`, `recognize()`, `process()`, and `warmup()`.
   - `detector.py`: Open Neural Network Exchange (ONNX) Runtime YOLO detector with NumPy letterbox and Non-Maximum Suppression (NMS).
   - `recognizer.py`: Compact Convolutional Transformer (CCT-XS) Optical Character Recognition (OCR) adapter.
   - `preprocessor.py`: Dynamic 4% padding, Retinex glare reduction, and mild contrast enhancement.
   - `validator.py`: Circular 24/2023/TT-BCA text normalization and regex validation.
   - `models.py`: Dataclasses (`BoundingBox`, `PlateCandidate`, `PipelineResult`) and JSON formatting.
   - `exceptions.py`: Typed exceptions (`InvalidInputError`, `ModelNotFoundError`, `ConfigurationError`).

2. **`models/`**: Frozen release assets and verification hashes.
   - `detector.onnx`: YOLO11n plate detection weights (10.9 MB).
   - `cct_xs.onnx`: CCT-XS plate recognition weights (3.3 MB).
   - `cct_xs_plate_config.yaml`: Character alphabet and geometry configuration.
   - `checksums.sha256`: SHA-256 cryptographic hashes for release verification.

3. **`dist/`**: Pre-built wheel package.
   - `eiu_lpr-0.1.0-py3-none-any.whl`: Standalone wheel installable via `pip`.

4. **`benchmarks/`**: Latency and evaluation tools.
   - `benchmark_linux_cpu.py`: CPU latency benchmark measuring startup, detection, and OCR percentiles.
   - `run_benchmarks.py`: Batch evaluation script across image fleets.
   - `BENCHMARK_SUMMARY.md`: Historical benchmark performance ledger.

5. **`tests/` and `run_tests.py`**: Automated verification test suite.
   - 10 unit tests verifying in-memory arrays, zero disk I/O, zero prints, split APIs, and threshold overrides.

6. **`scripts/`**: Utility scripts.
   - `generate_checksums.py`: Recalculates SHA-256 hashes when model assets in `models/` change.

7. **`demo.py`**: Standalone verification script that processes an image and prints structured JSON output.

---

## 2. Commands Reference

### Package Installation
```bash
pip install eiu_lpr-0.1.0-py3-none-any.whl
```

### Run Test Suite
```bash
python run_tests.py
```

### Run Latency Benchmark
```bash
python benchmarks/benchmark_linux_cpu.py --iterations 100
```

### Verify Model Checksums
On Linux:
```bash
sha256sum -c models/checksums.sha256
```
On Windows PowerShell:
```powershell
Get-FileHash models/detector.onnx -Algorithm SHA256
```

### Run Quick Verification Demo
```bash
python demo.py
```

---

## 3. Integration Code Examples

### Example 1: Combined Recognition (`pipeline.recognize`)
Takes an in-memory frame, detects all plates, and returns a list of `PlateCandidate` dataclass objects directly:

```python
import cv2
from eiu_lpr import LPRPipeline

# 1. Initialize pipeline
pipeline = LPRPipeline(
    detector_path="models/detector.onnx",
    ocr_path="models/cct_xs.onnx",
    ocr_config_path="models/cct_xs_plate_config.yaml",
)

# 2. Warmup sessions
pipeline.warmup(iterations=2)

# 3. Direct recognition on an image (RGB uint8 array [H, W, 3])
image = cv2.imread("test_images/car_test_2.jpg")
candidates = pipeline.recognize(image, color_format="BGR")

for candidate in candidates:
    print(f"Plate Text : {candidate.normalized_text}")  # e.g. "30A69770"
    print(f"Raw OCR    : {candidate.raw_text}")         # e.g. "30A69770"
    print(f"Valid Form : {candidate.is_valid}")          # True
    print(f"Confidence : {candidate.ocr_confidence:.4f}")
    print(f"BoundingBox: {candidate.bbox}")
```

### Example 2: Decoupled Tracking Workflow (`detect` and `read_plate`)
Run fast spatial detection on each video frame, and trigger plate reading only when a box stabilizes:

```python
# Step 1: Fast bounding box detection on incoming frame
boxes = pipeline.detect(frame, color_format="RGB")

# Step 2: Read characters only from the selected stable box
if boxes:
    selected_box = boxes[0]
    candidate = pipeline.read_plate(frame, bbox=selected_box, color_format="RGB")
    print(f"Stable Plate: {candidate.normalized_text} (Valid: {candidate.is_valid})")
```

### Example 3: Timed Telemetry Processing (`pipeline.process`)
Runs recognition, times inference in milliseconds, and formats JSON telemetry matching the edge backend schema:

```python
result = pipeline.process(image, color_format="BGR", request_id="req-001")
print(f"Execution Time: {result.processing_time_ms:.1f} ms")

# Get exact reference schema JSON
json_output = result.to_dict(minimal=True)
```

### Example 4: Dynamic Threshold Override
Override detection threshold on specific frames without re-initializing the pipeline:

```python
# Uses default 0.20 threshold
boxes = pipeline.detect(frame)

# Applies custom 0.40 threshold for this call only
boxes_strict = pipeline.detect(frame, detection_threshold=0.40)
candidates = pipeline.recognize(frame, detection_threshold=0.40)
```

---

## 4. API Reference

### Constructor Parameters

```python
pipeline = LPRPipeline(
    detector_path="models/detector.onnx",
    ocr_path="models/cct_xs.onnx",
    ocr_config_path="models/cct_xs_plate_config.yaml",
    detection_threshold=0.20,
    nms_iou_threshold=0.45,
    max_detections=10,
    ocr_mean_threshold=0.70,
    ocr_min_threshold=0.50,
    intra_op_threads=None,
    model_version="alpr-v0.1",
    schema_version="1.0",
)
```

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `detector_path` | `str` | *(Required)* | Path to detector ONNX weights. |
| `ocr_path` | `str` | *(Required)* | Path to CCT-XS OCR ONNX weights. |
| `ocr_config_path` | `str` | *(Required)* | Path to CCT-XS YAML configuration. |
| `detection_threshold` | `float` | `0.20` | Minimum detector confidence to retain bounding box. |
| `nms_iou_threshold` | `float` | `0.45` | Box overlap threshold for Non-Maximum Suppression (NMS). |
| `max_detections` | `int` | `10` | Maximum candidate boxes retained after NMS per frame. |
| `ocr_mean_threshold` | `float` | `0.70` | Expected mean character confidence score. |
| `ocr_min_threshold` | `float` | `0.50` | Lowest acceptable character probability threshold. |
| `intra_op_threads` | `int` | `None` | Optional CPU core limit for ONNX Runtime. |
| `model_version` | `str` | `"alpr-v0.1"` | Model version tag included in telemetry output. |
| `schema_version` | `str` | `"1.0"` | Schema version tag included in telemetry output. |

### Available Methods

#### `pipeline.detect(image, color_format="RGB", detection_threshold=None) -> list[BoundingBox]`
Runs plate detection on an in-memory frame (`np.ndarray`, `uint8`, shape `[H, W, 3]`). Returns detected bounding boxes with original image pixel coordinates.

#### `pipeline.read_plate(image, bbox, color_format="RGB", include_debug_crop=False) -> PlateCandidate`
Crops, enhances, and reads characters from a specific bounding box. If `include_debug_crop=True`, attaches the enhanced plate crop `np.ndarray` to `candidate.debug_crop`.

#### `pipeline.recognize(image, color_format="RGB", detection_threshold=None, include_debug_crop=False) -> list[PlateCandidate]`
Runs detection followed by plate reading across all detected boxes. Returns pure `list[PlateCandidate]`.

#### `pipeline.process(image, color_format="RGB", detection_threshold=None, request_id=None, include_debug_crop=False, include_annotated_image=False) -> PipelineResult`
Executes recognition, measures execution time in milliseconds, and returns a `PipelineResult`. When `include_annotated_image=True`, attaches the full-frame visualized image array with drawn bounding boxes and text to `result.annotated_image`.

#### `pipeline.annotate(image, candidates) -> np.ndarray`
Generates a copy of the in-memory frame with bounding boxes, plate text, and confidence annotations drawn. Returns pure `np.ndarray` without modifying the original or writing to disk.

#### `pipeline.warmup(iterations=2) -> None`
Executes dummy inference passes to initialize memory buffers and prime execution providers.

---

## 5. Operational Safeguards: Detection Threshold and Max Detections

### Detection Threshold (`0.20`)
The default `detection_threshold` is set to `0.20`. This ensures high recall under adverse conditions (rain streaks, lens moisture, or headlight glare) where initial detector scores may drop below `0.40`. Non-plate candidate boxes are discarded downstream by low OCR confidence and regex syntax validation.

### Max Detections Ceiling (`10`)
The default `max_detections` is set to `10`. 
- **Physical Context**: At a parking boom barrier or toll gate, only 1 to 3 vehicles realistically appear in the camera frame simultaneously.
- **Latency Protection**: If a camera encounters extreme noise (headlight lens flare, complex foliage, or chain-link fence reflections), setting a low detection threshold like `0.20` could theoretically propose dozens of candidate boxes.
- Capping NMS output to the top 10 highest-confidence boxes prevents catastrophic CPU latency spikes (which would occur if the worker ran OCR on 50+ noise regions), keeping maximum single-frame processing time well within the operational gate budget (< 150 ms).
