# CerberusEdge ALPR Vision Engine: Integration Guide and Technical Reference

## 1. System and Integration Overview

The `eiu_lpr` library is a standalone, in-memory Automatic License Plate Recognition (ALPR) package designed for edge deployments on CerberusEdge smart parking worker nodes. It processes raw camera frames directly in memory, performs license plate detection and character recognition using pure Open Neural Network Exchange (ONNX) Runtime, and returns structured metadata without writing temporary files to disk.

### Deliverables in This Repository
* **Package Wheel**: `dist/eiu_lpr-0.1.0-py3-none-any.whl` (compiled distribution package).
* **Detector Model**: `models/detector.onnx` (YOLO11n model trained for vehicle plate detection).
* **Recognizer Model**: `models/cct_xs.onnx` (Compact Vision Transformer OCR model).
* **Recognizer Configuration**: `models/cct_xs_plate_config.yaml` (character dictionary and normalization settings).
* **Dependency Constraints**: `constraints.txt` (pinned dependencies for Python 3.10 to 3.12).
* **Test Suite**: `run_tests.py` and `tests/test_golden_dataset.py`.

---

### Step-by-Step Integration Flow

1. **Install Package Dependencies**
   ```bash
   pip install dist/eiu_lpr-0.1.0-py3-none-any.whl -c constraints.txt
   ```
   * *Decision*: Pinned constraints prevent binary incompatibilities across numpy, opencv-python-headless, and onnxruntime.

2. **Deploy and Verify Model Assets**
   * Place `models/` in an accessible path on the edge filesystem.
   * Verify file checksums against the manifest:
     ```bash
     sha256sum -c models/checksums.sha256
     ```

3. **Initialize the Pipeline Once at Worker Startup**
   * Construct `LPRPipeline` during service startup. Do not reconstruct per frame:
     ```python
     from eiu_lpr import LPRPipeline

     pipeline = LPRPipeline(
         detector_path="models/detector.onnx",
         ocr_path="models/cct_xs.onnx",
         ocr_config_path="models/cct_xs_plate_config.yaml",
         intra_op_threads=2,
         detection_threshold=0.20,
         max_detections=10,
     )
     ```

4. **Prime Execution Providers (Warmup)**
   ```python
   pipeline.warmup(iterations=2)
   ```
   * *Decision*: Pre-allocates execution graphs and memory buffers, eliminating latency spikes on the first vehicle.

5. **Pass Camera Frames Directly from RAM**
   * Pass frames directly as 3-channel unsigned 8-bit integer (`uint8`) NumPy arrays:
     * For OpenCV streams (`cv2.VideoCapture`), pass `color_format="BGR"`.
     * For Real-Time Streaming Protocol (RTSP) streams via PyAV or PIL, pass `color_format="RGB"`.

6. **Process the Returned Payload**
   * Call `result.to_dict(minimal=True)` for standard event dispatching.
   * If `candidates` is empty, no vehicle plate is present; maintain gate status.
   * If `candidates` contains entries, evaluate `candidates[0]["plate"]` against access rules.

---

### Verification and Testing

#### Running the Core Test Suite
Verify that all 11 unit tests pass:
```bash
python run_tests.py
```
**Expected Output:**
```text
test_01_rgb_in_memory_inference ... ok
test_02_bgr_color_format ... ok
test_03_zero_disk_io ... ok
test_04_split_apis_consistency ... ok
test_05_pure_onnx_no_ultralytics ... ok
test_06_invalid_input_exceptions ... ok
test_07_invalid_config_exceptions ... ok
test_08_warmup ... ok
test_09_custom_versioning ... ok
test_10_threshold_override ... ok
test_11_in_memory_debug_images ... ok

----------------------------------------------------------------------
Ran 11 tests in 5.510s

OK
```

#### Running the Golden Dataset Regression
Test recognition accuracy against the 20-image golden dataset (including low-light, fog, angled, and negative samples):
```bash
python tests/test_golden_dataset.py -v
```
**Expected Output:**
```text
test_golden_dataset_execution ... ok

----------------------------------------------------------------------
Ran 1 test in 6.273s

OK
```

#### Running the Interactive Demo
Run a single-image demonstration showing formatted JSON payloads:
```bash
python demo.py
```

#### Extracting In-Memory Visual Outputs
The library operates strictly in RAM. To extract visual evidence for debugging or UI display:

* **Annotated Frame (Boxes & Labels on Full Frame)**:
  ```python
  # Pass include_annotated_image=True to process()
  result = pipeline.process(frame, color_format="BGR", include_annotated_image=True)
  if result.annotated_image is not None:
      cv2.imwrite("debug_annotated.jpg", result.annotated_image)
  ```

* **Enhanced Plate Crops (Cropped Region Fed to OCR)**:
  ```python
  # Pass include_debug_crop=True to process()
  result = pipeline.process(frame, color_format="BGR", include_debug_crop=True)
  for i, cand in enumerate(result.candidates):
      if cand.debug_crop is not None:
          cv2.imwrite(f"plate_crop_{i}.jpg", cand.debug_crop)
  ```

---

## 2. Architecture and Evolution

### What Changed and What Was Retired

| Architectural Domain | Legacy Research Implementation (`lab/`) | Production Library Implementation (`src/eiu_lpr/`) | Engineering Decision & Rationale |
| :--- | :--- | :--- | :--- |
| **Package Structure** | Monolithic standalone scripts (`lpr_engine.py`, `infer.py`) executed via shell commands. | Modular Python package installable as a `.whl` wheel artifact (`pyproject.toml`). | Eliminates `sys.path` tampering. Enables clean imports across worker services (`from eiu_lpr import LPRPipeline`). |
| **Filesystem & Disk I/O** | Captured frames and intermediate crops were written to `Outputs/` before OCR read them back with `cv2.imread()`. | Strict zero-disk-I/O policy. Images, crops, and annotated frames are processed entirely in RAM. | Prevents flash drive degradation on edge computers and removes 20 to 50 ms of disk latency per frame. |
| **ML Framework** | Depended on the full `ultralytics` package, which pulls PyTorch, torchvision, and training libraries. | Replaced with pure `onnxruntime` inference, custom NumPy letterboxing, and vectorized Non-Maximum Suppression (NMS). | Eliminates PyTorch/Ultralytics runtime overhead, dependencies, and CUDA library baggage. |
| **Motorcycle Plate OCR** | Relied on geometric deskewing and horizontal projection slicing (`deskew_and_slice_plate`) to split two-line plates into separate crops. | Compact Convolutional Transformer (CCT-XS) recognizes both lines in a single forward pass without slicing. | Heuristic line slicing frequently failed on angled or dirty plates. The transformer handles two-line text naturally. |
| **Business Logic Scope** | Embedded hardcoded vehicle dictionaries (`REGISTERED_VEHICLES`) and simulated barrier relays (`GATE_OPEN`). | Completely removed. The library is strictly a perception engine returning raw and canonical text. | Access control rules, whitelists, and hardware relays belong to the backend service, not the vision library. |
| **Validation Standards** | Ad-hoc regex rules scattered across inference scripts. | Centralized validator conforming to Vietnam Ministry of Public Security Circular 24/2023 regulations. | Guarantees consistent validation rules across all cameras and provides clean separation between raw OCR and canonical plates. |

---

### Module and Directory Breakdown

The production codebase is organized under `src/eiu_lpr/`:

* **`src/eiu_lpr/__init__.py`**: Public API export (`LPRPipeline`, `PlateCandidate`, `PipelineResult`, `BoundingBox`, exceptions).
* **`src/eiu_lpr/pipeline.py`**: Central coordinator for `process()`, `recognize()`, `detect()`, `read_plate()`, and `annotate()`.
* **`src/eiu_lpr/detector.py`**: Pure ONNX Runtime YOLO11n plate detector with letterbox padding and vectorized NMS.
* **`src/eiu_lpr/recognizer.py`**: CCT-XS Vision Transformer OCR model with CTC greedy decoding and character probabilities.
* **`src/eiu_lpr/preprocessor.py`**: Input image validation (`uint8` 3D ndarray), letterboxing, and Laplacian blur evaluation.
* **`src/eiu_lpr/validator.py`**: Circular 24/2023 Vietnamese plate normalisation and regex validation.
* **`src/eiu_lpr/models.py`**: Strongly-typed dataclasses with `to_dict(minimal=True/False)` serialization.
* **`src/eiu_lpr/exceptions.py`**: Typed exception hierarchy rooted at `EIULPRError`.

---

### Dataflow of an In-Memory Frame

When invoking `pipeline.process(image, color_format="BGR")`:

1. **Validation**: `PlatePreprocessor.validate_image()` confirms input is a non-empty, 3-channel `uint8` NumPy array.
2. **Color Handling**: If `"BGR"`, converted to RGB in memory using OpenCV. If already `"RGB"`, no copy is made.
3. **Plate Detection**: Frame is letterboxed to 640x640 and evaluated by `detector.onnx`. Bounding boxes exceeding `detection_threshold` (0.20) are filtered with NMS (`iou=0.45`) and mapped to original frame coordinates.
4. **Adaptive Preprocessing**: For each crop, if dimensions are small (`< 110x60` px) or blur score is low (`< 150.0`), adaptive contrast enhancement is applied (`pass_level="enhanced_crop"`).
5. **Character Recognition**: Crop is resized to 48x128 and evaluated by `cct_xs.onnx`. CTC decoding produces raw text and softmax probabilities.
6. **Syntax Normalisation**: `VietnamPlateValidator` cleans separators, joins two-line plates, and checks Circular 24/2023 validity.
7. **Packaging**: Candidates are assembled into `PipelineResult(status="OK", candidates=[...])`. If no plates passed detection, `candidates` is `[]`.

---

## 3. Outputs, Status Dictionary and Error Handling

### Visual Model Reference (`src/eiu_lpr/models.py`)

```python
@dataclass
class PlateCandidate:
    bbox: Tuple[float, float, float, float]
    detection_score: float
    raw_text: str
    normalized_text: str
    is_valid: bool
    ocr_confidence: float
    ocr_min_confidence: float
    character_details: List[CharacterDetail]
    region: str = "Vietnam"
    region_prob: float = 0.0
    pass_level: str = "base_640px"
    debug_crop: Optional[np.ndarray] = None

@dataclass
class PipelineResult:
    candidates: List[PlateCandidate]
    processing_time_ms: float
    status: str = "OK"
    schema_version: str = "1.0"
    model_version: str = "alpr-v0.1"
    request_id: Optional[str] = None
    annotated_image: Optional[np.ndarray] = None
```

---

### Output Field Dictionary

| Field | Type | Expected Values | Description & Usage |
| :--- | :--- | :--- | :--- |
| **`status`** | `str` | `"OK"`, `"ERROR"` | Operational execution health. Returns `"OK"` whenever inference completes normally (with or without detected plates). Returns `"ERROR"` only on unhandled runtime failures (such as corrupted buffers or hardware faults). |
| **`candidates`** | `list` | `[PlateCandidate, ...]` | List of detected plate objects. Returns an empty list (`[]`) if no plate is detected in the image. Ordered by detector confidence (`candidates[0]` is the top candidate). |
| **`processing_time_ms`** | `float` | `>= 0.0` | Total wall-clock processing latency for the frame in milliseconds. |
| **`schema_version`** | `str` | `"1.0"` | JSON schema version compatibility identifier. |
| **`model_version`** | `str` | `"alpr-v0.1"` | Model bundle version identifier for telemetry auditing. |
| **`request_id`** | `str` \| `null` | String or `None` | Client tracking identifier passed back unchanged for end-to-end event tracing. |
| **`annotated_image`** | `ndarray` \| `null` | `[H, W, 3] uint8` | Visualization with drawn boxes and labels. Populated in memory only if `include_annotated_image=True`. Never serialized into JSON. |
| **`plate`** | `str` | e.g. `"29D11077"` | Canonicalized plate string formatted according to Vietnam Circular 24/2023 regulations. Used for database lookups and whitelist verification. |
| **`raw_plate`** | `str` | e.g. `"29-D1 10.77"` | Unaltered character sequence produced directly by OCR before normalisation. Used for audit logs and manual review. |
| **`confidence`** | `float` | `0.0` – `1.0` | Mean softmax probability across all recognized characters. Filter low scores before gate opening (threshold: `>= 0.70`). |
| **`ocr_min_confidence`** | `float` | `0.0` – `1.0` | Softmax probability of the lowest-scoring character. Used to identify ambiguous characters without mutating raw text. |
| **`detection_confidence`**| `float` | `0.0` – `1.0` | Bounding box confidence score from the YOLO detector. |
| **`bbox`** | `[int, int, int, int]` | `[x1, y1, x2, y2]` | Bounding box coordinates mapped to original frame pixel space. |
| **`is_valid`** | `bool` | `True`, `False` | Boolean flag indicating whether plate syntax conforms to Circular 24/2023. |
| **`character_details`** | `list` | `[{"char": str, "confidence": float}]` | Character-by-character prediction breakdown with individual softmax scores. |
| **`region`** | `str` | `"Vietnam"` | Regional plate head classification. |
| **`region_prob`** | `float` | `0.0` – `1.0` | Softmax probability score for the regional classification prediction. |
| **`pass_level`** | `str` | `"base_640px"`, `"enhanced_crop"` | Preprocessing path applied. `"enhanced_crop"` indicates adaptive contrast adjustment was triggered for small or blurry crops. |

---

### Expected Output Payload Examples

#### Standard Vehicle Detection
```json
{
  "schema_version": "1.0",
  "request_id": "gate-entry-01",
  "status": "OK",
  "candidates": [
    {
      "plate": "29D11077",
      "raw_plate": "29D11077",
      "confidence": 1.0,
      "detection_confidence": 0.8938,
      "bbox": [556, 1148, 874, 1319]
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 38.42
}
```

#### Negative Sample (No Plate in View)
* *Behavior*: Status remains `"OK"` because the pipeline executed successfully; `candidates` is an empty list:
```json
{
  "schema_version": "1.0",
  "request_id": "gate-entry-02",
  "status": "OK",
  "candidates": [],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 14.18
}
```

#### Syntax Anomaly Detection
* *Evidence*: Preserves raw OCR text and character probabilities when validation fails:
```json
{
  "schema_version": "1.0",
  "status": "OK",
  "candidates": [
    {
      "plate": "29X99999",
      "raw_plate": "29X99999",
      "confidence": 0.742,
      "detection_confidence": 0.812,
      "bbox": [320, 640, 580, 780],
      "is_valid": false,
      "ocr_min_confidence": 0.421,
      "character_details": [
        {"char": "2", "confidence": 0.992},
        {"char": "9", "confidence": 0.985},
        {"char": "X", "confidence": 0.421},
        {"char": "9", "confidence": 0.812}
      ],
      "region": "Vietnam",
      "region_prob": 0.994,
      "pass_level": "base_640px"
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 39.15
}
```

---

### Exception Hierarchy and Worker Template (`src/eiu_lpr/exceptions.py`)

All exceptions inherit from `EIULPRError`:
* **`EIULPRError`**: Base library exception.
* **`InvalidInputError`**: Raised on non-array, empty, non-`uint8`, or malformed input shapes.
* **`ModelNotFoundError`**: Raised when model weights (`.onnx`) or configs (`.yaml`) are missing.
* **`ConfigurationError`**: Raised when parameters are invalid (e.g. threshold outside `[0.0, 1.0]`).
* **`ModelInferenceError`**: Raised on ONNX Runtime session execution errors.

#### Worker Implementation Template
* Recommended pattern for wrapping camera frame ingest:

```python
import logging
from eiu_lpr import LPRPipeline
from eiu_lpr.exceptions import InvalidInputError, ModelInferenceError, EIULPRError

logger = logging.getLogger("cerberus_edge.lpr")

def process_camera_frame(pipeline: LPRPipeline, frame, request_id: str):
    try:
        result = pipeline.process(image=frame, color_format="BGR", request_id=request_id)
        return result.to_dict(minimal=True)

    except InvalidInputError as err:
        logger.warning(f"[{request_id}] Invalid camera frame input: {err}")
        return {"status": "ERROR", "error_code": "INVALID_IMAGE_FRAME", "message": str(err), "candidates": []}

    except ModelInferenceError as err:
        logger.error(f"[{request_id}] Neural network inference error: {err}")
        return {"status": "ERROR", "error_code": "INFERENCE_FAILURE", "message": "Vision engine execution failed.", "candidates": []}

    except EIULPRError as err:
        logger.error(f"[{request_id}] Vision engine error: {err}")
        return {"status": "ERROR", "error_code": "LPR_INTERNAL_ERROR", "message": str(err), "candidates": []}
```

---

## 4. Variable and Parameter References (`src/eiu_lpr/pipeline.py`)

### Visual Constructor Reference

```python
class LPRPipeline:
    def __init__(
        self,
        detector_path: str,
        ocr_path: str,
        ocr_config_path: str,
        detection_threshold: float = 0.20,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 10,
        ocr_mean_threshold: float = 0.70,
        ocr_min_threshold: float = 0.50,
        enable_adaptive_enhancement: bool = True,
        intra_op_threads: Optional[int] = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        schema_version: str = SCHEMA_VERSION,
    ):
```

---

### Initialization Parameters

* **`detector_path`** (`str`, required): Path to `models/detector.onnx`. Raises `ModelNotFoundError` if missing. Explicit paths prevent resolving against `os.getcwd()` inside Docker containers.
* **`ocr_path`** (`str`, required): Path to `models/cct_xs.onnx`. Raises `ModelNotFoundError` if missing.
* **`ocr_config_path`** (`str`, required): Path to `models/cct_xs_plate_config.yaml`. Raises `ModelNotFoundError` if missing.
* **`detection_threshold`** (`float`, default: `0.20`, range: `[0.0, 1.0]`): Minimum confidence for YOLO to propose a candidate box. Prioritizes recall in bad weather; false positives are cheaply rejected downstream by OCR and regex validation.
* **`nms_iou_threshold`** (`float`, default: `0.45`, range: `[0.0, 1.0]`): IoU overlap threshold for Non-Maximum Suppression to eliminate duplicate boxes on the same plate.
* **`max_detections`** (`int`, default: `10`, range: `>= 1`): Caps proposals evaluated per frame. Acts as an architectural circuit-breaker against CPU starvation during noisy scenes.
* **`ocr_mean_threshold`** (`float`, default: `0.70`, range: `[0.0, 1.0]`): Minimum average character softmax probability for confident reads. Recommended threshold for automated barrier opening.
* **`ocr_min_threshold`** (`float`, default: `0.50`, range: `[0.0, 1.0]`): Minimum softmax probability for the lowest-scoring character. Used to detect ambiguous characters without altering raw text.
* **`enable_adaptive_enhancement`** (`bool`, default: `True`): Toggles adaptive CLAHE and glare attenuation for small (`< 110x60` px) or blurry (Laplacian variance `< 150.0`) crops.
* **`intra_op_threads`** (`Optional[int]`, default: `None`): CPU threads allocated to ONNX Runtime. Set to `None` for dedicated single workers; set to `2` in multi-worker environments to prevent core contention.
* **`model_version`** & **`schema_version`** (`str`, defaults: `"alpr-v0.1"`, `"1.0"`): Telemetry identifiers attached to outgoing results.

---

### Runtime Call Parameters (`pipeline.process`)

* **`image`** (`np.ndarray`): 3D NumPy array of shape `[Height, Width, 3]` and data type `uint8`.
* **`color_format`** (`str`, default: `"RGB"`): Pass `"BGR"` for OpenCV frames (`cv2.VideoCapture`); pass `"RGB"` for RTSP streams via PyAV or PIL.
* **`detection_threshold`** (`Optional[float]`, default: `None`): Dynamic per-call threshold override without re-initializing the pipeline.
* **`request_id`** (`Optional[str]`, default: `None`): Correlation identifier echoed back in the response for distributed tracing.
* **`include_debug_crop`** (`bool`, default: `False`): Attaches in-memory crop array to `candidate.debug_crop`.
* **`include_annotated_image`** (`bool`, default: `False`): Attaches in-memory visualized frame to `result.annotated_image`.

---

## 5. Recommended Integration Patterns and Verification

*Precondition: All patterns assume `pipeline = LPRPipeline(...)` was initialized and warmed up once during worker boot.*

---

### Standard In-Memory Processing (`src/eiu_lpr/pipeline.py`)
* **If**: Processing standard camera frames where an end-to-end result and structured payload are required per frame.

```python
# Before this line: pipeline was initialized and frame was read into RAM
result = pipeline.process(frame, color_format="BGR", request_id="req-101")
payload = result.to_dict(minimal=True)
```

#### Terminal Execution Command
```bash
python -c "import cv2, json; from eiu_lpr import LPRPipeline; p = LPRPipeline('models/detector.onnx', 'models/cct_xs.onnx', 'models/cct_xs_plate_config.yaml'); img = cv2.imread('test_images/car_sample_29D.jpg'); res = p.process(img, color_format='BGR', request_id='req-101'); print(json.dumps(res.to_dict(minimal=True), indent=2))"
```
#### Verified Terminal Output
```json
{
  "schema_version": "1.0",
  "request_id": "req-101",
  "status": "OK",
  "candidates": [
    {
      "plate": "29D11077",
      "raw_plate": "29D11077",
      "confidence": 1.0,
      "detection_confidence": 0.7843,
      "bbox": [994, 1247, 1299, 1354]
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 360.33
}
```

---

### Split Workflow: Independent Detection and Recognition (`src/eiu_lpr/pipeline.py`)
* **If**: Tracking vehicles across video frames where you want to detect bounding boxes first and invoke OCR only once the plate stabilizes at the barrier stop line.

```python
# Before this line: pipeline was initialized and frame was read into RAM
# 1. Detect candidate boxes only
boxes = pipeline.detect(frame, color_format="BGR")

# 2. Invoke OCR selectively on the primary bounding box
if boxes:
    candidate = pipeline.read_plate(frame, bbox=boxes[0], color_format="BGR")
    print(f"Plate: {candidate.normalized_text} | Conf: {candidate.ocr_confidence:.1%}")
```

#### Terminal Execution Command
```bash
python -c "import cv2; from eiu_lpr import LPRPipeline; p = LPRPipeline('models/detector.onnx', 'models/cct_xs.onnx', 'models/cct_xs_plate_config.yaml'); img = cv2.imread('test_images/car_sample_29D.jpg'); boxes = p.detect(img, color_format='BGR'); cand = p.read_plate(img, boxes[0], color_format='BGR'); print(f'Plate: {cand.normalized_text} | Conf: {cand.ocr_confidence:.1%}')"
```
#### Verified Terminal Output
```text
Plate: 29D11077 | Conf: 100.0%
```

---

### In-Memory Visualization and Annotation (`src/eiu_lpr/pipeline.py`)
* **If**: Displaying real-time video streams in monitoring web dashboards or operator UIs with bounding boxes and plate labels.

```python
# Before this line: pipeline was initialized and frame was read into RAM
result = pipeline.process(frame, color_format="BGR", include_annotated_image=True)
annotated_frame = result.annotated_image  # np.ndarray [H, W, 3] uint8 in RAM

# Or invoke directly on existing candidates:
annotated_frame = pipeline.annotate(frame, result.candidates)
```

#### Terminal Execution Command
```bash
python -c "import cv2; from eiu_lpr import LPRPipeline; p = LPRPipeline('models/detector.onnx', 'models/cct_xs.onnx', 'models/cct_xs_plate_config.yaml'); img = cv2.imread('test_images/car_sample_29D.jpg'); res = p.process(img, color_format='BGR', include_annotated_image=True); print('Shape:', res.annotated_image.shape, '| Dtype:', res.annotated_image.dtype)"
```
#### Verified Terminal Output
```text
Shape: (2048, 1536, 3) | Dtype: uint8
```

---

### In-Memory Enhanced Plate Crops (`src/eiu_lpr/pipeline.py`)
* **If**: Auditing image quality, reviewing OCR input tensors, or saving cropped plate evidence for manual operator verification.

```python
# Before this line: pipeline was initialized and frame was read into RAM
result = pipeline.process(frame, color_format="BGR", include_debug_crop=True)
for i, candidate in enumerate(result.candidates):
    crop_array = candidate.debug_crop  # np.ndarray [H, W, 3] uint8 in RAM
```

#### Terminal Execution Command
```bash
python -c "import cv2; from eiu_lpr import LPRPipeline; p = LPRPipeline('models/detector.onnx', 'models/cct_xs.onnx', 'models/cct_xs_plate_config.yaml'); img = cv2.imread('test_images/car_sample_29D.jpg'); res = p.process(img, color_format='BGR', include_debug_crop=True); top = res.candidates[0]; print('Crop Shape:', top.debug_crop.shape, '| Pass Level:', top.pass_level)"
```
#### Verified Terminal Output
```text
Crop Shape: (115, 329, 3) | Pass Level: base_640px
```

---

### Dynamic Threshold Override (`src/eiu_lpr/pipeline.py`)
* **If**: Encountering sudden environmental changes (such as dusk, glare, or heavy monsoon rain) and needing a temporary high-recall sweep without re-initializing the pipeline.

```python
# Before this line: pipeline was initialized with default threshold 0.50
candidates = pipeline.recognize(frame, color_format="BGR", detection_threshold=0.10)
```

#### Terminal Execution Command
```bash
python -c "import cv2; from eiu_lpr import LPRPipeline; p = LPRPipeline('models/detector.onnx', 'models/cct_xs.onnx', 'models/cct_xs_plate_config.yaml', detection_threshold=0.50); img = cv2.imread('test_images/car_sample_29D.jpg'); res_default = p.detect(img, color_format='BGR'); res_override = p.detect(img, color_format='BGR', detection_threshold=0.10); print(f'Default (0.50): {len(res_default)} | Override (0.10): {len(res_override)}')"
```
#### Verified Terminal Output
```text
Default (0.50): 1 | Override (0.10): 1
```

---

### Multi-Worker CPU Concurrency Tuning (`src/eiu_lpr/pipeline.py`)
* **If**: Deploying multiple parallel worker processes behind Gunicorn or Uvicorn on multi-core edge servers.

```python
# Partition cores per worker process (e.g. 4 workers on an 8-core CPU = 2 threads each)
pipeline = LPRPipeline(
    detector_path="models/detector.onnx",
    ocr_path="models/cct_xs.onnx",
    ocr_config_path="models/cct_xs_plate_config.yaml",
    intra_op_threads=2,
    max_detections=10,
)
```

---

## 6. Done When Traceability Matrix

This section maps each requirement from the CerberusEdge integration specifications to its implementation in the codebase.

---

### Packaging and In-Memory Execution Contract

* **Wheel Packaging & Python 3.12** (`pyproject.toml`, `dist/eiu_lpr-0.1.0-py3-none-any.whl`): Configured packaging metadata and compiled a self-contained wheel artifact for zero-source deployment.
* **In-Memory Frame Processing** (`src/eiu_lpr/preprocessor.py`, `src/eiu_lpr/pipeline.py`): Accepts standard `uint8` 3D NumPy arrays directly from memory; converts BGR to RGB in RAM without filesystem interaction.
* **Constructor Dependency Injection** (`src/eiu_lpr/pipeline.py`): All asset paths and thresholds are passed directly into `__init__()`. Never resolves paths relative to `os.getcwd()`, preventing container path failures.
* **Zero-Disk-I/O Policy** (`src/eiu_lpr/pipeline.py`): Removed all directory creations and `cv2.imwrite` disk writes. Preserves flash storage lifespan and eliminates filesystem latency.
* **Privacy-Preserving Logging** (`src/eiu_lpr/models.py`): Removed all plate text print statements; candidate `__repr__` suppresses raw plate strings to protect vehicle PII in telemetry logs.
* **Input Validation & Non-Mutation** (`src/eiu_lpr/preprocessor.py`): Rejects non-arrays, empty frames, and invalid dtypes with `InvalidInputError`. Never mutates caller array memory in-place.

---

### API Modularity: Split Detection and Recognition

* **Independent `detect()` and `read_plate()` APIs** (`src/eiu_lpr/pipeline.py`): Enables callers to track moving boxes across video frames and invoke OCR only when a plate comes to a stop.
* **Unified Coordinate Core** (`src/eiu_lpr/pipeline.py`): Both `recognize()` and `read_plate()` share the exact same preprocessor and OCR model. Bounding box coordinates always reference original frame pixels `[x1, y1, x2, y2]`.

---

### Structured Per-Candidate Perception Scope

* **Per-Candidate Schema** (`src/eiu_lpr/models.py`): Emits structured `PlateCandidate` objects with both minimal (5-field) and full telemetry JSON serialization.
* **Empty Detection vs. Runtime Failure** (`src/eiu_lpr/pipeline.py`): Clear frames with zero plates return `status: "OK"` and `candidates: []`. Runtime errors raise typed exceptions, preventing false system alerts.
* **Perception-Only Scope** (`lab/`): Archived simulation scripts (`main.py`, `simulate_rfid.py`) and barrier relays to `lab/`. Perception library outputs perception metadata only.
* **Raw Evidence Retention on Invalid Syntax** (`src/eiu_lpr/pipeline.py`): When syntax validation fails, `is_valid` is set to `False`, but candidate bounding boxes, raw text, and softmax confidence scores are preserved.

---

### Canonical Normalisation and Validation Rules

* **Circular 24/2023 Compliance** (`src/eiu_lpr/validator.py`): Strips punctuation, joins two-line motorcycle plates top-to-bottom, and verifies province codes and series patterns.
* **Zero Heuristic Character Mutation** (`src/eiu_lpr/validator.py`): Avoids letter-to-digit swapping (`O` <-> `0`). Ambiguities are exposed through `character_details` softmax probabilities for backend database matching.

---

### Threshold Management and Operational Modes

* **Configurable Thresholds** (`src/eiu_lpr/pipeline.py`): Defaults `detection_threshold=0.20` to maximize recall in rain and glare; downstream OCR thresholds default to `0.70` mean and `0.50` min confidence.
* **Circuit-Breaker Proposal Capping** (`src/eiu_lpr/detector.py`): Caps candidate proposals at `max_detections=10` per frame to prevent CPU starvation on noisy scenes.

---

### Pure ONNX Runtime, Warmup, and Concurrency

* **Ultralytics & PyTorch Elimination** (`src/eiu_lpr/detector.py`): Reimplemented letterboxing, ONNX evaluation, and NMS directly in NumPy, dropping `ultralytics` imports.
* **Pinned Dependency Constraints** (`constraints.txt`): Pins compatible versions of `numpy`, `onnxruntime`, `opencv-python-headless`, and `fast-plate-ocr` for Python 3.10 to 3.12.
* **Explicit Pipeline Warmup** (`src/eiu_lpr/pipeline.py`): `warmup()` runs dummy passes during service startup to allocate memory kernels and eliminate initial vehicle latency spikes.
* **Multi-Worker Concurrency Support** (`src/eiu_lpr/detector.py`, `src/eiu_lpr/recognizer.py`): Supports `intra_op_threads` for clean CPU core partitioning across Gunicorn or Uvicorn worker pools.


---

## 7. Release Engineering: Upgrading Models, Versions, and Git Tags

This runbook documents the exact step-by-step process for upgrading model weights, bumping release versions, building new distribution wheels, and publishing Git tags.

---

### Step 1: Upgrading Model Weights (`models/`)

1. **Replace ONNX Weights or Configs**
   * Copy the newly trained or exported `.onnx` model files into `models/`:
     * Detector: `models/detector.onnx`
     * Recognizer: `models/cct_xs.onnx`
     * Charset/Config: `models/cct_xs_plate_config.yaml`

2. **Regenerate SHA-256 Checksums (`scripts/generate_checksums.py`)**
   * Run the checksum generation utility to update the cryptographic manifest:
     ```bash
     python scripts/generate_checksums.py
     ```
   * Verify the manifest matches the new binary files on disk:
     ```bash
     # On Linux:
     sha256sum -c models/checksums.sha256

     # On Windows (PowerShell):
     Get-FileHash models/detector.onnx, models/cct_xs.onnx, models/cct_xs_plate_config.yaml -Algorithm SHA256
     ```

---

### Step 2: Verification and Regression Testing

Before releasing any updated model, run the full verification battery:

1. **Run Unit Tests**:
   ```bash
   python run_tests.py
   ```
   *Confirms that in-memory APIs, error handling, split methods, and zero-disk-I/O contracts remain intact.*

2. **Run Golden Dataset Regression**:
   ```bash
   python tests/test_golden_dataset.py -v
   ```
   *Verifies accuracy across all 20 reference scenarios (cars, motorcycles, low light, rain, and negative frames).*

3. **Run CPU Latency Benchmark**:
   ```bash
   python benchmark_linux_cpu.py --iterations 3
   ```
   *Confirms that inference speed meets edge latency budgets (P50, P90, P99).*

---

### Step 3: Semantic Versioning Rules and Bump Locations

Follow Semantic Versioning (`MAJOR.MINOR.PATCH`):
* **`MAJOR` (e.g. `1.0.0`)**: Incompatible API breaking changes or schema overhauls (`schema_version = "2.0"`).
* **`MINOR` (e.g. `0.2.0`)**: New model weights (improved recall/precision), new optional parameters, or backward-compatible feature additions.
* **`PATCH` (e.g. `0.1.1`)**: Internal bug fixes, dependency constraint updates, or documentation refinements.

#### Where to Update Version Numbers (Update All 3 Locations)

1. **`pyproject.toml`** (line 7):
   ```toml
   [project]
   version = "0.2.0"
   ```

2. **`src/eiu_lpr/__init__.py`** (line 24):
   ```python
   __version__ = "0.2.0"
   ```

3. **`src/eiu_lpr/models.py`** (line 8):
   ```python
   DEFAULT_MODEL_VERSION = "alpr-v0.2"
   ```

---

### Step 4: Rebuilding the Distribution Wheel (`dist/`)

1. **Compile the Wheel Artifact**:
   ```bash
   python -m pip wheel --no-build-isolation --no-deps -w dist .
   ```
   *Generates `dist/eiu_lpr-0.2.0-py3-none-any.whl`.*

2. **Test Clean Installation**:
   ```bash
   pip install --force-reinstall dist/eiu_lpr-0.2.0-py3-none-any.whl -c constraints.txt
   ```

---

### Step 5: Git Commit, Tagging, and Pushing

1. **Stage Modified Files**:
   ```bash
   git add pyproject.toml src/eiu_lpr/ models/ dist/
   ```

2. **Commit Release**:
   ```bash
   git commit -m "release: v0.2.0 - update model weights and bump version"
   ```

3. **Create Annotated Git Tag**:
   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0: updated model weights and performance improvements"
   ```

4. **Push Branch and Tags to Remote**:
   ```bash
   git push origin main --tags
   ```
