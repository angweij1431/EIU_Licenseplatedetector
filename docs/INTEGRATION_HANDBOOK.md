# Developer Integration and Operations Handbook

Practical reference and operational guide for integrating and maintaining the `eiu_lpr` vision engine.

---

## 1. How Do I Integrate This?

### A. The Core Integration Pattern
The library is designed for in-memory CPU inference in edge worker services (e.g., CerberusEdge). It accepts raw image byte arrays from camera streams, executes detection and character recognition, and emits structured results without touching the filesystem.

```python
import cv2
from eiu_lpr import LPRPipeline

# 1. Initialize pipeline once during service startup
pipeline = LPRPipeline(
    detector_path="models/detector.onnx",
    ocr_path="models/cct_xs.onnx",
    ocr_config_path="models/cct_xs_plate_config.yaml",
    detection_threshold=0.20,
)

# 2. Prime ONNX execution providers
pipeline.warmup(iterations=2)

# 3. Ingest frame from camera (in-memory numpy array [H, W, 3], uint8)
frame = cv2.imread("test_images/car_test_2.jpg")

# 4. Recognize and retrieve candidate records
candidates = pipeline.recognize(frame, color_format="BGR")
for cand in candidates:
    print(f"Plate: {cand.normalized_text}, Conf: {cand.ocr_confidence:.3f}, Valid: {cand.is_valid}")
```

### B. High-Throughput Video Tracking Pattern
In continuous RTSP video processing, do not run OCR on every video frame. Run fast 640px bounding box detection on every frame, track boxes across time, and trigger plate reading only when a box stabilizes:

```python
# Step 1: Detect bounding boxes on incoming video frame (~15-30ms)
boxes = pipeline.detect(frame, color_format="BGR")

# Step 2: Read characters only from the selected stable box (~20-40ms)
if boxes:
    best_box = boxes[0]  # Or highest-confidence tracked box
    candidate = pipeline.read_plate(frame, bbox=best_box, color_format="BGR")
    if candidate.is_valid and candidate.ocr_confidence >= 0.70:
        # Forward to access decision worker
        print(f"Verified Plate: {candidate.normalized_text}")
```

### C. Standardized Edge Backend Serialization
To emit the exact JSON payload expected by the CerberusEdge worker schema:

```python
# Minimal 5-field reference schema
result = pipeline.process(frame, color_format="BGR", request_id="req-101")
payload = result.to_dict(minimal=True)

# Full telemetry evidence schema (includes character softmax probabilities and bounding boxes)
full_evidence = result.to_dict(minimal=False)
```

---

## 2. How Do I Do This / That? (Operational FAQs)

### How do I handle OpenCV frames vs camera RGB streams? (RGB vs. BGR Guide)

#### Technical Difference: Byte Channel Ordering
In memory, both formats store a 3D NumPy array of shape `[Height, Width, 3]` with `uint8` values from 0 to 255. The only difference is the sequence of the 3 color channels across the depth dimension:

| Channel Index | RGB Format (Standard Vision) | BGR Format (OpenCV Default) |
| :---: | :---: | :---: |
| `[..., 0]` | **Red** channel | **Blue** channel |
| `[..., 1]` | **Green** channel | **Green** channel |
| `[..., 2]` | **Blue** channel | **Red** channel |

#### Why Does OpenCV Use BGR?
OpenCV was originally developed at Intel in the late 1990s. At that time, many hardware frame grabbers and Windows bitmap structures (`BITMAPINFOHEADER`) natively stored pixels in BGR order. OpenCV kept BGR as its default format for backward compatibility, whereas modern web standards, camera SDKs, PIL, PyAV, GStreamer, and PyTorch standardize on RGB.

#### What Happens If Color Formats Are Swapped in Neural Networks?
Both the YOLO plate detector and the CCT-XS transformer OCR model were trained on RGB data:
1. **Inverted Chrominance**: Red and Blue channels are swapped. A yellow commercial plate (high Red + Green) appears cyan (high Blue + Green) to the model.
2. **Confidence Drop**: While plate detection can often still find the rectangular boundary using luminance edges, the OCR transformer confidence drops significantly, leading to misread characters and failed validations.
3. **Preprocessing Distortion**: Contrast enhancement (CLAHE) and Retinex glare attenuation depend on accurate luminance and chrominance separation.

#### How `eiu_lpr` Handles This Automatically
The library handles channel conversion in RAM without temporary disk writes:

- **When receiving frames from RTSP streams, PyAV, PIL, or GStreamer (RGB native)**:
  ```python
  # Default is RGB - no conversion overhead
  candidates = pipeline.recognize(rgb_frame, color_format="RGB")
  ```

- **When reading images or video via OpenCV (`cv2.imread`, `cv2.VideoCapture`) (BGR native)**:
  ```python
  # Pass color_format="BGR" - automatically converted to RGB in memory
  bgr_frame = cv2.imread("plate.jpg")
  candidates = pipeline.recognize(bgr_frame, color_format="BGR")
  ```

### How do I retrieve in-memory debug images and crops?
The library never writes debug files to disk, preserving storage lifespan on edge devices. Both plate crops and annotated frames are available directly as NumPy arrays in RAM:

#### 1. Retrieve In-Memory Enhanced Plate Crops
Pass `include_debug_crop=True` to `read_plate()`, `recognize()`, or `process()`. The cropped, padded, and enhanced image array fed into OCR is attached to `candidate.debug_crop`:
```python
candidates = pipeline.recognize(frame, color_format="BGR", include_debug_crop=True)
for cand in candidates:
    crop_array = cand.debug_crop  # np.ndarray [H, W, 3], uint8
    if crop_array is not None:
        print(f"Crop shape: {crop_array.shape}")
```

#### 2. Retrieve In-Memory Annotated Full-Frame Image
Pass `include_annotated_image=True` to `pipeline.process()`, or invoke `pipeline.annotate()` directly:
```python
# Option A: Through pipeline.process()
result = pipeline.process(frame, color_format="BGR", include_annotated_image=True)
annotated_frame = result.annotated_image  # np.ndarray [H, W, 3] with drawn boxes & labels

# Option B: Direct call on any existing candidates list
annotated_frame = pipeline.annotate(frame, result.candidates)
```

### How do I tune thresholds for adverse weather (rain, fog, glare)?
- The default `detection_threshold=0.20` is tuned for high recall under monsoon rain, headlight glare, and dust.
- You can override the threshold dynamically for individual calls without re-initializing the pipeline:
  ```python
  boxes = pipeline.detect(frame, detection_threshold=0.35)
  candidates = pipeline.recognize(frame, detection_threshold=0.35)
  ```

### How do I control CPU resource usage across worker processes?
To avoid CPU spikes and thread contention across multiple edge workers:
1. **Limit ONNX Runtime Cores**: Set `intra_op_threads` during initialization.
2. **Cap Max Proposals**: The pipeline defaults to `max_detections=10`. This acts as an architectural circuit-breaker, ensuring that even under extreme noise, headlight flare, or foliage, the CPU never evaluates more than 10 candidate crops in a single frame.

```python
pipeline = LPRPipeline(
    detector_path="models/detector.onnx",
    ocr_path="models/cct_xs.onnx",
    ocr_config_path="models/cct_xs_plate_config.yaml",
    intra_op_threads=2,    # Limits execution to 2 CPU cores
    max_detections=10,     # Safeguards against noisy scenes running excessive OCR passes
)
```

### How do I eliminate startup latency spikes?
Run `pipeline.warmup(iterations=2)` immediately after instantiating `LPRPipeline`. This forces ONNX Runtime to allocate memory tensors and build execution kernels before the first live vehicle frame arrives.

### When does `enable_adaptive_enhancement` trigger?
`enable_adaptive_enhancement` is enabled (`True`) by default, and triggers `PlatePreprocessor.mild_enhance_for_transformer()` **only when a plate crop is genuinely blurred (`blur_score < 150.0`)**.
- **Why blur-only (not crop size)**: Field testing on sharp downscaled overhead motorcycle feeds (e.g., `1024x576`, where plate crops are ~`63x53` px but sharp with `blur_score > 7000`) showed that running 2x2 morphological stroke erosion on sharp small crops distorted tight dual-line characters (`7` into `1`, `E` into `G`).
- **Channel safety**: `pipeline.read_plate()` converts `BGR` crops to `RGB` before computing blur scores and applying Retinex/CLAHE enhancement.
- **Best practice**: Feed native camera resolution whenever possible; set `enable_adaptive_enhancement=False` only if you want raw crops in all conditions.

---

## 3. What About This? What to Do About This? (Edge Cases & Caveats)

### What about ambiguous characters (letter 'O' vs digit '0', letter 'I' vs digit '1')?
- **Policy**: The vision engine never performs manual character substitution.
- **Why**: Heuristic character swapping corrupts valid plates and masks visual ambiguity.
- **What to do**: Inspect `candidate.character_details`. The engine outputs the exact character together with its softmax probability score (e.g. `[("3", 0.999), ("0", 0.980), ("A", 1.0)]`). The downstream backend can evaluate character alternatives against its database of registered vehicles.

### What about false positive bounding boxes?
- A low detection threshold (`0.20`) may occasionally propose non-vehicle regions.
- **Filter**: Downstream CCT-XS OCR softmax confidence (`ocr_confidence < 0.70`) and Circular 24/2023 regex validation (`is_valid == False`) automatically filter out false alarms.

### What about dual-line motorcycle plates?
- CCT-XS recognizes both lines simultaneously in a single forward pass.
- Geometric slicing (`deskew_and_slice_plate`) is retired. The pipeline preserves natural top-to-bottom reading order automatically.

### What about bounding box coordinate space?
- YOLO operates internally on a 640x640 letterboxed square.
- The detector automatically maps coordinates back to the original image pixel resolution `(x1, y1, x2, y2)`. Crops and overlays align with the camera frame.

---

## 4. What Are the Working Steps to Version Properly?

Follow this step-by-step checklist whenever publishing a new version of the vision engine:

### Step 1: Update Source Code or Model Assets
Make any necessary code changes under `src/eiu_lpr/` or update weights in `models/`.

### Step 2: Run Verification Test Suite
Execute the full test suite and confirm all 10 unit tests and golden dataset pass:
```bash
python run_tests.py
python tests/test_golden_dataset.py
```

### Step 3: Regenerate Cryptographic Checksums
If model weights or `cct_xs_plate_config.yaml` changed, update the SHA-256 manifest:
```bash
python scripts/generate_checksums.py
```
Verify hashes match on disk:
```bash
# On Linux:
sha256sum -c models/checksums.sha256
```

### Step 4: Bump Version Numbers
Update the version string in two places:
1. `pyproject.toml`:
   ```toml
   [project]
   version = "0.2.0"
   ```
2. `src/eiu_lpr/__init__.py`:
   ```python
   __version__ = "0.2.0"
   ```

### Step 5: Build Distribution Wheel
Build the self-contained wheel artifact:
```bash
python -m pip wheel --no-build-isolation --no-deps -w dist .
```

### Step 6: Commit and Tag Release in Git
```bash
# Stage updated files
git add src/ models/ pyproject.toml dist/ docs/

# Commit with a clear description
git commit -m "Bump version to v0.2.0: update model weights and documentation"

# Create annotated release tag
git tag -a v0.2.0 -m "Release v0.2.0"
```

### Step 7: Deploy Wheel to Edge Worker
Copy the `.whl` from `dist/` and install on the target edge host:
```bash
pip install --upgrade eiu_lpr-0.2.0-py3-none-any.whl
```

---

## 5. Optional Future Extensions (Copy-Paste Templates — On Hold)

These patterns are intentionally kept **outside** `eiu_lpr` core to avoid overengineering. Paste and adapt them into your edge worker later if needed.

### A. Lane ROI Overlap Filter (Ignore Adjacent Lanes / Pedestrians Outside Lane)
Filter `pipeline.detect()` boxes or `result.candidates` against a lane Region of Interest (`roi = (rx1, ry1, rx2, ry2)` in original image pixels):

```python
def is_inside_roi(bbox, roi, min_overlap_ratio=0.5):
    """Returns True if at least min_overlap_ratio of bbox area lies inside roi."""
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi

    ix1, iy1 = max(x1, rx1), max(y1, ry1)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    box_area = max(1e-6, (x2 - x1) * (y2 - y1))
    return (inter / box_area) >= min_overlap_ratio

# Example usage (tune LANE_ROI to your camera's pixel coordinates, e.g. 2880x1620):
LANE_ROI = (1000, 400, 1900, 1300)
result = pipeline.process(frame, color_format="BGR")
lane_candidates = [c for c in result.candidates if is_inside_roi(c.bbox, LANE_ROI)]
```

### B. Motion-Burst Best-Frame Selector (Multi-Shot Motion Trigger)
If the edge worker buffers multiple frames while motion occurs and selects a winner when motion stops, rank candidate frames by **plate validity and minimum character confidence** rather than pixel sharpness alone (which helps avoid frames where an overhanging taillight occludes the top row):

```python
def select_best_burst_result(pipeline, frames, color_format="BGR", lane_roi=None):
    """Evaluates a small burst (e.g., top 3-5 frames) and returns the strongest read."""
    best_result = None
    best_score = (-1, -1.0, -1.0)  # (is_valid, ocr_min_confidence, ocr_confidence)

    for frame in frames:
        res = pipeline.process(frame, color_format=color_format)
        cands = res.candidates
        if lane_roi is not None:
            cands = [c for c in cands if is_inside_roi(c.bbox, lane_roi)]

        if not cands:
            continue

        top = max(cands, key=lambda c: (int(c.is_valid), c.ocr_min_confidence, c.ocr_confidence))
        score = (int(top.is_valid), top.ocr_min_confidence, top.ocr_confidence)
        if score > best_score:
            best_score = score
            best_result = res

    return best_result
```

### C. Future Option: Adding `vehicle_class` to Telemetry JSON (On Hold)
The current production schema (`1.0`) intentionally omits `vehicle_class` to keep the edge contract unchanged. Note the key findings if you revisit adding vehicle classification later:
1. **Current `models/detector.onnx` (`v0.3.0`) is plate-only**: Its ONNX output tensor is `[1, 8400, 6]` (`num_classes = 1`, `plate` only), so it cannot visually detect vehicle bodies (`motorcycle` / `car`).
2. **Off-the-shelf COCO models fail on steep overhead motorcycle views**: Testing an 80-class COCO Nano model on overhead parking-lane shots yielded `motorcycle` confidence of `0.005–0.097` on 3 of 4 motorcycles because riders obscure the bike body from above.
3. **Two future paths if needed**:
   - **Zero-model (plate-inferred)**: Expose `vehicle_type` from [validator.py](../src/eiu_lpr/validator.py) using Circular 24/2023 digit lengths (`{4,5}`) plus plate bounding-box aspect ratio (`w/h < 1.65`) for 8-char disambiguation. Zero latency cost, but cannot detect plate-less vehicles and can mislabel heavily angled square 2-line car plates as `"motorcycle"`.
   - **True visual vehicle detection (for retaining plate-less vehicles)**: Fine-tune a lightweight Apache 2.0 `yolox_nano` on ~300–500 labeled frames from the target overhead camera (or lower camera pitch to 15°–30° so standard COCO models regain recall) and load it as an optional pre-capture gate (`detect_vehicles()`).

