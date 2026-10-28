# Production Integration and Packaging Plan

Date: 2026-10-27  
Target Repository: `c:\Users\User\EIU_OITP\lpr_research`  
Target Host Architecture: Linux Edge Worker (CerberusEdge), Python 3.12  
Execution Context: Central Processing Unit (CPU) edge inference without dedicated graphics hardware  
Objective: Refactor the research pipeline into an installable, in-memory, production-grade vision library fulfilling all ten technical requirements established by the backend systems review.

---

## 1. Executive Overview and Engineering Shift

The transition from research prototype to production library requires four core structural changes:

1. In-Memory Data Flow: Replace all filesystem calls (`cv2.imread`, `cv2.imwrite`, directory generation in `Outputs/`) with in-memory NumPy arrays (`np.ndarray` in RGB uint8 `[H, W, 3]` format).
2. Decoupled Functional APIs: Expose three clean interfaces:
   - `detect(image)` for spatial bounding boxes.
   - `read_plate(image, bbox)` for cropping, enhancement, character recognition, and syntax validation.
   - `recognize(image)` as a combined convenience pipeline.
3. Pure ONNX Runtime Inference: Eliminate the `ultralytics` package from the inference runtime. Run both the detector and the recognizer directly on `onnxruntime.InferenceSession` with explicit session warmup.
4. Business Logic Separation: Strip all Radio Frequency Identification (RFID) card checks, vehicle database lookups, and boom barrier access control decisions from the vision package. Return pure visual candidate data.

Deliverable Format:
The deliverable is an installable Python wheel package (`.whl`), not an HTTP REST service. The existing `main.py` (FastAPI) and `simulate_rfid.py` scripts remain in the repository as local testing and evaluation tools. The production backend (`CerberusEdge`) imports the library directly into its vision worker process, integrating it through its internal `Recognizer(Protocol)` adapter.

---

## 2. Target Package Directory Structure: The Lab vs. Factory Separation

To prevent breaking existing research notebooks or benchmarks while building the production library, the repository maintains a strict separation between the installable package (`src/eiu_lpr/`) and the research evaluation harness:


- lets create another branch and stage edits/

```
EIU_YOLO_PaddleOCR_Licenseplatedetection/
│
├── pyproject.toml                     # Python packaging file (builds only src/eiu_lpr)
├── README.md                          # Repository documentation
├── LICENSE                            # MIT License
│
├── src/eiu_lpr/                       # THE FACTORY: Production Vision Library (Clean)
│   ├── __init__.py                    # Public API exports (LPRPipeline, BoundingBox, PlateCandidate)
│   ├── pipeline.py                    # Core pipeline orchestrating detect(), read_plate(), recognize(), warmup()
│   ├── detector.py                    # Direct ONNX Runtime detector (pure NumPy preprocessing + NMS)
│   ├── recognizer.py                  # CCT-XS ONNX recognizer adapter (in-memory RGB crops)
│   ├── preprocessor.py                # In-memory crop padding (+4%) and mild adaptive enhancement
│   ├── validator.py                   # Circular 24/2023 normalization and regex validation
│   ├── models.py                      # Data models: BoundingBox, PlateCandidate, PipelineResult
│   └── exceptions.py                  # Custom typed exceptions (InvalidInputError, ModelError)
│
├── models/                            # THE ASSETS: Frozen release weights (SHA-256 verified)
│   ├── detector.onnx                  # balanced_skew_robust_yolo11n.onnx (10.9 MB)
│   ├── cct_xs.onnx                    # Compact Convolutional Transformer OCR model (1.8 MB)
│   ├── cct_xs_plate_config.yaml       # Recognizer character vocabulary and input geometry
│   └── checksums.sha256               # Cryptographic SHA-256 verification hashes
│
├── benchmarks/                        # THE LAB: Retained evaluation harness (Untouched)
│   ├── run_benchmarks.py              # Batch benchmark runner against test suites
│   ├── BENCHMARK_SUMMARY.md           # Master historical performance ledger
│   └── benchmark_linux_cpu.py         # Dedicated Linux CPU latency test script
│
├── test_images/                       # 18-image regression test suite
├── ColabNotebookTemplate/             # Research training notebooks (Untouched)
├── simulate_rfid.py                   # Local access control simulation script
├── main.py                            # Local FastAPI demo server
└── tests/                             # Unit and integration test suite
    ├── test_in_memory_api.py          # Verifies ndarray input, zero disk writes, zero prints
    ├── test_split_apis.py             # Verifies detect() and read_plate() consistency
    ├── test_onnx_runtime.py           # Verifies pure ONNX execution without ultralytics imports
    └── test_warmup.py                 # Verifies session priming and steady-state latency
```

---

## 3. Phase 1: Blocking Deliverables (Must Complete First)

### Item 1: Importable Package with In-Memory API
- Input Contract: Methods accept `image: np.ndarray` with shape `[H, W, 3]` and dtype `uint8`.
- Color Space Handling (RGB vs. BGR):
  - CerberusEdge frames arrive in memory as **RGB**.
  - OpenCV workflows natively use **BGR**.
  - To support both without ambiguity, APIs accept `color_format: str = "RGB"` by default. If `color_format="BGR"` is passed (e.g. from OpenCV `cv2.VideoCapture`), the pipeline automatically converts the channels internally.
- Zero Disk I/O: No `cv2.imread()`, no `cv2.imwrite()`, and no automatic creation of `Outputs/` directories.
- Privacy Compliance: No `print()` calls and no logging of plate text to stdout/stderr. License plate text is Personally Identifiable Information (PII) and must only be returned in data structures.
- Constructor Paths: Pass model weights and configuration file paths explicitly into the constructor; never resolve paths relative to the current working directory (`os.getcwd()`).
- Done When: The package installs via `pip install eiu_lpr-*.whl` in a clean Python 3.12 Linux environment, executes from an unrelated directory, creates zero files, emits zero console output, and rejects invalid inputs with documented exceptions.

### Item 2: Separate Detection from Plate Reading
- `detect(image: np.ndarray, color_format: str = "RGB") -> list[BoundingBox]`: Runs detector ONNX inference at 640px base resolution. Returns all detected boxes with original image pixel coordinates `[x1, y1, x2, y2]` and detector confidence scores.
- `read_plate(image: np.ndarray, bbox: BoundingBox, color_format: str = "RGB") -> PlateCandidate`: Takes the image and a specific bounding box. Applies 4% padding (clamped between 3 and 12 pixels), crops the region, runs Compact Convolutional Transformer (CCT-XS) Optical Character Recognition (OCR), and performs Vietnamese regulatory validation.
- `recognize(image: np.ndarray, color_format: str = "RGB") -> list[PlateCandidate]`: Convenience wrapper executing detection followed by plate reading across all detected boxes.
- Retirement of Legacy Geometric Slicing: Functions `deskew_and_slice_plate` and `canonical_perspective_unwarp` are formally deprecated and omitted from `eiu_lpr`. CCT-XS natively processes dual-row motorcycle plates in a single forward pass without horizontal image slicing.
- Done When: A runnable test script demonstrates calling `detect()` across sequential frames, selecting a single stable box, and calling `read_plate()` on that box without re-running object detection.

### Item 3: Structured Results with Recognition-Only Scope
- Candidate Data Model (`PlateCandidate`):
  - `bbox`: Tuple `(x1, y1, x2, y2)` in original image coordinates.
  - `detection_score`: Float between 0.0 and 1.0.
  - `raw_text`: Exact raw string output from the OCR model.
  - `normalized_text`: Cleaned alphanumeric string without separators.
  - `is_valid`: Boolean indicating adherence to Vietnamese Circular 24/2023 syntax.
  - `ocr_confidence`: Mean character softmax probability.
  - `ocr_min_confidence`: Lowest individual character confidence.
  - `character_details`: List of tuples `(char, prob)`.
  - `region`: Geographical classification label (e.g., `Vietnam`).
  - `region_prob`: Confidence of regional classification.
  - `pass_level`: String metadata (`base_640px`, `enhanced_crop`, or `retry_1024px`).
- Mapping to CerberusEdge Internal Types:
  - `PlateCandidate.bbox` maps to `cerberus_edge.worker.pipeline.model.BoundingBox`.
  - `PlateCandidate.normalized_text` maps to `Candidate.normalized_text`.
  - `PlateCandidate.detection_score` maps to `Candidate.detection_score`.
  - `PlateCandidate.ocr_confidence` maps to `Candidate.ocr_score`.
- Scope Reduction: Delete `REGISTERED_VEHICLES`, RFID matching, and `GATE_OPEN/GATE_CLOSED` logic from the package.
- Reference Output JSON Schema:
  ```json
  {
    "schema_version": "1.0",
    "request_id": "req-001",
    "status": "OK",
    "candidates": [{
      "plate": "51A12345",
      "raw_plate": "51A-123.45",
      "confidence": 0.94,
      "detection_confidence": 0.97,
      "bbox": [842, 531, 1112, 645]
    }],
    "model_version": "alpr-v0.1",
    "processing_time_ms": 87
  }
  ```
- Done When: The API returns an empty candidate list `[]` for frames with no plates, returns structured records for valid plates, and accepts no database or access control parameters.

### Item 6: Direct ONNX Runtime, Pinned Dependencies, and Warmup
- Eliminate Ultralytics: Implement a lightweight `ONNXYoloDetector` using `onnxruntime.InferenceSession`. Handle letterbox resizing, normalization (`/ 255.0`), transpose (`HWC -> CHW`), and bounding box Non-Maximum Suppression (NMS) using pure NumPy.
- Package Dependencies: Pin dependencies in `pyproject.toml` (`onnxruntime>=1.19.0`, `numpy>=1.26.0`, `opencv-python-headless>=4.8.0`, `fast-plate-ocr>=1.1.0`, `pyyaml>=6.0`). Note the use of `opencv-python-headless` to avoid Linux X11/GUI library dependencies.
- Session Warmup: Implement `pipeline.warmup(iterations: int = 2)`. Prime both detector and OCR sessions using dummy input tensors during service startup.
- Concurrency Model: Document that each Linux worker process should maintain its own pipeline instance.
- Done When: A clean Linux CPU environment runs the pipeline without the `ultralytics` package installed, warms up on initialization, and processes repeated frames without memory leaks or session re-creations.

### Item 7: Versioned Model Release and Model Card
- Freeze Weights: Select exactly one detector model (`balanced_skew_robust_yolo11n.onnx`) and one recognizer model (`cct_xs.onnx`).
- Cryptographic Verification: Generate a `checksums.sha256` file containing SHA-256 hashes for both ONNX files and the YAML configuration.
- Model Card: Create `MODEL_CARD.md` documenting input dimensions, color space (RGB), supported plate formats (single-line car, two-line motorcycle, modern two-letter series), benchmark metrics, and operational limitations (e.g., Dutch roll angles beyond 45 degrees).
- Done When: A tagged GitHub Release contains the wheel distribution, the frozen ONNX models, the YAML configuration, and matching checksums.

---

## 4. Phase 2: Non-Blocking Deliverables (Can Follow Initial Release)

### Item 4: Canonical Plate Normalization Specification
- Document the exact normalization sequence:
  1. Uppercase transformation.
  2. Removal of non-alphanumeric characters (spaces, hyphens, dots, bolts).
  3. Two-line joining order: Top line followed immediately by bottom line.
  4. Explicit statement confirming no ambiguous character substitution (e.g., no automatic swapping between letter 'O' and digit '0').
- File: `docs/NORMALIZATION_SPEC.md`.

### Item 5: Configurable Thresholds and Recommendations
- Parameterize all operational thresholds in the constructor:
  - `detection_threshold: float = 0.20` (recommended default).
  - `ocr_mean_threshold: float = 0.70`.
  - `ocr_min_threshold: float = 0.50`.
  - `enable_adaptive_enhancement: bool = True`.
- Engineering Rationale for 0.20 Threshold:
  - Tropical weather (monsoon rain, fog, headlight glare) degrades initial detector confidence on real plates into the 0.25 to 0.45 range.
  - Setting `detection_threshold = 0.20` achieves high recall without missing challenging vehicles.
  - Downstream CCT-XS OCR softmax confidence and Vietnamese regex syntax validation act as secondary filters, safely discarding false-positive bounding boxes before any plate record is accepted.
- File: `docs/CONFIGURATION_GUIDE.md`.

### Item 8: Linux CPU Latency Benchmark
- Provide an automated benchmark script (`benchmarks/benchmark_linux_cpu.py`) executing on Linux.
- Report metrics across 100 test iterations:
  - Cold startup latency (including model load and first inference).
  - Warm steady-state detection latency.
  - Warm steady-state OCR latency.
  - Combined end-to-end pipeline latency (p50, p95, and maximum).

### Item 9: Golden Test Suite with Human Verification
- Deliver a curated dataset of 20 verified test images in `golden_dataset/` with an accompanying `ground_truth.json`.
- Cover: single-line car, dual-line motorcycle, two-letter series, rainy/foggy scenes, steep angles, and negative scenes (no vehicles).
- Include written confirmation of provenance and storage permissions.

### Item 10: Licensing and Legal Documentation
- Add an open-source `LICENSE` file (MIT License) to the repository root.
- Document license compliance:
  - Pipeline framework code: MIT License.
  - OCR Engine (`fast-plate-ocr` / CCT-XS): MIT License.
  - ONNX Runtime: MIT License.
  - Model weights: Open research distribution.

---

## 5. Implementation Roadmap and Task Checklist

```
[Phase 1: Core Library Refactor]
├── Task 1.1: Create eiu_lpr/ models.py with dataclasses
├── Task 1.2: Implement eiu_lpr/detector.py using pure ONNX Runtime (no ultralytics)
├── Task 1.3: Refactor eiu_lpr/recognizer.py for pure in-memory RGB crops
├── Task 1.4: Refactor eiu_lpr/preprocessor.py and validator.py (strip prints/disk I/O)
├── Task 1.5: Build eiu_lpr/pipeline.py with detect(), read_plate(), recognize(), and warmup()
├── Task 1.6: Configure pyproject.toml with opencv-python-headless
└── Task 1.7: Write verification test suite in tests/

[Phase 2: Packaging, Checksums, and Documentation]
├── Task 2.1: Generate SHA-256 checksums for release models
├── Task 2.2: Write MODEL_CARD.md and docs/NORMALIZATION_SPEC.md
├── Task 2.3: Build .whl wheel distribution
├── Task 2.4: Validate installation and execution in clean Linux container
└── Task 2.5: Publish tagged GitHub release
```

note for me;
1. to keep our library stateless

