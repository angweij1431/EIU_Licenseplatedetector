# EIU Automated License Plate Recognition (ALPR) Engine

Production-grade in-memory Vietnamese license plate detection and character recognition library optimized for edge Central Processing Unit (CPU) deployment.

Designed for edge worker integration (e.g., CerberusEdge) running Python 3.12 on Linux, with zero external cloud dependencies, zero disk I/O, and zero console logging of Personally Identifiable Information (PII).

---

## 1. Quickstart

### Installation
Install the pre-built standalone wheel package:

```bash
pip install dist/eiu_lpr-0.1.0-py3-none-any.whl
```

### Basic Usage
Process an in-memory frame (`np.ndarray`, `uint8`, shape `[H, W, 3]`):

```python
import cv2
from eiu_lpr import LPRPipeline

# Initialize pipeline
pipeline = LPRPipeline(
    detector_path="models/detector.onnx",
    ocr_path="models/cct_xs.onnx",
    ocr_config_path="models/cct_xs_plate_config.yaml",
)
pipeline.warmup(iterations=2)

# Ingest frame (RGB by default; set color_format='BGR' for OpenCV)
image = cv2.imread("test_images/car_test_2.jpg")
candidates = pipeline.recognize(image, color_format="BGR")

for cand in candidates:
    print(f"Plate: {cand.normalized_text} | Conf: {cand.ocr_confidence:.4f} | Valid: {cand.is_valid}")
```

### High-Level Telemetry Output
Retrieve standardized JSON telemetry matching the edge backend schema:

```python
result = pipeline.process(image, color_format="BGR", request_id="req-001")
print(result.to_dict(minimal=True))
```

```json
{
  "schema_version": "1.0",
  "request_id": "req-001",
  "status": "OK",
  "candidates": [
    {
      "plate": "30A69770",
      "raw_plate": "30A69770",
      "confidence": 0.9969,
      "detection_confidence": 0.7569,
      "bbox": [178, 234, 242, 266]
    }
  ],
  "model_version": "alpr-v0.1",
  "processing_time_ms": 65.2
}
```

---

## 2. Architecture & The Factory vs. Lab Separation

The repository enforces a strict separation between the production library (**The Factory**) and archived research experiments (**The Lab**):

```
lpr_research/
├── pyproject.toml              # Build definition for eiu_lpr package
├── demo.py                     # Standalone verification demo
├── run_tests.py                # Automated test runner (11/11 tests)
├── benchmark_linux_cpu.py      # Dedicated Linux CPU latency benchmark
├── constraints.txt             # Pinned dependency constraints for Linux CPU
├── LICENSE                     # MIT License
├── MODEL_CARD.md               # Model specifications and limitations
│
├── src/eiu_lpr/                # THE FACTORY: Clean Production Package
│   ├── pipeline.py             # LPRPipeline orchestrator
│   ├── detector.py             # Pure ONNX Runtime YOLO detector (NumPy letterbox & NMS)
│   ├── recognizer.py           # CCT-XS ONNX OCR recognizer
│   ├── preprocessor.py         # Dynamic 4% padding and adaptive enhancement
│   ├── validator.py            # Circular 24/2023 syntax validation
│   ├── models.py               # Typed dataclasses (BoundingBox, PlateCandidate, PipelineResult)
│   └── exceptions.py           # Custom exceptions (InvalidInputError, ModelNotFoundError)
│
├── models/                     # THE ASSETS: Frozen Release Weights & Checksums
│   ├── detector.onnx           # Plate detection weights (10.9 MB)
│   ├── cct_xs.onnx             # Plate OCR weights (3.3 MB)
│   ├── cct_xs_plate_config.yaml# Alphabet vocabulary and input configuration
│   └── checksums.sha256        # SHA-256 cryptographic verification manifest
│
├── dist/                       # DISTRIBUTION: Pre-built wheels
│   └── eiu_lpr-0.1.0-py3-none-any.whl
│
├── docs/                       # DOCUMENTATION
│   ├── CONFIGURATION_GUIDE.md  # Configuration options, directory reference, and APIs
│   ├── INTEGRATION_HANDBOOK.md # Operational FAQs, integration patterns, versioning guide
│   ├── NORMALIZATION_SPEC.md   # Circular 24/2023 normalization rules
│   └── GOLDEN_DATASET.md       # 20-image golden benchmark dataset documentation
│
├── golden_dataset/             # GOLDEN BENCHMARK
│   └── ground_truth.json       # Ground truth annotations and expectations
│
├── tests/                      # Automated unit tests
│   ├── test_in_memory_api.py   # In-memory arrays, zero disk I/O, zero prints
│   ├── test_split_apis.py      # detect() and read_plate() consistency
│   ├── test_onnx_runtime.py    # Zero ultralytics imports
│   ├── test_warmup.py          # Session priming and latency checks
│   └── test_golden_dataset.py  # 20-image regression test
│
├── test_images/                # Standard test images
│
└── lab/                        # THE LAB: Archived Research & Prototypes
    ├── benchmarks/             # Historical evaluation harness (run_benchmarks.py, BENCHMARK_SUMMARY.md)
    ├── scripts/                # Legacy scripts (infer.py, lpr_engine.py, main.py, simulate_rfid.py)
    ├── notebooks/              # Google Colab training notebooks
    ├── models_legacy/          # Raw research weights (.pt models, legacy ONNX)
    ├── research_docs/          # Historical handoffs, research reports, and design notes
    └── debug_slices/           # Visual perspective slices and crops
```

---

## 3. Essential Commands

### Run Test Suite
```bash
python run_tests.py
```

### Run Golden Dataset Regression
```bash
python tests/test_golden_dataset.py
```

### Run Verification Demo
```bash
python demo.py
```

### Run CPU Latency Benchmark
```bash
python benchmark_linux_cpu.py --iterations 100
```

### Verify Checksums
On Linux:
```bash
sha256sum -c models/checksums.sha256
```
On Windows PowerShell:
```powershell
Get-FileHash models/detector.onnx -Algorithm SHA256
```

---

## 4. Documentation Index

- [Configuration Guide](file:///c:/Users/User/EIU_OITP/lpr_research/docs/CONFIGURATION_GUIDE.md): Parameters, thresholds, and full method signatures.
- [Integration Handbook](file:///c:/Users/User/EIU_OITP/lpr_research/docs/INTEGRATION_HANDBOOK.md): Operational FAQs, troubleshooting, and step-by-step versioning runbook.
- [Normalization Specification](file:///c:/Users/User/EIU_OITP/lpr_research/docs/NORMALIZATION_SPEC.md): Vietnamese Ministry of Public Security Circular 24/2023/TT-BCA syntax rules.
- [Model Card](file:///c:/Users/User/EIU_OITP/lpr_research/MODEL_CARD.md): YOLO11n + CCT-XS architectures, resolutions, and operational limitations.
- [Golden Dataset Specification](file:///c:/Users/User/EIU_OITP/lpr_research/docs/GOLDEN_DATASET.md): 20 verified reference images and expected values.
