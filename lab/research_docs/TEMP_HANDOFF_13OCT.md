I am continuing our Vietnamese License Plate Recognition (LPR) & RFID access control project. Here is the context, current state, and what we are doing next:

### 1. Goal
We are systematically benchmarking detector models across YOLO variants (nano, small, medium, large, etc. for YOLOv10, YOLO11, and latest architectures) and OCR engines (PaddleOCR vs EasyOCR). We need to measure detection recall, syntax validity rate, and steady-state inference latency, and basically just compare performance

### 2. Architecture & File Structure
- Project Root: c:\Users\User\EIU_OITP\lpr_research
- Python Environment: .\venv_models\Scripts\python.exe
- Core Engine: `lpr_engine.py` (contains `YOLODetector`, `PaddleOCRAdapter`, `EasyOCRAdapter`, and `LPRPipeline` with the 3-try escalation ladder).
- Syntax & Validation: `lpr_core/validator.py` (strict Vietnamese regex + Levenshtein structural template matching `DDLDDDDD`. No hardcoded character hack dictionaries).
- Image Enhancement: `lpr_core/preprocessor.py` (CLAHE, bilateral filter, 2.5x bicubic upscaling for faint/shadowed plates).
- Production Service: `main.py` (FastAPI backend for RFID card tap events).
- Single-Image Inference: `infer.py` (standalone prediction tool returning bounding boxes, confidence %, plate text, and stage latencies).
- Benchmarking Suite:
  - `benchmarks/run_benchmarks.py`: runs batch evaluation over `test_images/`, reports detection recall %, validation pass rate %, and steady-state latency. Automatically appends runs to `benchmarks/BENCHMARK_SUMMARY.md`.
  - `benchmarks/compare_preprocessing.py`: side-by-side diagnostic script comparing raw crop vs CLAHE-enhanced crop.
  - Outputs are sent to /Outputs

### 3. Model Training (Google Colab)
- Notebook Location: `ColabNotebookTemplate/train_vietnam_lpr_colab_YOLO11.ipynb`
- Training Environment: Google Colab with free T4 GPU.
- Dataset: Roboflow Universe dataset `vietnamese-car-license-plate-vlbzz` (classes: `plate` / `license_plate`). If suggested for more dataset, please suggest the next steps!
- Exported Weights: Saved locally into `Models_yolo/` (e.g., `Models_yolo/best.pt`, `Models_yolo/yolo11s.pt`, etc.).
- I believe for testing, all model exporting should be in .onnx format (since i believe it reduces the latency overhead that we're trying to test for.) - please adjust the existing formatted code to change the format.
 
### 4. Non-Negotiable Project Rules
- NO emojis in any code, console output, or documentation files. Keep comments in '#', one liners, simple and straightforward. No need any excessive analogies. Just tell it as it is an what it does.
- NO hardcoded string replacement dictionaries (e.g. no manual `6->G` or `O->0` remapping). Must use pure computer vision and format similarity.
- Operating system: Windows (PowerShell commands only; never use `cd` as it can cause directory issues).
- Research and analysis focused



### 5. Next Task
Guide me through creating and training multiple YOLO models from ver 10 (s onwards) to 26. Priortise accuracy and speed (we are just detecting license plates; so i dont think  super advanced model is required). If needed; we can standardise a modification across all YOLO or OCR models (i.e, all yolo models are only trained on license plates or cars to be optimised?)

1. I'd like for the model training to be abit quicker, less than 45 minutes; so perhaps standardise Epochs to be 10 - 20?
2. I'd also want to know how to utilise GPU in colab (so it trains faster) without impacting on accuracy, if you suggest us to take a look at adjusting parameters such as learning rate, or etc, that can really boost accuracy without impacting and is global and every model can follow through without any biasness or impacted performance, then that would be nice.
3. Let me know what you think is the best way to keep benchmarks equal, readable, easy to se tup, reliable testing with existing code (if you super-duper think something needs to be changed in existing code, let me know)
4. I'm looking at finding better OCR models other than paddle. If you think of ways to find one that'd be good. Same goes for YOLO if necessary.
5. Because we're just doing license plate detection and extraction (perhaps include some capabilities for car / bike / truck / vehicle detections with license plates); some model capabilities can be 'ignored'; so loading time, model size, and processing time, etc can be cut down to priortise latency - one thing i'm quite concern about is latency or time. I need it to be less than 2 seconds MAX for the entire process from 'scan' to 'gate open'. Lower is better. But of course, do not compromise on accuracy and efficiency or unrealistic standards or be unable to generalise. Be optimised !