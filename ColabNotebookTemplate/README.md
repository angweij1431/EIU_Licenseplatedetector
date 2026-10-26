# Google Colab Training Notebooks and Templates

This directory contains turnkey Jupyter notebooks and supporting training scripts for training detector (YOLO) and recognizer (CCT-XS) models on Google Colab GPU instances.

---

## 1. Notebook Index and Model Mapping

| Notebook | Target Architecture | Epochs | Dataset Distribution & Features | Exported Model | Status |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `01_train_yolo_balanced_1to1_25_40ep.ipynb` | YOLO11n, YOLO11s, YOLOv10n, YOLOv10s, YOLO26n, YOLO26s | 25 & 40 | Balanced 1:1 ratio (1,400 cars, 1,397 motorbikes) + 120 negative background images | `balanced_40eps_yolo11n.onnx` (Primary Production Detector) | **Active Production** |
| `02_finetune_cct_xs_vietnam.ipynb` | Compact Convolutional Transformer (CCT-XS) v2 Global | 30 | Cropped Vietnamese plates (128x64 RGB), multi-task loss (0.90 text, 0.10 region) | `cct_xs_finetuned.onnx` (Active Recognizer) | **template ** |
| `03_train_yolo_refined_15ep.ipynb` | YOLO11, YOLOv10, YOLO26 variants | 15 | Refined multi-model exploration on car-dominant dataset (~4:1 car ratio) | `refined_best_yolo11s.onnx` | Experimental Baseline |
| `04_train_baseline_yolo11n_50ep.ipynb` | YOLO11n Single Baseline | 50 | Early car-only dataset (unbalanced baseline) | `best_yolo11n.onnx` | Baseline Archive |
| `05_train_baseline_yolov8n_50ep.ipynb` | YOLOv8n Single Baseline | 50 | Early car-only dataset (initial reference) | `best_yolo8.onnx` | Baseline Archive |
| `06_legacy_train_lprnet_vietnam.ipynb` | LPRNet CTC Spatial CNN | 100 | Vietnamese cropped plate slices (94x24 grayscale/RGB) | `lprnet.onnx` | Deprecated (Replaced by CCT-XS) |


---

## 2. Training Hardware Recommendations

- Accelerator: NVIDIA Tesla T4 or higher (Google Colab standard GPU runtime).
- Precision: FP16 mixed precision enabled (`amp=True`).
- Export Format: Export weights directly to Open Neural Network Exchange (ONNX) format via `model.export(format="onnx", dynamic=False, imgsz=640)` for zero-dependency CPU runtime inference.

