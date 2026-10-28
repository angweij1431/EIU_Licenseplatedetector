# Research Laboratory and Legacy Prototypes (`lab/`)

This directory houses the historical research codebase, training notebooks, raw experimental models, and prototypes that preceded the production `eiu_lpr` library.

---

## Directory Organization

- **`scripts/`**: Legacy standalone scripts from the research phase (`infer.py`, `lpr_engine.py`, `main.py` FastAPI demo, `simulate_rfid.py`, `requirements.txt`).
- **`notebooks/`**: Google Colab training and fine-tuning notebooks for YOLO detectors, CCT-XS, and legacy LPRNet.
- **`models_legacy/`**: Raw research model weights, including intermediate PyTorch `.pt` checkpoints, legacy LPRNet ONNX models, and previous iteration weights.
- **`lpr_core/`**: Research version of preprocessing and validation modules.
- **`debug_slices/`**: Intermediate crops, perspective deskewing slices, and visual verification artifacts.
- **`research_docs/`**: Historical handoff reports, evaluation summaries, and early design documents.
- **`archive/`**: Early test scripts and initial baseline comparisons.
- **`dataset/`**: Large research fleet scrapes (e.g. `CAR MOTOR` web dumps).

---

## Production Library Reference

For production integration, import the standalone library from `src/eiu_lpr/` or install the pre-built wheel from `dist/`. See root `README.md` and `docs/CONFIGURATION_GUIDE.md` for production APIs.

