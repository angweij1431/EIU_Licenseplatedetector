# License Plate Recognition (LPR) Benchmark Summary

Comprehensive empirical evaluation log across Vietnamese vehicle license plate recognition pipelines.
All recorded metrics (detection confidence, OCR confidence, resolution, try escalation, and millisecond latencies) represent **real measured model outputs** from physical pipeline execution.


updated benchmark tests for 15 oct with new test images incl neg samples and bicycle focused. testing for 'refined' and bench models for testing
1. *best_yolo(xx) - models trained on 15 epochs for general model performance and latency tests
2. *refined_best_yolo... - models trained on 15 epochs AND new sampled training data (with more focus on motorcycles) for general model performance and latency tests and non-biased-ness
3. *best.pt/.onnx - baseline model trained on 50 epochs with general dataset.



---

## Benchmark Log

| Timestamp | Model | Image | Try Level | Resolution | Boxes | Plate | Valid | Det (ms) | OCR (ms) | Total (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
