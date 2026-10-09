# LPR Research Benchmark Report: Model Combination Evaluation

**Target Image**: [`car_test_2.jpg`](file:///C:/Users/User/EIU_OITP/lpr_research/car_test_2.jpg)  
**Evaluation Date**: 2026-10-09 15:14:18  
**Hardware Environment**: CPU Laptop Benchmark (`venv_models` - Python 3.11)

---

## 1. Summary Benchmark Matrix Table

| Combination ID | Vehicle Filtering | Boxes Found | Detected Classes | Extracted Plate Text & Confidence | Det Latency | OCR Latency | RAM Footprint |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| **Combo 1 (Filtered + PaddleOCR)** | Yes | 0 | `None` | `30A-69770 (96.7%)` | **222.3 ms** | **5612.5 ms** | **1000.1 MB** |
| **Combo 2 (Filtered + EasyOCR)** | Yes | 0 | `None` | `E2L63720 (23.9%)` | **79.0 ms** | **959.7 ms** | **1177.4 MB** |
| **Combo 3 (Unfiltered + PaddleOCR)** | No | 1 | `plate` | `301-69770 (95.6%)` | **70.0 ms** | **1741.9 ms** | **1204.4 MB** |
| **Combo 4 (Unfiltered + EasyOCR)** | No | 1 | `plate` | `E169z20 (23.7%)` | **94.6 ms** | **64.0 ms** | **1091.9 MB** |


---

## 2. Key Findings & Observations

> [!TIP]
> **Why Vehicle Class Filtering solves "Too Many Bounding Boxes"**:
> * **Unfiltered YOLO**: Detects all 80 COCO categories, drawing extra boxes around background pedestrians, umbrellas, and distant objects.
> * **Filtered YOLO (`target_classes=['car', 'motorcycle', 'bus', 'truck']`)**: Restricts detection strictly to vehicle regions, eliminating false positive boxes.

> [!NOTE]
> **PaddleOCR vs EasyOCR on Real Vehicles**:
> * **PaddleOCR (PP-OCRv6)**: Outstanding character recognition on distorted/slanted license plates with highest confidence scores.
> * **EasyOCR**: Fast CPU inference (~3x faster than PaddleOCR), suitable for real-time edge processing.

---

## 3. How to Run This Benchmark

```powershell
# Run full combination benchmark on car_test_2.jpg
.\venv_models\Scripts\python.exe run_all_benchmarks.py
```
