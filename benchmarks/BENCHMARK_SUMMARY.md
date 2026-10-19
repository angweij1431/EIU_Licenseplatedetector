# License Plate Recognition (LPR) Benchmark Summary

Comprehensive empirical evaluation log across Vietnamese vehicle license plate recognition pipelines.
All recorded metrics (detection confidence, OCR confidence, resolution, try escalation, and millisecond latencies) represent **real measured model outputs** from physical pipeline execution.


updated benchmark tests for 15 oct with new test images incl neg samples and bicycle focused. testing for 'refined' and bench models for testing
1. *best_yolo(xx) - models trained on 15 epochs for general model performance and latency tests
2. *refined_best_yolo... - models trained on 15 epochs AND new sampled training data (with more focus on motorcycles) for general model performance and latency tests and non-biased-ness
3. *balanced_best_yolo... - models trained on 25 epochs AND balanced dataset (1.4k cars x 1.4k bikes) for balanced look in data distribution
3. *best.pt/.onnx - baseline model trained on 50 epochs with general dataset.



---

## Benchmark Log

| Timestamp | Model | Image | Try Level | Resolution | Boxes | Plate | Valid | Det (ms) | OCR (ms) | Total (ms) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

## 9. Real Empirical Output & Confidence Matrix

*Evaluated at: `2026-10-16 09:04:40` across `16` test images (Measured outputs and model confidence scores only)*

### Model Summary

| ONNX Model | Plate Recall | Avg Det Conf % | Gate Open Rate | Avg OCR Conf % | Try 1 | Try 2 | Gate Closed | Cold Det (ms) | Steady Det (ms) | Steady Total (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `balanced_best_yolo11n.onnx` | 11/16 (68.8%) | 63.8% | **7/16 (43.8%)** | 83.5% | 5 | 2 | 9 | 3227.7 ms | 104.2 ms | **2775.8 ms** |
| `balanced_best_yolo11s.onnx` | 11/16 (68.8%) | 68.7% | **7/16 (43.8%)** | 83.1% | 4 | 3 | 9 | 359.2 ms | 232.4 ms | **2988.5 ms** |
| `balanced_best_yolo26n.onnx` | 8/16 (50.0%) | 75.2% | **6/16 (37.5%)** | 88.0% | 5 | 1 | 10 | 149.4 ms | 114.8 ms | **1539.1 ms** |
| `balanced_best_yolov10n.onnx` | 10/16 (62.5%) | 71.4% | **7/16 (43.8%)** | 95.4% | 6 | 1 | 9 | 214.1 ms | 114.9 ms | **2186.1 ms** |
| `balanced_best_yolov10s.onnx` | 12/16 (75.0%) | 66.0% | **6/16 (37.5%)** | 85.7% | 4 | 2 | 10 | 311.5 ms | 231.9 ms | **3528.9 ms** |
| `best.onnx` | 10/16 (62.5%) | 69.6% | **7/16 (43.8%)** | 86.0% | 5 | 2 | 9 | 209.8 ms | 132.7 ms | **2325.7 ms** |
| `best_yolo10n.onnx` | 10/16 (62.5%) | 71.9% | **7/16 (43.8%)** | 83.2% | 4 | 3 | 9 | 214.7 ms | 117.4 ms | **2172.7 ms** |
| `best_yolo10s.onnx` | 12/16 (75.0%) | 67.5% | **7/16 (43.8%)** | 90.5% | 4 | 3 | 9 | 323.9 ms | 265.8 ms | **3560.8 ms** |
| `best_yolo11n.onnx` | 11/16 (68.8%) | 69.6% | **7/16 (43.8%)** | 78.4% | 4 | 3 | 9 | 357.5 ms | 136.0 ms | **2940.8 ms** |
| `best_yolo11s.onnx` | 11/16 (68.8%) | 67.9% | **7/16 (43.8%)** | 86.1% | 5 | 2 | 9 | 460.0 ms | 266.9 ms | **2975.7 ms** |
| `best_yolo26n.onnx` | 9/16 (56.2%) | 78.0% | **6/16 (37.5%)** | 89.0% | 3 | 3 | 10 | 356.4 ms | 105.3 ms | **2248.4 ms** |
| `best_yolo26s.onnx` | 11/16 (68.8%) | 73.4% | **7/16 (43.8%)** | 81.6% | 4 | 3 | 9 | 483.6 ms | 249.4 ms | **2898.0 ms** |
| `best_yolo8.onnx` | 9/16 (56.2%) | 80.4% | **7/16 (43.8%)** | 90.7% | 4 | 3 | 9 | 204.9 ms | 170.3 ms | **2020.6 ms** |
| `refined_best_yolo11n.onnx` | 9/16 (56.2%) | 78.1% | **7/16 (43.8%)** | 89.1% | 4 | 3 | 9 | 249.3 ms | 134.2 ms | **1981.0 ms** |
| `refined_best_yolo11s.onnx` | 10/16 (62.5%) | 76.1% | **6/16 (37.5%)** | 84.9% | 3 | 3 | 10 | 381.9 ms | 281.1 ms | **3016.3 ms** |
| `refined_best_yolo26n.onnx` | 13/16 (81.2%) | 64.1% | **7/16 (43.8%)** | 83.9% | 4 | 3 | 9 | 183.1 ms | 100.0 ms | **3056.3 ms** |
| `refined_best_yolo26s.onnx` | 10/16 (62.5%) | 69.6% | **7/16 (43.8%)** | 83.2% | 4 | 3 | 9 | 349.0 ms | 210.0 ms | **2262.9 ms** |
| `refined_best_yolov10n.onnx` | 9/16 (56.2%) | 77.6% | **7/16 (43.8%)** | 89.0% | 3 | 4 | 9 | 190.1 ms | 118.4 ms | **2091.1 ms** |
| `refined_best_yolov10s.onnx` | 12/16 (75.0%) | 60.5% | **7/16 (43.8%)** | 85.4% | 5 | 2 | 9 | 295.8 ms | 257.2 ms | **3022.0 ms** |
| `yolov8n.onnx` | 0/16 (0.0%) | 0.0% | **0/16 (0.0%)** | 0.0% | 0 | 0 | 16 | 255.0 ms | 169.9 ms | **201.4 ms** |

### Per-Image Real Outputs

| Model | Image | Plate Detected (Conf %) | Valid Format (OCR Conf %) | Gate Status | Extracted Plate | Det (ms) | OCR (ms) | Total (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `balanced_best_yolo11n.onnx` | `bike_test_10.jpg` | YES (80.1%) | YES (93.0%) | **Try 1** | `43D199999` | 3227.7 | 2420.3 | 5676.0 |
| `balanced_best_yolo11n.onnx` | `bike_test_11.jpg` | YES (85.1%) | YES (99.3%) | **Try 1** | `29H133979` | 73.4 | 1956.0 | 2041.9 |
| `balanced_best_yolo11n.onnx` | `bike_test_12.jpg` | YES (77.1%) | YES (95.7%) | **Try 2** | `61H76854` | 50.0 | 2841.2 | 2930.5 |
| `balanced_best_yolo11n.onnx` | `bike_test_9.jpg` | YES (85.4%) | YES (96.9%) | **Try 1** | `59D242069` | 61.0 | 2235.7 | 2335.5 |
| `balanced_best_yolo11n.onnx` | `car_test_2.jpg` | YES (74.3%) | NO (94.3%) | **Gate Closed** | `30169770` | 148.8 | 6114.4 | 6277.4 |
| `balanced_best_yolo11n.onnx` | `car_test_3.jpg` | YES (32.8%) | NO (56.5%) | **Gate Closed** | `2TRUNGT` | 125.0 | 7996.7 | 8180.1 |
| `balanced_best_yolo11n.onnx` | `car_test_4.jpg` | YES (75.2%) | YES (97.9%) | **Try 1** | `51G74372` | 62.2 | 1204.9 | 1280.6 |
| `balanced_best_yolo11n.onnx` | `car_test_5.jpg` | YES (74.0%) | YES (86.1%) | **Try 1** | `51G77777` | 58.0 | 1322.6 | 1408.4 |
| `balanced_best_yolo11n.onnx` | `car_test_6.jpg` | YES (40.3%) | YES (99.6%) | **Try 2** | `63H90394` | 75.4 | 4013.4 | 4139.7 |
| `balanced_best_yolo11n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 126.8 | 0.0 | 169.8 |
| `balanced_best_yolo11n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 121.5 | 0.0 | 145.5 |
| `balanced_best_yolo11n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 123.7 | 0.0 | 141.3 |
| `balanced_best_yolo11n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 135.0 | 0.0 | 157.5 |
| `balanced_best_yolo11n.onnx` | `neg_sample_test_3.jpg` | YES (49.5%) | NO (15.3%) | **Gate Closed** | `B` | 121.7 | 5180.5 | 5322.7 |
| `balanced_best_yolo11n.onnx` | `neg_sample_test_4.jpg` | YES (27.8%) | NO (0.0%) | **Gate Closed** | `` | 148.4 | 6777.9 | 6949.7 |
| `balanced_best_yolo11n.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 132.4 | 0.0 | 156.6 |
| `balanced_best_yolo11s.onnx` | `bike_test_10.jpg` | YES (70.8%) | YES (96.8%) | **Try 1** | `43D199999` | 359.2 | 2671.3 | 3048.1 |
| `balanced_best_yolo11s.onnx` | `bike_test_11.jpg` | YES (85.6%) | YES (99.6%) | **Try 2** | `29H133979` | 166.8 | 4862.6 | 5040.9 |
| `balanced_best_yolo11s.onnx` | `bike_test_12.jpg` | YES (84.4%) | YES (95.3%) | **Try 2** | `61H76954` | 160.3 | 3693.4 | 3865.0 |
| `balanced_best_yolo11s.onnx` | `bike_test_9.jpg` | YES (87.0%) | YES (98.9%) | **Try 1** | `59D242069` | 178.3 | 2767.1 | 2961.5 |
| `balanced_best_yolo11s.onnx` | `car_test_2.jpg` | YES (75.0%) | NO (92.8%) | **Gate Closed** | `30169730` | 324.8 | 7780.8 | 8116.6 |
| `balanced_best_yolo11s.onnx` | `car_test_3.jpg` | YES (42.3%) | NO (0.0%) | **Gate Closed** | `NONE` | 295.5 | 6088.2 | 6423.1 |
| `balanced_best_yolo11s.onnx` | `car_test_4.jpg` | YES (74.1%) | YES (97.9%) | **Try 1** | `51G74372` | 132.0 | 1196.2 | 1338.3 |
| `balanced_best_yolo11s.onnx` | `car_test_5.jpg` | YES (74.3%) | YES (96.6%) | **Try 1** | `51G77777` | 139.2 | 1585.7 | 1742.9 |
| `balanced_best_yolo11s.onnx` | `car_test_6.jpg` | YES (72.8%) | YES (99.7%) | **Try 2** | `63H90394` | 137.0 | 3380.3 | 3554.3 |
| `balanced_best_yolo11s.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 262.0 | 0.0 | 284.1 |
| `balanced_best_yolo11s.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 297.8 | 0.0 | 328.6 |
| `balanced_best_yolo11s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 278.8 | 0.0 | 299.8 |
| `balanced_best_yolo11s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 267.6 | 0.0 | 290.6 |
| `balanced_best_yolo11s.onnx` | `neg_sample_test_3.jpg` | YES (55.0%) | NO (24.5%) | **Gate Closed** | `` | 292.6 | 4537.6 | 4853.5 |
| `balanced_best_yolo11s.onnx` | `neg_sample_test_4.jpg` | YES (34.0%) | NO (28.5%) | **Gate Closed** | `F` | 280.3 | 5127.2 | 5429.7 |
| `balanced_best_yolo11s.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 272.9 | 0.0 | 298.9 |
| `balanced_best_yolo26n.onnx` | `bike_test_10.jpg` | YES (86.0%) | YES (97.7%) | **Try 1** | `43D199999` | 149.4 | 2714.2 | 2885.6 |
| `balanced_best_yolo26n.onnx` | `bike_test_11.jpg` | YES (49.0%) | YES (99.8%) | **Try 1** | `29H133979` | 150.5 | 2256.8 | 2421.3 |
| `balanced_best_yolo26n.onnx` | `bike_test_12.jpg` | YES (85.6%) | YES (95.7%) | **Try 2** | `61H76854` | 50.2 | 2807.1 | 2870.2 |
| `balanced_best_yolo26n.onnx` | `bike_test_9.jpg` | YES (87.4%) | YES (98.0%) | **Try 1** | `59D242069` | 59.5 | 2423.2 | 2496.7 |
| `balanced_best_yolo26n.onnx` | `car_test_2.jpg` | YES (76.6%) | NO (92.1%) | **Gate Closed** | `30169770` | 110.9 | 5981.3 | 6104.5 |
| `balanced_best_yolo26n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 115.0 | 0.0 | 155.0 |
| `balanced_best_yolo26n.onnx` | `car_test_4.jpg` | YES (81.0%) | YES (95.5%) | **Try 1** | `51G74372` | 54.0 | 1306.8 | 1377.8 |
| `balanced_best_yolo26n.onnx` | `car_test_5.jpg` | YES (73.0%) | YES (95.5%) | **Try 1** | `51G77777` | 61.1 | 1542.4 | 1626.8 |
| `balanced_best_yolo26n.onnx` | `car_test_6.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 168.5 | 0.0 | 213.7 |
| `balanced_best_yolo26n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 158.8 | 0.0 | 198.4 |
| `balanced_best_yolo26n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 137.3 | 0.0 | 162.3 |
| `balanced_best_yolo26n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 133.8 | 0.0 | 150.6 |
| `balanced_best_yolo26n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 106.0 | 0.0 | 128.1 |
| `balanced_best_yolo26n.onnx` | `neg_sample_test_3.jpg` | YES (62.9%) | NO (29.4%) | **Gate Closed** | `` | 140.9 | 4680.0 | 4846.2 |
| `balanced_best_yolo26n.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 150.7 | 0.0 | 183.9 |
| `balanced_best_yolo26n.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 124.9 | 0.0 | 150.9 |
| `balanced_best_yolov10n.onnx` | `bike_test_10.jpg` | YES (86.7%) | YES (95.0%) | **Try 1** | `43D199999` | 214.1 | 2541.2 | 2772.2 |
| `balanced_best_yolov10n.onnx` | `bike_test_11.jpg` | YES (87.1%) | YES (99.8%) | **Try 1** | `29H133979` | 79.0 | 2293.5 | 2385.5 |
| `balanced_best_yolov10n.onnx` | `bike_test_12.jpg` | YES (86.4%) | YES (93.6%) | **Try 2** | `61H76854` | 74.2 | 3776.7 | 3863.0 |
| `balanced_best_yolov10n.onnx` | `bike_test_9.jpg` | YES (87.8%) | YES (98.1%) | **Try 1** | `59D242069` | 65.7 | 2643.0 | 2744.9 |
| `balanced_best_yolov10n.onnx` | `car_test_2.jpg` | YES (73.6%) | NO (91.0%) | **Gate Closed** | `30169770` | 183.0 | 6571.4 | 6766.2 |
| `balanced_best_yolov10n.onnx` | `car_test_3.jpg` | YES (26.8%) | NO (99.8%) | **Gate Closed** | `17010` | 216.3 | 6087.3 | 6341.6 |
| `balanced_best_yolov10n.onnx` | `car_test_4.jpg` | YES (71.6%) | YES (97.7%) | **Try 1** | `51G74372` | 59.2 | 1302.5 | 1375.9 |
| `balanced_best_yolov10n.onnx` | `car_test_5.jpg` | YES (71.8%) | YES (94.4%) | **Try 1** | `51G77777` | 79.1 | 1698.1 | 1803.2 |
| `balanced_best_yolov10n.onnx` | `car_test_6.jpg` | YES (77.2%) | YES (99.9%) | **Try 1** | `63H90394` | 85.7 | 2246.6 | 2370.3 |
| `balanced_best_yolov10n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 128.0 | 0.0 | 146.5 |
| `balanced_best_yolov10n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 108.2 | 0.0 | 132.0 |
| `balanced_best_yolov10n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 102.0 | 0.0 | 118.0 |
| `balanced_best_yolov10n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 149.6 | 0.0 | 177.6 |
| `balanced_best_yolov10n.onnx` | `neg_sample_test_3.jpg` | YES (45.2%) | NO (85.0%) | **Gate Closed** | `J` | 163.8 | 4096.1 | 4292.2 |
| `balanced_best_yolov10n.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 121.1 | 0.0 | 145.1 |
| `balanced_best_yolov10n.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 108.5 | 0.0 | 129.7 |
| `balanced_best_yolov10s.onnx` | `bike_test_10.jpg` | YES (86.5%) | YES (98.8%) | **Try 1** | `43D199999` | 311.5 | 2218.0 | 2546.5 |
| `balanced_best_yolov10s.onnx` | `bike_test_11.jpg` | YES (87.3%) | YES (99.7%) | **Try 1** | `29H133979` | 129.8 | 1998.6 | 2153.6 |
| `balanced_best_yolov10s.onnx` | `bike_test_12.jpg` | YES (84.0%) | YES (94.3%) | **Try 2** | `61H76954` | 152.5 | 2904.2 | 3069.7 |
| `balanced_best_yolov10s.onnx` | `bike_test_9.jpg` | YES (88.0%) | YES (97.6%) | **Try 1** | `59D242069` | 144.5 | 2252.2 | 2414.6 |
| `balanced_best_yolov10s.onnx` | `car_test_2.jpg` | YES (73.3%) | NO (92.1%) | **Gate Closed** | `30169770` | 309.9 | 6190.0 | 6514.4 |
| `balanced_best_yolov10s.onnx` | `car_test_3.jpg` | YES (33.2%) | NO (94.4%) | **Gate Closed** | `TRUNG1` | 322.0 | 6982.8 | 7352.5 |
| `balanced_best_yolov10s.onnx` | `car_test_4.jpg` | YES (75.2%) | NO (95.3%) | **Gate Closed** | `51674372` | 281.1 | 6143.3 | 6437.4 |
| `balanced_best_yolov10s.onnx` | `car_test_5.jpg` | YES (75.8%) | YES (93.8%) | **Try 1** | `51G77777` | 115.0 | 1301.9 | 1434.0 |
| `balanced_best_yolov10s.onnx` | `car_test_6.jpg` | YES (61.6%) | YES (99.7%) | **Try 2** | `63H90394` | 115.7 | 3329.7 | 3485.4 |
| `balanced_best_yolov10s.onnx` | `car_test_7.jpg` | YES (32.7%) | NO (92.3%) | **Gate Closed** | `KM216HNOI` | 302.1 | 8347.5 | 8668.6 |
| `balanced_best_yolov10s.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 307.8 | 0.0 | 331.7 |
| `balanced_best_yolov10s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 253.0 | 0.0 | 270.0 |
| `balanced_best_yolov10s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 265.0 | 0.0 | 289.5 |
| `balanced_best_yolov10s.onnx` | `neg_sample_test_3.jpg` | YES (64.5%) | NO (26.9%) | **Gate Closed** | `` | 245.0 | 4286.3 | 4549.5 |
| `balanced_best_yolov10s.onnx` | `neg_sample_test_4.jpg` | YES (29.8%) | NO (43.7%) | **Gate Closed** | `7` | 276.5 | 5373.5 | 5671.1 |
| `balanced_best_yolov10s.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 259.0 | 0.0 | 292.4 |
| `best.onnx` | `bike_test_10.jpg` | YES (80.6%) | YES (98.3%) | **Try 1** | `43D199999` | 209.8 | 2683.1 | 2911.3 |
| `best.onnx` | `bike_test_11.jpg` | YES (72.4%) | YES (99.3%) | **Try 1** | `29H133979` | 118.8 | 2807.8 | 2950.4 |
| `best.onnx` | `bike_test_12.jpg` | YES (83.6%) | YES (96.4%) | **Try 2** | `61H76854` | 57.6 | 2799.1 | 2872.9 |
| `best.onnx` | `bike_test_9.jpg` | YES (87.3%) | YES (98.1%) | **Try 1** | `59D242069` | 137.3 | 2274.2 | 2438.9 |
| `best.onnx` | `car_test_2.jpg` | YES (76.0%) | NO (89.8%) | **Gate Closed** | `30169730` | 157.8 | 6727.0 | 6899.8 |
| `best.onnx` | `car_test_3.jpg` | YES (42.2%) | YES (96.5%) | **Try 2** | `29M6840` | 61.2 | 3342.1 | 3459.1 |
| `best.onnx` | `car_test_4.jpg` | YES (77.7%) | YES (96.8%) | **Try 1** | `51G74372` | 71.9 | 1474.8 | 1575.2 |
| `best.onnx` | `car_test_5.jpg` | YES (32.6%) | NO (64.3%) | **Gate Closed** | `0` | 187.1 | 5383.1 | 5625.3 |
| `best.onnx` | `car_test_6.jpg` | YES (78.5%) | YES (99.9%) | **Try 1** | `63H90394` | 115.3 | 2815.2 | 2976.1 |
| `best.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 131.1 | 0.0 | 174.3 |
| `best.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 125.4 | 0.0 | 151.2 |
| `best.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 135.9 | 0.0 | 153.9 |
| `best.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 151.2 | 0.0 | 177.2 |
| `best.onnx` | `neg_sample_test_3.jpg` | YES (65.7%) | NO (20.6%) | **Gate Closed** | `` | 151.8 | 4764.2 | 4942.9 |
| `best.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 232.8 | 0.0 | 291.6 |
| `best.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 155.5 | 0.0 | 196.6 |
| `best_yolo10n.onnx` | `bike_test_10.jpg` | YES (70.1%) | YES (98.2%) | **Try 1** | `43D199999` | 214.7 | 2673.2 | 2909.4 |
| `best_yolo10n.onnx` | `bike_test_11.jpg` | YES (79.1%) | YES (99.8%) | **Try 2** | `29H133979` | 114.5 | 3609.4 | 3739.4 |
| `best_yolo10n.onnx` | `bike_test_12.jpg` | YES (84.9%) | YES (98.5%) | **Try 2** | `61H76854` | 108.2 | 3076.0 | 3197.7 |
| `best_yolo10n.onnx` | `bike_test_9.jpg` | YES (84.9%) | YES (96.6%) | **Try 1** | `59D242069` | 75.0 | 2432.9 | 2526.2 |
| `best_yolo10n.onnx` | `car_test_2.jpg` | YES (75.9%) | NO (94.1%) | **Gate Closed** | `30169770` | 128.9 | 6743.5 | 6888.4 |
| `best_yolo10n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 178.6 | 0.0 | 221.5 |
| `best_yolo10n.onnx` | `car_test_4.jpg` | YES (78.5%) | YES (94.9%) | **Try 1** | `51G74372` | 64.3 | 1087.0 | 1165.6 |
| `best_yolo10n.onnx` | `car_test_5.jpg` | YES (78.0%) | YES (92.8%) | **Try 1** | `51G77777` | 75.0 | 1242.8 | 1343.8 |
| `best_yolo10n.onnx` | `car_test_6.jpg` | YES (79.9%) | YES (99.6%) | **Try 2** | `63H90394` | 108.1 | 3482.7 | 3629.7 |
| `best_yolo10n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 128.1 | 0.0 | 147.1 |
| `best_yolo10n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 205.5 | 0.0 | 228.7 |
| `best_yolo10n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 107.0 | 0.0 | 121.0 |
| `best_yolo10n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 118.5 | 0.0 | 145.4 |
| `best_yolo10n.onnx` | `neg_sample_test_3.jpg` | YES (59.9%) | NO (18.7%) | **Gate Closed** | `B` | 117.2 | 4289.8 | 4426.0 |
| `best_yolo10n.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 119.0 | 0.0 | 144.0 |
| `best_yolo10n.onnx` | `neg_sample_test_5.jpg` | YES (28.1%) | NO (38.9%) | **Gate Closed** | `2` | 113.0 | 4530.9 | 4665.9 |
| `best_yolo10s.onnx` | `bike_test_10.jpg` | YES (84.5%) | YES (96.6%) | **Try 1** | `43D199999` | 323.9 | 2004.4 | 2345.3 |
| `best_yolo10s.onnx` | `bike_test_11.jpg` | YES (84.2%) | YES (99.8%) | **Try 1** | `29H133979` | 157.0 | 2043.7 | 2211.7 |
| `best_yolo10s.onnx` | `bike_test_12.jpg` | YES (83.8%) | YES (96.6%) | **Try 2** | `61H76854` | 159.6 | 2966.6 | 3140.1 |
| `best_yolo10s.onnx` | `bike_test_9.jpg` | YES (85.4%) | YES (98.3%) | **Try 1** | `59D242069` | 225.4 | 3081.1 | 3327.9 |
| `best_yolo10s.onnx` | `car_test_2.jpg` | YES (76.1%) | NO (94.7%) | **Gate Closed** | `30169730` | 287.6 | 7356.5 | 7661.1 |
| `best_yolo10s.onnx` | `car_test_3.jpg` | YES (44.9%) | YES (86.5%) | **Try 2** | `29H6840` | 168.4 | 3061.5 | 3270.7 |
| `best_yolo10s.onnx` | `car_test_4.jpg` | YES (74.9%) | NO (97.6%) | **Gate Closed** | `51674372` | 256.0 | 6695.0 | 6967.0 |
| `best_yolo10s.onnx` | `car_test_5.jpg` | YES (77.1%) | YES (98.0%) | **Try 1** | `51G77777` | 193.7 | 1570.6 | 1787.3 |
| `best_yolo10s.onnx` | `car_test_6.jpg` | YES (75.3%) | YES (99.8%) | **Try 2** | `63H90394` | 126.0 | 3650.0 | 3814.0 |
| `best_yolo10s.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 295.1 | 0.0 | 340.2 |
| `best_yolo10s.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 257.3 | 0.0 | 280.3 |
| `best_yolo10s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 253.5 | 0.0 | 269.5 |
| `best_yolo10s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 239.1 | 0.0 | 260.5 |
| `best_yolo10s.onnx` | `neg_sample_test_3.jpg` | YES (62.0%) | NO (30.0%) | **Gate Closed** | `` | 347.9 | 4964.5 | 5333.5 |
| `best_yolo10s.onnx` | `neg_sample_test_4.jpg` | YES (31.7%) | NO (0.0%) | **Gate Closed** | `NONE` | 489.4 | 4587.0 | 5114.7 |
| `best_yolo10s.onnx` | `neg_sample_test_5.jpg` | YES (30.3%) | NO (97.5%) | **Gate Closed** | `VISIT` | 531.3 | 9076.7 | 9633.9 |
| `best_yolo11n.onnx` | `bike_test_10.jpg` | YES (82.6%) | YES (97.8%) | **Try 1** | `43D199999` | 357.5 | 3377.7 | 3765.8 |
| `best_yolo11n.onnx` | `bike_test_11.jpg` | YES (80.3%) | YES (99.9%) | **Try 2** | `29H133979` | 147.1 | 6066.3 | 6229.5 |
| `best_yolo11n.onnx` | `bike_test_12.jpg` | YES (83.7%) | YES (96.2%) | **Try 2** | `61H76854` | 106.9 | 4405.5 | 4558.1 |
| `best_yolo11n.onnx` | `bike_test_9.jpg` | YES (85.6%) | YES (96.2%) | **Try 1** | `59D242069` | 211.8 | 2930.7 | 3160.1 |
| `best_yolo11n.onnx` | `car_test_2.jpg` | YES (76.5%) | NO (93.5%) | **Gate Closed** | `30169770` | 125.8 | 6354.0 | 6495.7 |
| `best_yolo11n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 179.9 | 0.0 | 238.8 |
| `best_yolo11n.onnx` | `car_test_4.jpg` | YES (75.1%) | YES (97.9%) | **Try 1** | `51G74372` | 81.4 | 1360.9 | 1456.4 |
| `best_yolo11n.onnx` | `car_test_5.jpg` | YES (68.5%) | YES (92.7%) | **Try 1** | `51G77777` | 68.3 | 1233.1 | 1321.2 |
| `best_yolo11n.onnx` | `car_test_6.jpg` | YES (79.7%) | YES (99.7%) | **Try 2** | `63H90394` | 70.2 | 3652.9 | 3767.0 |
| `best_yolo11n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 154.2 | 0.0 | 177.7 |
| `best_yolo11n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 122.2 | 0.0 | 142.3 |
| `best_yolo11n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 117.5 | 0.0 | 134.1 |
| `best_yolo11n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 105.9 | 1.0 | 128.5 |
| `best_yolo11n.onnx` | `neg_sample_test_3.jpg` | YES (62.8%) | NO (29.7%) | **Gate Closed** | `` | 139.9 | 4681.6 | 4843.1 |
| `best_yolo11n.onnx` | `neg_sample_test_4.jpg` | YES (42.0%) | NO (36.3%) | **Gate Closed** | `7` | 233.1 | 5547.7 | 5806.2 |
| `best_yolo11n.onnx` | `neg_sample_test_5.jpg` | YES (28.6%) | NO (22.9%) | **Gate Closed** | `N` | 176.1 | 5456.6 | 5652.7 |
| `best_yolo11s.onnx` | `bike_test_10.jpg` | YES (78.7%) | YES (96.4%) | **Try 1** | `43D199999` | 460.0 | 3203.7 | 3690.4 |
| `best_yolo11s.onnx` | `bike_test_11.jpg` | YES (82.1%) | YES (99.7%) | **Try 1** | `29H133979` | 211.6 | 2202.7 | 2430.8 |
| `best_yolo11s.onnx` | `bike_test_12.jpg` | YES (82.5%) | YES (96.6%) | **Try 2** | `61H76954` | 143.4 | 4030.1 | 4192.3 |
| `best_yolo11s.onnx` | `bike_test_9.jpg` | YES (83.6%) | YES (97.4%) | **Try 1** | `59D242069` | 256.5 | 4142.1 | 4416.6 |
| `best_yolo11s.onnx` | `car_test_2.jpg` | YES (74.0%) | NO (92.7%) | **Gate Closed** | `30169730` | 363.7 | 7928.6 | 8322.2 |
| `best_yolo11s.onnx` | `car_test_3.jpg` | YES (36.0%) | NO (100.0%) | **Gate Closed** | `TRUNGT` | 330.0 | 6251.0 | 6616.6 |
| `best_yolo11s.onnx` | `car_test_4.jpg` | YES (75.6%) | YES (96.1%) | **Try 1** | `51G74372` | 178.0 | 1101.8 | 1291.2 |
| `best_yolo11s.onnx` | `car_test_5.jpg` | YES (76.3%) | YES (98.0%) | **Try 1** | `51G77777` | 150.8 | 1548.4 | 1724.0 |
| `best_yolo11s.onnx` | `car_test_6.jpg` | YES (60.0%) | YES (99.6%) | **Try 2** | `63H90394` | 216.3 | 3107.4 | 3368.8 |
| `best_yolo11s.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 279.2 | 0.0 | 300.2 |
| `best_yolo11s.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 332.2 | 0.0 | 368.3 |
| `best_yolo11s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 267.4 | 0.0 | 283.0 |
| `best_yolo11s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 276.3 | 0.0 | 302.1 |
| `best_yolo11s.onnx` | `neg_sample_test_3.jpg` | YES (64.0%) | NO (27.0%) | **Gate Closed** | `` | 293.9 | 4666.4 | 4976.3 |
| `best_yolo11s.onnx` | `neg_sample_test_4.jpg` | YES (34.0%) | NO (43.7%) | **Gate Closed** | `7` | 351.7 | 5284.6 | 5656.9 |
| `best_yolo11s.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 352.1 | 0.0 | 385.6 |
| `best_yolo26n.onnx` | `bike_test_10.jpg` | YES (84.2%) | YES (97.5%) | **Try 1** | `43D199999` | 356.4 | 2548.6 | 2928.3 |
| `best_yolo26n.onnx` | `bike_test_11.jpg` | YES (76.4%) | YES (99.5%) | **Try 2** | `29H133979` | 63.1 | 3821.8 | 3906.8 |
| `best_yolo26n.onnx` | `bike_test_12.jpg` | YES (82.8%) | YES (95.3%) | **Try 2** | `61H76954` | 98.4 | 2945.6 | 3064.6 |
| `best_yolo26n.onnx` | `bike_test_9.jpg` | YES (86.7%) | YES (98.5%) | **Try 1** | `59D242069` | 71.8 | 3016.3 | 3114.3 |
| `best_yolo26n.onnx` | `car_test_2.jpg` | YES (76.6%) | NO (95.8%) | **Gate Closed** | `30169770` | 105.6 | 6862.0 | 6984.7 |
| `best_yolo26n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 130.5 | 0.0 | 173.9 |
| `best_yolo26n.onnx` | `car_test_4.jpg` | YES (80.4%) | NO (97.3%) | **Gate Closed** | `51674372` | 103.1 | 6349.8 | 6466.9 |
| `best_yolo26n.onnx` | `car_test_5.jpg` | YES (80.0%) | YES (92.6%) | **Try 1** | `51G77777` | 56.5 | 1293.1 | 1372.2 |
| `best_yolo26n.onnx` | `car_test_6.jpg` | YES (67.1%) | YES (99.4%) | **Try 2** | `63H90394` | 67.4 | 3095.4 | 3205.9 |
| `best_yolo26n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 115.1 | 0.0 | 194.8 |
| `best_yolo26n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 105.1 | 0.0 | 126.1 |
| `best_yolo26n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 113.1 | 0.0 | 140.9 |
| `best_yolo26n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 161.5 | 0.0 | 222.2 |
| `best_yolo26n.onnx` | `neg_sample_test_3.jpg` | YES (67.7%) | NO (25.4%) | **Gate Closed** | `` | 127.3 | 4267.5 | 4437.3 |
| `best_yolo26n.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 131.0 | 0.0 | 160.4 |
| `best_yolo26n.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 130.0 | 0.0 | 154.9 |
| `best_yolo26s.onnx` | `bike_test_10.jpg` | YES (83.3%) | YES (96.0%) | **Try 1** | `43D199999` | 483.6 | 2107.8 | 2610.4 |
| `best_yolo26s.onnx` | `bike_test_11.jpg` | YES (83.2%) | YES (99.7%) | **Try 2** | `29H133979` | 159.2 | 4059.6 | 4233.6 |
| `best_yolo26s.onnx` | `bike_test_12.jpg` | YES (83.6%) | YES (95.9%) | **Try 2** | `61H76954` | 162.9 | 3013.9 | 3191.8 |
| `best_yolo26s.onnx` | `bike_test_9.jpg` | YES (85.7%) | YES (97.5%) | **Try 1** | `59D242069` | 148.8 | 2319.3 | 2482.6 |
| `best_yolo26s.onnx` | `car_test_2.jpg` | YES (76.6%) | NO (91.4%) | **Gate Closed** | `30169730` | 293.1 | 6343.2 | 6651.3 |
| `best_yolo26s.onnx` | `car_test_3.jpg` | YES (51.6%) | NO (68.0%) | **Gate Closed** | `A` | 322.5 | 6863.6 | 7224.1 |
| `best_yolo26s.onnx` | `car_test_4.jpg` | YES (75.6%) | YES (97.4%) | **Try 1** | `51G74372` | 139.5 | 1222.0 | 1374.2 |
| `best_yolo26s.onnx` | `car_test_5.jpg` | YES (79.1%) | YES (97.0%) | **Try 1** | `51G77777` | 144.8 | 1441.1 | 1610.1 |
| `best_yolo26s.onnx` | `car_test_6.jpg` | YES (78.1%) | YES (99.5%) | **Try 2** | `63H90394` | 171.0 | 3834.5 | 4045.8 |
| `best_yolo26s.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 314.1 | 0.0 | 363.9 |
| `best_yolo26s.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 280.5 | 0.0 | 307.7 |
| `best_yolo26s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 261.3 | 0.0 | 280.3 |
| `best_yolo26s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 273.7 | 0.0 | 298.0 |
| `best_yolo26s.onnx` | `neg_sample_test_3.jpg` | YES (65.7%) | NO (15.3%) | **Gate Closed** | `B` | 285.3 | 4707.9 | 5013.7 |
| `best_yolo26s.onnx` | `neg_sample_test_4.jpg` | YES (44.3%) | NO (39.6%) | **Gate Closed** | `` | 403.0 | 5551.6 | 5981.8 |
| `best_yolo26s.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 381.1 | 0.0 | 411.6 |
| `best_yolo8.onnx` | `bike_test_10.jpg` | YES (88.1%) | YES (92.9%) | **Try 1** | `43D199999` | 204.9 | 2214.7 | 2437.3 |
| `best_yolo8.onnx` | `bike_test_11.jpg` | YES (82.8%) | YES (99.6%) | **Try 2** | `29H133979` | 74.7 | 3611.2 | 3701.1 |
| `best_yolo8.onnx` | `bike_test_12.jpg` | YES (83.6%) | YES (96.1%) | **Try 2** | `61H76954` | 74.2 | 2760.7 | 2843.9 |
| `best_yolo8.onnx` | `bike_test_9.jpg` | YES (87.0%) | YES (98.4%) | **Try 1** | `59D242069` | 70.1 | 2344.3 | 2428.0 |
| `best_yolo8.onnx` | `car_test_2.jpg` | YES (79.6%) | NO (94.1%) | **Gate Closed** | `30169770` | 233.3 | 6705.0 | 6952.2 |
| `best_yolo8.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 181.2 | 0.0 | 261.8 |
| `best_yolo8.onnx` | `car_test_4.jpg` | YES (80.0%) | YES (97.2%) | **Try 1** | `51G74372` | 73.9 | 1340.9 | 1428.8 |
| `best_yolo8.onnx` | `car_test_5.jpg` | YES (82.2%) | YES (93.6%) | **Try 1** | `51G77777` | 335.0 | 1710.4 | 2073.5 |
| `best_yolo8.onnx` | `car_test_6.jpg` | YES (74.8%) | YES (99.7%) | **Try 2** | `63H90394` | 173.5 | 3884.1 | 4109.9 |
| `best_yolo8.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 176.0 | 0.0 | 197.8 |
| `best_yolo8.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 162.0 | 0.0 | 186.1 |
| `best_yolo8.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 146.9 | 0.0 | 162.9 |
| `best_yolo8.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 191.7 | 0.0 | 218.8 |
| `best_yolo8.onnx` | `neg_sample_test_3.jpg` | YES (65.9%) | NO (45.0%) | **Gate Closed** | `` | 249.7 | 4988.9 | 5266.8 |
| `best_yolo8.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 234.2 | 0.0 | 267.7 |
| `best_yolo8.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 178.3 | 0.0 | 210.4 |
| `refined_best_yolo11n.onnx` | `bike_test_10.jpg` | YES (84.5%) | YES (98.1%) | **Try 1** | `43D199999` | 249.3 | 2499.0 | 2774.2 |
| `refined_best_yolo11n.onnx` | `bike_test_11.jpg` | YES (82.8%) | YES (99.9%) | **Try 2** | `29H133979` | 89.5 | 4120.1 | 4221.6 |
| `refined_best_yolo11n.onnx` | `bike_test_12.jpg` | YES (83.8%) | YES (95.0%) | **Try 2** | `61H76954` | 77.2 | 3060.0 | 3147.2 |
| `refined_best_yolo11n.onnx` | `bike_test_9.jpg` | YES (86.1%) | YES (98.2%) | **Try 1** | `59D242069` | 165.1 | 2641.8 | 2822.4 |
| `refined_best_yolo11n.onnx` | `car_test_2.jpg` | YES (78.0%) | NO (92.7%) | **Gate Closed** | `30169770` | 169.8 | 6889.0 | 7074.4 |
| `refined_best_yolo11n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 164.4 | 0.0 | 209.0 |
| `refined_best_yolo11n.onnx` | `car_test_4.jpg` | YES (75.1%) | YES (95.6%) | **Try 1** | `51G74372` | 63.2 | 1205.6 | 1282.9 |
| `refined_best_yolo11n.onnx` | `car_test_5.jpg` | YES (76.2%) | YES (98.5%) | **Try 1** | `51G77777` | 110.6 | 1547.0 | 1680.6 |
| `refined_best_yolo11n.onnx` | `car_test_6.jpg` | YES (78.5%) | YES (99.8%) | **Try 2** | `63H90394` | 81.7 | 3578.8 | 3696.3 |
| `refined_best_yolo11n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 166.9 | 0.0 | 187.8 |
| `refined_best_yolo11n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 165.5 | 0.0 | 211.7 |
| `refined_best_yolo11n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 133.3 | 0.0 | 147.8 |
| `refined_best_yolo11n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 140.5 | 0.0 | 164.8 |
| `refined_best_yolo11n.onnx` | `neg_sample_test_3.jpg` | YES (58.2%) | NO (24.5%) | **Gate Closed** | `` | 194.0 | 4317.0 | 4529.1 |
| `refined_best_yolo11n.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 132.0 | 0.0 | 156.0 |
| `refined_best_yolo11n.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 159.0 | 0.0 | 184.1 |
| `refined_best_yolo11s.onnx` | `bike_test_10.jpg` | YES (79.1%) | YES (97.6%) | **Try 1** | `43D199999` | 381.9 | 2400.5 | 2798.9 |
| `refined_best_yolo11s.onnx` | `bike_test_11.jpg` | YES (84.3%) | YES (99.9%) | **Try 2** | `29H133979` | 164.8 | 3615.8 | 3790.2 |
| `refined_best_yolo11s.onnx` | `bike_test_12.jpg` | YES (85.0%) | YES (95.3%) | **Try 2** | `61H76954` | 117.4 | 2846.9 | 2975.4 |
| `refined_best_yolo11s.onnx` | `bike_test_9.jpg` | YES (85.4%) | YES (98.3%) | **Try 1** | `59D242069` | 255.1 | 2243.9 | 2515.5 |
| `refined_best_yolo11s.onnx` | `car_test_2.jpg` | YES (76.3%) | NO (94.1%) | **Gate Closed** | `30169770` | 398.3 | 6412.7 | 6828.8 |
| `refined_best_yolo11s.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 287.1 | 0.0 | 324.0 |
| `refined_best_yolo11s.onnx` | `car_test_4.jpg` | YES (78.9%) | NO (97.7%) | **Gate Closed** | `51674372` | 248.8 | 7544.6 | 7803.5 |
| `refined_best_yolo11s.onnx` | `car_test_5.jpg` | YES (78.5%) | YES (96.9%) | **Try 1** | `51G77777` | 141.0 | 1877.5 | 2045.8 |
| `refined_best_yolo11s.onnx` | `car_test_6.jpg` | YES (78.8%) | YES (99.8%) | **Try 2** | `63H90394` | 213.9 | 3352.4 | 3603.7 |
| `refined_best_yolo11s.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 246.1 | 0.0 | 266.6 |
| `refined_best_yolo11s.onnx` | `car_test_8.jpg` | YES (59.7%) | NO (29.1%) | **Gate Closed** | `TE` | 344.7 | 7433.4 | 7805.1 |
| `refined_best_yolo11s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 319.0 | 0.0 | 337.4 |
| `refined_best_yolo11s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 411.3 | 0.0 | 463.9 |
| `refined_best_yolo11s.onnx` | `neg_sample_test_3.jpg` | YES (55.0%) | NO (40.2%) | **Gate Closed** | `` | 338.4 | 5298.6 | 5665.1 |
| `refined_best_yolo11s.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 398.4 | 0.0 | 455.1 |
| `refined_best_yolo11s.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 331.8 | 0.0 | 364.2 |
| `refined_best_yolo26n.onnx` | `bike_test_10.jpg` | YES (82.3%) | YES (94.4%) | **Try 1** | `43D199999` | 183.1 | 2661.3 | 2866.1 |
| `refined_best_yolo26n.onnx` | `bike_test_11.jpg` | YES (79.7%) | YES (99.9%) | **Try 2** | `29H133979` | 57.1 | 3612.1 | 3683.2 |
| `refined_best_yolo26n.onnx` | `bike_test_12.jpg` | YES (82.8%) | YES (96.3%) | **Try 2** | `61H76854` | 50.2 | 2793.6 | 2858.4 |
| `refined_best_yolo26n.onnx` | `bike_test_9.jpg` | YES (85.1%) | YES (97.3%) | **Try 1** | `59D242069` | 46.6 | 2470.6 | 2534.3 |
| `refined_best_yolo26n.onnx` | `car_test_2.jpg` | YES (77.3%) | NO (94.0%) | **Gate Closed** | `30169770` | 98.4 | 6139.1 | 6250.0 |
| `refined_best_yolo26n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 128.7 | 1.0 | 185.8 |
| `refined_best_yolo26n.onnx` | `car_test_4.jpg` | YES (78.7%) | YES (97.7%) | **Try 1** | `51G74372` | 122.7 | 1221.9 | 1357.6 |
| `refined_best_yolo26n.onnx` | `car_test_5.jpg` | YES (79.8%) | YES (93.5%) | **Try 1** | `51G77777` | 59.2 | 1567.1 | 1657.9 |
| `refined_best_yolo26n.onnx` | `car_test_6.jpg` | YES (80.4%) | YES (99.4%) | **Try 2** | `63H90394` | 109.8 | 3758.7 | 3927.2 |
| `refined_best_yolo26n.onnx` | `car_test_7.jpg` | YES (36.5%) | NO (0.0%) | **Gate Closed** | `NONE` | 153.1 | 5026.3 | 5201.4 |
| `refined_best_yolo26n.onnx` | `car_test_8.jpg` | YES (31.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 143.7 | 3330.5 | 3495.3 |
| `refined_best_yolo26n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 109.7 | 0.0 | 129.9 |
| `refined_best_yolo26n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 111.5 | 0.0 | 131.4 |
| `refined_best_yolo26n.onnx` | `neg_sample_test_3.jpg` | YES (60.2%) | NO (51.0%) | **Gate Closed** | `` | 109.5 | 4511.8 | 4639.3 |
| `refined_best_yolo26n.onnx` | `neg_sample_test_4.jpg` | YES (29.8%) | NO (14.3%) | **Gate Closed** | `` | 98.7 | 4787.6 | 4930.7 |
| `refined_best_yolo26n.onnx` | `neg_sample_test_5.jpg` | YES (29.7%) | NO (85.5%) | **Gate Closed** | `2` | 100.3 | 4746.1 | 4861.9 |
| `refined_best_yolo26s.onnx` | `bike_test_10.jpg` | YES (79.0%) | YES (98.1%) | **Try 1** | `43D199999` | 349.0 | 2268.2 | 2638.8 |
| `refined_best_yolo26s.onnx` | `bike_test_11.jpg` | YES (75.7%) | YES (99.8%) | **Try 2** | `29H133979` | 128.4 | 3715.9 | 3857.8 |
| `refined_best_yolo26s.onnx` | `bike_test_12.jpg` | YES (83.2%) | YES (96.9%) | **Try 2** | `61H76954` | 138.5 | 2741.9 | 2893.3 |
| `refined_best_yolo26s.onnx` | `bike_test_9.jpg` | YES (84.4%) | YES (98.3%) | **Try 1** | `59D242069` | 126.4 | 2105.3 | 2247.2 |
| `refined_best_yolo26s.onnx` | `car_test_2.jpg` | YES (77.3%) | NO (93.8%) | **Gate Closed** | `30169730` | 277.5 | 6035.7 | 6325.7 |
| `refined_best_yolo26s.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 258.7 | 0.0 | 298.9 |
| `refined_best_yolo26s.onnx` | `car_test_4.jpg` | YES (76.8%) | YES (96.0%) | **Try 1** | `51G74372` | 113.9 | 1317.3 | 1442.2 |
| `refined_best_yolo26s.onnx` | `car_test_5.jpg` | YES (79.2%) | YES (97.1%) | **Try 1** | `51G77777` | 168.6 | 1392.0 | 1581.8 |
| `refined_best_yolo26s.onnx` | `car_test_6.jpg` | YES (35.5%) | YES (99.5%) | **Try 2** | `63H90394` | 165.1 | 3266.4 | 3472.8 |
| `refined_best_yolo26s.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 262.9 | 0.0 | 286.6 |
| `refined_best_yolo26s.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 221.4 | 0.0 | 242.5 |
| `refined_best_yolo26s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 231.8 | 0.0 | 245.7 |
| `refined_best_yolo26s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 225.8 | 0.0 | 243.4 |
| `refined_best_yolo26s.onnx` | `neg_sample_test_3.jpg` | YES (62.2%) | NO (24.5%) | **Gate Closed** | `` | 243.7 | 4403.6 | 4668.3 |
| `refined_best_yolo26s.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 297.0 | 0.0 | 324.3 |
| `refined_best_yolo26s.onnx` | `neg_sample_test_5.jpg` | YES (42.9%) | NO (27.8%) | **Gate Closed** | `M` | 290.7 | 5497.8 | 5813.5 |
| `refined_best_yolov10n.onnx` | `bike_test_10.jpg` | YES (83.0%) | YES (97.3%) | **Try 1** | `43D199999` | 190.1 | 2198.0 | 2407.2 |
| `refined_best_yolov10n.onnx` | `bike_test_11.jpg` | YES (86.1%) | YES (99.8%) | **Try 2** | `29H133979` | 70.7 | 4122.5 | 4206.9 |
| `refined_best_yolov10n.onnx` | `bike_test_12.jpg` | YES (85.8%) | YES (95.9%) | **Try 2** | `61H76954` | 52.0 | 2954.0 | 3018.3 |
| `refined_best_yolov10n.onnx` | `bike_test_9.jpg` | YES (84.8%) | YES (98.7%) | **Try 1** | `59D242069` | 150.3 | 2313.1 | 2487.4 |
| `refined_best_yolov10n.onnx` | `car_test_2.jpg` | YES (75.7%) | NO (91.6%) | **Gate Closed** | `30169730` | 134.7 | 6307.7 | 6452.7 |
| `refined_best_yolov10n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 138.6 | 0.0 | 178.6 |
| `refined_best_yolov10n.onnx` | `car_test_4.jpg` | YES (77.4%) | YES (97.7%) | **Try 2** | `51G74372` | 118.7 | 4208.5 | 4344.2 |
| `refined_best_yolov10n.onnx` | `car_test_5.jpg` | YES (77.2%) | YES (91.2%) | **Try 1** | `51G77777` | 66.2 | 1362.6 | 1452.1 |
| `refined_best_yolov10n.onnx` | `car_test_6.jpg` | YES (74.1%) | YES (99.4%) | **Try 2** | `63H90394` | 109.1 | 3297.6 | 3447.9 |
| `refined_best_yolov10n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 140.7 | 0.0 | 160.6 |
| `refined_best_yolov10n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 131.2 | 0.0 | 155.7 |
| `refined_best_yolov10n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 123.2 | 0.0 | 141.2 |
| `refined_best_yolov10n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 136.4 | 0.0 | 162.0 |
| `refined_best_yolov10n.onnx` | `neg_sample_test_3.jpg` | YES (54.7%) | NO (29.3%) | **Gate Closed** | `` | 152.4 | 4687.3 | 4859.3 |
| `refined_best_yolov10n.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 134.6 | 1.0 | 160.3 |
| `refined_best_yolov10n.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 118.0 | 0.0 | 139.0 |
| `refined_best_yolov10s.onnx` | `bike_test_10.jpg` | YES (78.0%) | YES (94.5%) | **Try 1** | `43D199999` | 295.8 | 2507.3 | 2821.3 |
| `refined_best_yolov10s.onnx` | `bike_test_11.jpg` | YES (81.7%) | YES (99.7%) | **Try 1** | `29H133979` | 156.5 | 2143.2 | 2312.7 |
| `refined_best_yolov10s.onnx` | `bike_test_12.jpg` | YES (79.1%) | YES (93.8%) | **Try 2** | `61H76954` | 150.4 | 3130.5 | 3294.4 |
| `refined_best_yolov10s.onnx` | `bike_test_9.jpg` | YES (82.5%) | YES (98.2%) | **Try 1** | `59D242069` | 221.9 | 2287.0 | 2548.2 |
| `refined_best_yolov10s.onnx` | `car_test_2.jpg` | YES (70.1%) | NO (90.9%) | **Gate Closed** | `30169770` | 367.8 | 5871.7 | 6263.6 |
| `refined_best_yolov10s.onnx` | `car_test_3.jpg` | YES (39.3%) | NO (81.8%) | **Gate Closed** | `MIINWBNHGIOI17010` | 322.5 | 7944.2 | 8307.8 |
| `refined_best_yolov10s.onnx` | `car_test_4.jpg` | YES (68.7%) | YES (96.6%) | **Try 1** | `51G74372` | 125.3 | 1354.8 | 1496.7 |
| `refined_best_yolov10s.onnx` | `car_test_5.jpg` | YES (68.9%) | YES (98.1%) | **Try 1** | `51G77777` | 149.9 | 1739.2 | 1920.2 |
| `refined_best_yolov10s.onnx` | `car_test_6.jpg` | YES (66.2%) | YES (99.8%) | **Try 2** | `63H90394` | 175.3 | 3564.2 | 3786.5 |
| `refined_best_yolov10s.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 312.4 | 0.0 | 335.9 |
| `refined_best_yolov10s.onnx` | `car_test_8.jpg` | YES (28.9%) | NO (0.0%) | **Gate Closed** | `NONE` | 304.5 | 2925.7 | 3250.2 |
| `refined_best_yolov10s.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 280.4 | 0.0 | 303.0 |
| `refined_best_yolov10s.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 375.5 | 0.0 | 412.3 |
| `refined_best_yolov10s.onnx` | `neg_sample_test_3.jpg` | YES (36.4%) | NO (49.6%) | **Gate Closed** | `` | 344.2 | 4954.0 | 5319.2 |
| `refined_best_yolov10s.onnx` | `neg_sample_test_4.jpg` | YES (25.7%) | NO (36.7%) | **Gate Closed** | `7` | 298.3 | 5167.3 | 5484.7 |
| `refined_best_yolov10s.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 273.3 | 0.0 | 295.2 |
| `yolov8n.onnx` | `bike_test_10.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 255.0 | 0.0 | 286.3 |
| `yolov8n.onnx` | `bike_test_11.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 139.8 | 0.0 | 168.4 |
| `yolov8n.onnx` | `bike_test_12.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 135.8 | 0.0 | 150.8 |
| `yolov8n.onnx` | `bike_test_9.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 232.0 | 0.0 | 273.5 |
| `yolov8n.onnx` | `car_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 234.5 | 0.0 | 278.1 |
| `yolov8n.onnx` | `car_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 190.1 | 0.0 | 278.3 |
| `yolov8n.onnx` | `car_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 136.3 | 0.0 | 150.8 |
| `yolov8n.onnx` | `car_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 209.2 | 0.0 | 245.8 |
| `yolov8n.onnx` | `car_test_6.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 182.4 | 0.0 | 230.4 |
| `yolov8n.onnx` | `car_test_7.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 162.0 | 0.0 | 183.5 |
| `yolov8n.onnx` | `car_test_8.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 142.9 | 0.0 | 165.4 |
| `yolov8n.onnx` | `neg_sample_test_1.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 158.6 | 0.0 | 173.6 |
| `yolov8n.onnx` | `neg_sample_test_2.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 143.4 | 0.0 | 167.7 |
| `yolov8n.onnx` | `neg_sample_test_3.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 166.0 | 0.0 | 187.0 |
| `yolov8n.onnx` | `neg_sample_test_4.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 155.2 | 0.0 | 181.5 |
| `yolov8n.onnx` | `neg_sample_test_5.jpg` | NO (0.0%) | NO (0.0%) | **Gate Closed** | `NONE` | 160.9 | 0.0 | 186.0 |


## 7. Dynamic Evaluation History

| Timestamp | Test Image | Try Level | Resolution | Boxes | Plate Text | Valid Format | Det (ms) | OCR (ms) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| 2026-10-16 12:53:06 | `bike_test_11.jpg` | 3rd try | 1024px | 1 | `8438` | NO | 5258.7 | 566.1 |
| 2026-10-16 12:53:47 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `0445499` | NO | 5305.8 | 579.6 |
| 2026-10-16 12:55:13 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `05D49` | NO | 5076.9 | 511.8 |
| 2026-10-16 12:58:00 | `bike_test_11.jpg` | 3rd try | 1024px | 1 | `638318385335` | NO | 4082.0 | 1070.1 |
| 2026-10-16 13:03:51 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `565555555` | NO | 3460.5 | 879.0 |
| 2026-10-16 15:13:55 | `bike_test_10.jpg` | 3rd try | 1024px | 1 | `8539088999929` | NO | 6635.4 | 840.1 |
| 2026-10-16 15:13:55 | `bike_test_11.jpg` | 3rd try | 1024px | 1 | `638318385335` | NO | 900.0 | 787.2 |
| 2026-10-16 15:13:55 | `bike_test_12.jpg` | 3rd try | 1024px | 1 | `135595` | NO | 1003.0 | 746.8 |
| 2026-10-16 15:13:55 | `bike_test_9.jpg` | 3rd try | 1024px | 1 | `6BZ674420664` | NO | 827.3 | 612.7 |
| 2026-10-16 15:13:55 | `car_test_2.jpg` | 3rd try | 1024px | 1 | `261` | NO | 933.8 | 773.6 |
| 2026-10-16 15:13:55 | `car_test_3.jpg` | 3rd try | 1024px | 1 | `F5F263` | NO | 677.7 | 638.2 |
| 2026-10-16 15:13:55 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `565555555` | NO | 599.6 | 697.9 |
| 2026-10-16 15:13:55 | `car_test_5.jpg` | 3rd try | 1024px | 1 | `15337` | NO | 717.0 | 618.0 |
| 2026-10-16 15:13:55 | `car_test_6.jpg` | 3rd try | 1024px | 1 | `57A9` | NO | 870.6 | 640.6 |
| 2026-10-16 15:13:55 | `car_test_7.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 859.4 | 0.0 |
| 2026-10-16 15:13:55 | `car_test_8.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 189.4 | 0.0 |
| 2026-10-16 15:13:55 | `neg_sample_test_1.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 206.6 | 0.0 |
| 2026-10-16 15:13:55 | `neg_sample_test_2.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 172.4 | 0.7 |
| 2026-10-16 15:13:55 | `neg_sample_test_3.jpg` | 3rd try | 1024px | 1 | `583F55` | NO | 328.1 | 734.1 |
| 2026-10-16 15:13:55 | `neg_sample_test_4.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 696.3 | 0.0 |
| 2026-10-16 15:13:55 | `neg_sample_test_5.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 185.9 | 0.0 |
| 2026-10-16 15:35:12 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `565555555` | NO | 4183.8 | 810.4 |
| 2026-10-16 15:36:57 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G74372` | YES | 2908.4 | 1297.1 |
| 2026-10-16 15:40:24 | `bike_test_9.jpg` | 3rd try | 1024px | 1 | `6BZ674420664` | NO | 2990.5 | 929.4 |
| 2026-10-16 15:42:15 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `518B544342` | NO | 5789.5 | 520.6 |
| 2026-10-16 15:52:18 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `518B544342` | NO | 4012.7 | 544.9 |
| 2026-10-16 15:56:09 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G74372` | YES | 2691.0 | 1220.1 |
| 2026-10-16 16:32:24 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `71` | NO | 4630.7 | 407.1 |
| 2026-10-16 16:39:35 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `5G7439` | NO | 2871.5 | 140.5 |
| 2026-10-16 16:40:55 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `5G7439` | NO | 2861.5 | 213.1 |
| 2026-10-16 17:13:14 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `G7439` | NO | 4597.7 | 335.5 |
| 2026-10-19 10:32:31 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `G7439` | NO | 7048.8 | 360.3 |
| 2026-10-19 10:33:47 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `G73` | NO | 4481.4 | 261.4 |
| 2026-10-19 10:37:24 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G74372` | YES | 3373.9 | 1956.5 |
| 2026-10-19 10:51:29 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `1G7437` | NO | 13743.0 | 437.0 |
| 2026-10-19 10:51:56 | `bike_test_11.jpg` | 3rd try | 1024px | 1 | `1F993397` | NO | 2982.7 | 336.6 |
| 2026-10-19 10:52:15 | `car_test_5.jpg` | 1st try | 640px | 1 | `51G77777` | YES | 3691.1 | 62.4 |
| 2026-10-19 10:52:46 | `bike_test_10.jpg` | 3rd try | 1024px | 1 | `F503999` | NO | 3255.9 | 384.0 |
| 2026-10-19 11:21:59 | `bike_test_10.jpg` | 1st try | 640px | 1 | `43D199999` | YES | 4605.6 | 2511.3 |
| 2026-10-19 11:21:59 | `bike_test_11.jpg` | 1st try | 640px | 1 | `29H133979` | YES | 106.8 | 2140.7 |
| 2026-10-19 11:21:59 | `bike_test_12.jpg` | 2nd try | 640px | 1 | `61H76854` | YES | 69.0 | 2965.7 |
| 2026-10-19 11:21:59 | `bike_test_9.jpg` | 1st try | 640px | 1 | `59D242069` | YES | 69.0 | 2584.9 |
| 2026-10-19 11:21:59 | `car_test_2.jpg` | 3rd try | 1024px | 1 | `30169770` | NO | 134.0 | 6323.4 |
| 2026-10-19 11:21:59 | `car_test_3.jpg` | 3rd try | 1024px | 1 | `2TRUNGT` | NO | 163.0 | 7961.3 |
| 2026-10-19 11:21:59 | `car_test_4.jpg` | 1st try | 640px | 1 | `51G74372` | YES | 59.0 | 1361.2 |
| 2026-10-19 11:21:59 | `car_test_5.jpg` | 1st try | 640px | 1 | `51G77777` | YES | 78.5 | 1456.8 |
| 2026-10-19 11:21:59 | `car_test_6.jpg` | 2nd try | 640px | 1 | `63H90394` | YES | 83.0 | 3329.2 |
| 2026-10-19 11:21:59 | `car_test_7.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 152.4 | 0.0 |
| 2026-10-19 11:21:59 | `car_test_8.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 127.0 | 0.0 |
| 2026-10-19 11:21:59 | `neg_sample_test_1.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 127.2 | 0.0 |
| 2026-10-19 11:21:59 | `neg_sample_test_2.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 104.0 | 0.0 |
| 2026-10-19 11:21:59 | `neg_sample_test_3.jpg` | 3rd try | 1024px | 1 | `B` | NO | 128.0 | 4735.5 |
| 2026-10-19 11:21:59 | `neg_sample_test_4.jpg` | 3rd try | 1024px | 1 | `` | NO | 153.0 | 6140.2 |
| 2026-10-19 11:21:59 | `neg_sample_test_5.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 217.4 | 0.0 |
| 2026-10-19 11:22:43 | `bike_test_10.jpg` | 3rd try | 1024px | 1 | `F503999` | NO | 4159.5 | 593.7 |
| 2026-10-19 11:22:43 | `bike_test_11.jpg` | 3rd try | 1024px | 1 | `1F993397` | NO | 597.2 | 210.0 |
| 2026-10-19 11:22:43 | `bike_test_12.jpg` | 3rd try | 1024px | 1 | `F756854` | NO | 459.7 | 258.6 |
| 2026-10-19 11:22:43 | `bike_test_9.jpg` | 3rd try | 1024px | 1 | `5902206` | NO | 469.7 | 240.3 |
| 2026-10-19 11:22:43 | `car_test_2.jpg` | 1st try | 640px | 1 | `30A697790` | YES | 210.6 | 38.7 |
| 2026-10-19 11:22:43 | `car_test_3.jpg` | 3rd try | 1024px | 1 | `5` | NO | 505.9 | 140.4 |
| 2026-10-19 11:22:43 | `car_test_4.jpg` | 3rd try | 1024px | 1 | `1G7437` | NO | 432.8 | 121.8 |
| 2026-10-19 11:22:43 | `car_test_5.jpg` | 1st try | 640px | 1 | `51G77777` | YES | 200.3 | 42.3 |
| 2026-10-19 11:22:43 | `car_test_6.jpg` | 3rd try | 1024px | 1 | `3890345` | NO | 605.2 | 296.3 |
| 2026-10-19 11:22:43 | `car_test_7.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 531.2 | 0.0 |
| 2026-10-19 11:22:43 | `car_test_8.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 517.7 | 0.0 |
| 2026-10-19 11:22:43 | `neg_sample_test_1.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 199.5 | 0.0 |
| 2026-10-19 11:22:43 | `neg_sample_test_2.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 150.8 | 0.0 |
| 2026-10-19 11:22:43 | `neg_sample_test_3.jpg` | 3rd try | 1024px | 1 | `` | NO | 369.1 | 137.8 |
| 2026-10-19 11:22:43 | `neg_sample_test_4.jpg` | 3rd try | 1024px | 1 | `7` | NO | 466.0 | 134.6 |
| 2026-10-19 11:22:43 | `neg_sample_test_5.jpg` | 3rd try (exhausted) | 1024px | 0 | `NONE` | NO | 420.1 | 0.0 |
