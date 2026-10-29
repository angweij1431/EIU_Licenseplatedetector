# Golden Verification Dataset Specification

Reference: `golden_dataset/ground_truth.json`  
Image Directory: `test_images/`  
Regulatory Standard: Circular 24/2023/TT-BCA  

## 1. Overview
The golden dataset consists of 20 verified test images designed for automated regression testing and validation of the `eiu_lpr` vision engine.

The suite evaluates five core operating conditions:
1. Standard passenger automobiles, taxis, SUVs, and commercial vans.
2. Dual-row square motorcycle plates with multi-line layout and cargo obstruction.
3. Challenging real-world edge cases (duplicate plates on identical vehicles, multi-vehicle queues).
4. Boundary and rejection cases (sub-resolution distant crops, extreme angle perspective, masked bumpers).
5. Open landscape scenes and complex traffic congestion to ensure zero false positive proposals.

## 2. Ground Truth Matrix

| ID | Filename | Vehicle Category | Ground Truth (Raw) | Target (Normalized) | Valid Syntax | Expected Result | Challenge / Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `golden-01` | `car_test_2.jpg` | Automobile (Taxi) | `30A-697.70` | `30A69770` | Yes | Detected & Recognized | Frontal passenger car / taxi in urban traffic |
| `golden-02` | `car_test_4.jpg` | Automobile (SUV) | `51G-743.72` | `51G74372` | Yes | Detected & Recognized | Front bumper plate on an off-road SUV in parking garage |
| `golden-03` | `car_test_5.jpg` | Automobile (Sedan) | `51G-777.77` | `51G77777` | Yes | Detected & Recognized | Rear view of luxury sedan with repetitive digit series (777.77) |
| `golden-04` | `car_test_6.jpg` | Motorcycle | `63-H9 0394` | `63H90394` | Yes | Detected & Recognized | Motorcycle rear view carrying bulky cargo (window/door frame) |
| `golden-05` | `car_foggy_test.jpg` | Automobile (Duplicate) | `30K-279.76` | `30K27976` | Yes | Detected & Recognized | Two luxury SUVs parked side-by-side with identical duplicate plates |
| `golden-06` | `bike_test_9.jpg` | Motorcycle | `59-D2 420.69` | `59D242069` | Yes | Detected & Recognized | Dual-row square motorcycle plate captured from rear |
| `golden-07` | `bike_test_10.jpg` | Motorcycle | `43-D1 999.99` | `43D199999` | Yes | Detected & Recognized | Dual-row motorcycle plate with high repetitive digits (999.99) |
| `golden-08` | `bike_test_11.jpg` | Motorcycle | `29-H1 339.79` | `29H133979` | Yes | Detected & Recognized | Dual-row motorcycle plate in dense multi-bike parking/store setting |
| `golden-09` | `bike_test_12.jpg` | Motorcycle | `61-H7 6854` | `61H76854` | Yes | Detected & Recognized | Dual-row 4-digit motorcycle plate in moving traffic |
| `golden-10` | `car_sample_29D.jpg` | Automobile (Hatchback) | `29D-110.77` | `29D11077` | Yes | Detected & Recognized | Compact passenger car with commercial D-series registration |
| `golden-11` | `car_sample_51L.jpg` | Commercial Van | `51L-953.46` | `51L95346` | Yes | Detected & Recognized | Frontal delivery van with modern L-series registration |
| `golden-12` | `car_test_3.jpg` | Queued Traffic | `29M-6840` | `29M6840` | Yes | Safe Sub-threshold | Vehicle queue at inspection station; foreground sedan plate sub-threshold |
| `golden-13` | `car_test_7.jpg` | Motorcycle (High Angle) | None | None | No | Zero Candidates | Two touring motorcycles on highway shoulder; steep angle |
| `golden-14` | `car_test_8.jpg` | Motorcycle (Rural) | None | None | No | Zero Candidates | Motorcycle parked in rural landscape beside field and mountain |
| `golden-15` | `car_very_angled.jpg` | Light Truck | None | None | No | Zero Candidates | Angled driving school truck with 'TẬP LÁI' sign masking bumper plate |
| `golden-16` | `neg_sample_test_1.jpg` | Negative Scene | None | None | No | Zero Candidates | Open sky, clouds, and tree landscape (0 false detections) |
| `golden-17` | `neg_sample_test_2.jpg` | Negative Scene | None | None | No | Zero Candidates | Empty rural asphalt road and markings (0 false detections) |
| `golden-18` | `neg_sample_test_3.jpg` | Negative Scene | None | None | No | Zero Candidates | Massive city traffic jam gridlock with heavy visual clutter |
| `golden-19` | `neg_sample_test_4.jpg` | Negative Scene | None | None | No | Zero Candidates | Dense oncoming commuter motorcycle flow on city avenue |
| `golden-20` | `neg_sample_test_5.jpg` | Negative Scene | None | None | No | Zero Candidates | Dense rush hour street crowd with helmeted riders |

## 3. Provenance and Compliance
- **Data Origin**: Collected for academic vehicle access control research and CerberusEdge validation.
- **Privacy Compliance**: All images were collected strictly for access control evaluation. Personally Identifiable Information (PII) is confined to in-memory evaluation and is not transmitted to external cloud services.
- **License**: Internal test suite assets under academic and research evaluation permissions.
