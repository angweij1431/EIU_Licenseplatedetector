# Golden Verification Dataset Specification

Reference: `golden_dataset/ground_truth.json`  
Image Directory: `test_images/`  
Regulatory Standard: Circular 24/2023/TT-BCA  

## 1. Overview
The golden dataset consists of 20 verified test images designed for automated regression testing and validation of the `eiu_lpr` vision engine.

The suite evaluates five core operating conditions:
1. Standard passenger automobiles and commercial trucks.
2. Dual-row square motorcycle plates with multi-line layout.
3. Adverse environmental conditions (atmospheric fog, glare, low contrast).
4. Boundary and rejection cases (sub-resolution distant crops, severe motion blur, steep perspective angles > 60°).
5. Campus negative background scenes (pavement, bike racks, traffic signs, gates) to ensure a 0% false alarm rate.

## 2. Ground Truth Matrix

| ID | Filename | Vehicle Category | Ground Truth (Raw) | Target (Normalized) | Valid Syntax | Expected Result | Challenge / Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `golden-01` | `car_test_2.jpg` | Automobile | `30A-697.70` | `30A69770` | Yes | Detected & Recognized | Frontal passenger car |
| `golden-02` | `car_test_4.jpg` | Automobile | `51G-743.72` | `51G74372` | Yes | Detected & Recognized | High-resolution rear plate |
| `golden-03` | `car_test_5.jpg` | Automobile | `51G-777.77` | `51G77777` | Yes | Detected & Recognized | Repetitive digit series |
| `golden-04` | `car_test_6.jpg` | Commercial Truck | `63H-903.94` | `63H90394` | Yes | Detected & Recognized | Heavy commercial vehicle |
| `golden-05` | `car_foggy_test.jpg` | Adverse Weather | `30K-279.76` | `30K27976` | Yes | Detected & Recognized | Fog, glare, and low contrast |
| `golden-06` | `bike_test_9.jpg` | Motorcycle | `59-D2 420.69` | `59D242069` | Yes | Detected & Recognized | Dual-row square layout |
| `golden-07` | `bike_test_10.jpg` | Motorcycle | `43-D1 999.99` | `43D199999` | Yes | Detected & Recognized | D vs 0 optical fidelity test |
| `golden-08` | `bike_test_11.jpg` | Motorcycle | `29-H1 339.79` | `29H133979` | Yes | Detected & Recognized | Two-line H series |
| `golden-09` | `bike_test_12.jpg` | Motorcycle | `61-H7 6854` | `61H76854` | Yes | Detected & Recognized | Small crop requiring enhancement |
| `golden-10` | `car_sample_29D.jpg` | Commercial Van | `29D-110.77` | `29D11077` | Yes | Detected & Recognized | Fleet commercial van |
| `golden-11` | `car_sample_51L.jpg` | Automobile | `51L-953.46` | `51L95346` | Yes | Detected & Recognized | Modern registration series |
| `golden-12` | `car_test_3.jpg` | Distant / Tiny | `29M-6840` | `29M6840` | Yes | Safe Sub-threshold | Sub-resolution (<45px plate width) |
| `golden-13` | `car_test_7.jpg` | Obscured Scene | Non-plate | None | No | Zero Candidates | Obscured background |
| `golden-14` | `car_test_8.jpg` | Motion Blur | Non-plate | None | No | Zero Candidates | High-speed shutter blur |
| `golden-15` | `car_very_angled.jpg` | Extreme Angle | Non-plate | None | No | Zero Candidates | Perspective angle > 60° |
| `golden-16` | `neg_sample_test_1.jpg` | Negative Scene | None | None | No | Zero Candidates | Empty asphalt road |
| `golden-17` | `neg_sample_test_2.jpg` | Negative Scene | None | None | No | Zero Candidates | Campus bicycle frame |
| `golden-18` | `neg_sample_test_3.jpg` | Negative Scene | None | None | No | Zero Candidates | Reflective road sign |
| `golden-19` | `neg_sample_test_4.jpg` | Negative Scene | None | None | No | Zero Candidates | Pedestrian barrier arm |
| `golden-20` | `neg_sample_test_5.jpg` | Negative Scene | None | None | No | Zero Candidates | Tree walkway |

## 3. Provenance and Compliance
- **Data Origin**: Captured at Eastern International University (EIU) campus access checkpoints and campus roadways.
- **Privacy Compliance**: All images were collected strictly for academic access control research. Personally Identifiable Information (PII) is confined to in-memory evaluation and is not transmitted to external cloud services.
- **License**: Internal test suite assets under academic and research evaluation permissions.

