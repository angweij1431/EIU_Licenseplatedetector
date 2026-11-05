# License Plate Detector Training (YOLOX)

[UPDATE FROM: 5/11 - Released 150-epoch retrained YOLOX-Nano detector (peak validation at Epoch 149 with mAP@50:95 of 0.668, achieving 100% Golden Dataset accuracy).]
[UPDATE FROM: 2/11 - Training YOLOX (Under apache 2.0 license)]

---

## 1. Overview

This directory contains the training configuration and execution guide for fine-tuning the vehicle license plate detector using Megvii YOLOX under the Apache 2.0 license.

The objective of this setup is to provide a clean-room, commercially permissive detector that replaces models trained under copyleft licenses such as AGPL-3.0 (GNU Affero General Public License v3.0).

---

## 2. Key Differences from YOLO11n

* **Software License**: YOLOX is released under the Apache 2.0 license, which allows embedding into proprietary, closed-source commercial applications without copyleft obligations. YOLO11n is governed by AGPL-3.0.
* **Detection Head Architecture**: YOLO11n uses a coupled head where classification and regression share convolutional features. YOLOX uses a decoupled head that separates bounding box regression from category classification, improving training convergence on rectangular objects.
* **Output Tensor Layout**:
  * YOLO11n emits a channels-first tensor of shape `[1, 5, 8400]`.
  * YOLOX emits a proposal-first tensor of shape `[1, 8400, 6]` (or `[1, 8400, 5]` if single-class scores are merged).
* **Confidence Computation**: YOLOX calculates total confidence as the product of objectness and class probability (`objectness * class_probability`), whereas YOLO11 outputs a single class score directly.

---

## 3. Output Tensor Specification & Column Attributes

The ONNX (Open Neural Network Exchange) model exported with the `--decode_in_inference` flag outputs a 3D array of shape `[1, 8400, 6]`.

The dimension breakdown is:
* Dimension 0: Batch size (`1`)
* Dimension 1: Candidate proposal count (`8400` anchors across three feature pyramid strides: 8, 16, and 32 on a 640x640 input)
* Dimension 2: Bounding box and score attributes (`6` columns)

### Column Index Definitions

| Column | Field Name | Description | Coordinate Space |
| :---: | :--- | :--- | :--- |
| `0` | `cx` | Center X of bounding box | 640x640 letterbox canvas |
| `1` | `cy` | Center Y of bounding box | 640x640 letterbox canvas |
| `2` | `bw` | Width of bounding box | 640x640 letterbox canvas |
| `3` | `bh` | Height of bounding box | 640x640 letterbox canvas |
| `4` | `obj_conf` | Objectness probability score | Range 0.0 to 1.0 |
| `5` | `cls_conf` | Class probability score (`plate`) | Range 0.0 to 1.0 |

### Coordinate Mapping Formula

To convert the bounding box from letterbox coordinates to the original image coordinates:

* `x1 = (cx - (bw / 2) - pad_x) / scale`
* `y1 = (cy - (bh / 2) - pad_y) / scale`
* `x2 = (cx + (bw / 2) - pad_x) / scale`
* `y2 = (cy + (bh / 2) - pad_y) / scale`

Coordinates are then clipped to `[0, image_width]` and `[0, image_height]` before Non-Maximum Suppression (NMS).

---

## 4. Training Guide (Google Colab / Local GPU)

The turnkey notebook is located at `training/01_train_yolox_nano.ipynb`.

### Execution Steps

1. **Environment Setup**: Clone the official Megvii YOLOX repository and install dependencies in editable mode:
   ```bash
   git clone https://github.com/Megvii-BaseDetection/YOLOX.git
   cd YOLOX && pip install -v -e .
   ```
2. **Download Base Weights**: Fetch the official Apache 2.0 pretrained checkpoint:
   ```bash
   wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
   ```
3. **Dataset Setup**: Prepare annotations in COCO (Common Objects in Context) JSON format (`train/_annotations.coco.json` and `valid/_annotations.coco.json`).
4. **Epoch Configuration**: The baseline configuration is set to **40 epochs** (`self.max_epoch = 40`) to match prior production baselines. If training loss continues to descend steadily at epoch 40, this value can be increased to 50 or 60 epochs.
5. **Launch Training**: Run the training module using FP16 (16-bit floating point) mixed precision (use `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for PyTorch 2.6+ compatibility):
   ```bash
   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -m yolox.tools.train -f exps/custom/yolox_nano_plate.py -d 1 -b 16 --fp16 -c yolox_nano.pth
   ```
6. **Export to ONNX**: Export the best checkpoint (`best_ckpt.pth`) with inference decoding enabled using the TorchScript tracer (`dynamo=False`):
   ```python
   import torch
   from yolox.exp import get_exp

   exp = get_exp("exps/custom/yolox_nano_plate.py")
   model = exp.get_model()
   ckpt = torch.load("YOLOX_outputs/yolox_nano_plate/best_ckpt.pth", map_location="cpu", weights_only=False)
   model.load_state_dict(ckpt["model"])
   model.eval()
   model.head.decode_in_inference = True

   dummy_input = torch.randn(1, 3, 640, 640)
   torch.onnx.export(
       model, dummy_input, "detector.onnx",
       input_names=["images"], output_names=["output"],
       opset_version=11, dynamo=False
   )
   ```
7. **Deploy**: Move `detector.onnx` into `models/detector.onnx`.

---

## 5. Benchmarks & Validation Record

*Recorded on Linux WSL2 x86_64 CPU (16 logical cores, ONNX Runtime CPUExecutionProvider)*

| Metric / Stage | Measured Value | Target / Baseline | Status |
| :--- | :--- | :--- | :--- |
| **Model Size** | 8.67 MB (9,088,879 bytes) | < 10.0 MB | PASS |
| **License Compliance** | Apache 2.0 (Clean-Room) | Permissive (No copyleft) | PASS |
| **Cold Startup Latency** | 278.1 ms | < 500 ms | PASS |
| **Detection Mean Latency** | 52.2 ms (p50: 50.6 ms, min: 45.4 ms) | < 250 ms | PASS |
| **OCR Mean Latency** | 4.5 ms (p50: 4.2 ms, min: 3.5 ms) | < 50 ms | PASS |
| **End-to-End Mean Latency** | 76.7 ms (p50: 73.9 ms, min: 54.2 ms) | < 350 ms | PASS |
| **Golden Dataset Precision** | 100% (10/10 verified matches) | >= 90% | PASS |
| **Safe Rejections (Negatives)** | 100% (9/9 safe rejections, 0 leaks) | >= 80% | PASS |
| **Full Unit/Regression Suite** | 13/13 tests passed | 100% pass | PASS |
