# YOLOX-Nano Training and Troubleshooting Guide

## 1. System Specifications

* **Model**: Megvii YOLOX-Nano (0.9M parameters, Apache 2.0 license)
* **Target Hardware**: Edge CPU deployment via ONNX Runtime (Open Neural Network Exchange Runtime)
* **Training Server**: Dell PowerEdge R760, Windows Server
* **Server GPU**: Dual NVIDIA RTX PRO 4000 Blackwell (24GB VRAM, Compute Capability 12.0 / `sm_120`)

> Note on Blackwell Architecture: The NVIDIA RTX PRO 4000 Blackwell GPU utilizes CUDA compute capability `sm_120`. Standard stable PyTorch builds (CUDA 12.1 and 12.4) only support up to `sm_90`. PyTorch Nightly with CUDA 12.8 (`cu128`) is required to run natively on this hardware.

---

## 2. Quickstart Training Guide

### Step 1: Environment Setup
Inside the project virtual environment (`.venv`), install dependencies:
```powershell
pip install -r training/requirements-train.txt
pip install --pre --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify GPU availability:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0))"
```

### Step 2: Clone YOLOX and Download Base Weights
```powershell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install --no-build-isolation -e .
cd ..
curl -L -o yolox_nano.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
```

### Step 3: Windows Evaluation Patch (Required)
To prevent C++ compilation errors on Windows systems without Visual Studio compilers, force standard Python evaluation:
```powershell
(Get-Content YOLOX\yolox\evaluators\coco_evaluator.py) -replace 'from yolox\.layers import COCOeval_opt as COCOeval', 'from pycocotools.cocoeval import COCOeval' | Set-Content YOLOX\yolox\evaluators\coco_evaluator.py
```

### Step 4: Run Training
```powershell
python training/train_yolox.py --epochs 150 --batch-size 32 --gpu 0
```
* For faster iteration on the 24GB Blackwell GPU, use `--batch-size 64`.
* Checkpoints are automatically saved to `YOLOX_outputs/yolox_nano_plate/latest_ckpt.pth` every epoch.

### Step 5: Pausing and Resuming
* To stop: Press `Ctrl + C` in PowerShell.
* To resume:
```powershell
python training/train_yolox.py --resume
```

### Step 6: Export to ONNX
```powershell
python training/export_onnx.py -f training/exps/yolox_nano_plate.py -c YOLOX_outputs/yolox_nano_plate/best_ckpt.pth -o models/detector.onnx
```

---

## 3. Troubleshooting Reference

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `CUDA error: no kernel image is available for execution` | PyTorch wheel lacks `sm_120` binaries for Blackwell. | Install PyTorch Nightly with CUDA 12.8 (`cu128`). |
| `RuntimeError: Ninja is required to load C++ extensions` | YOLOX attempts to JIT-compile `fast_cocoeval` at Epoch 5 on Windows. | Run the PowerShell patch in Step 3 to use standard `pycocotools`. |
| `head.reg_convs.2.1.conv.weight is not in the ckpt` | Normal transfer learning message: 80-class COCO head re-initialized for 1 class. | Benign warning; no action needed. |
| `[WinError 206] The filename or extension is too long` | Windows 260-character path limit during `onnxsim` installation. | Do not use `onnxsim`; use native TorchScript export in `training/export_onnx.py`. |
| `nvidia-smi` shows `No running processes found` | Windows Display Driver Model (WDDM) hides console compute PIDs. | Check memory usage in the top table or use Windows Task Manager GPU graph. |

