# Standalone ONNX export script for YOLOX-Nano license plate detector
import argparse
import os
import sys


def export_onnx(
    exp_file: str,
    ckpt_path: str,
    output_path: str = "detector.onnx",
    opset: int = 11,
):
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed. Please install torch with CUDA support on your training server.")
        sys.exit(1)
    try:
        from yolox.exp import get_exp
    except ImportError:
        print("[ERROR] YOLOX is not installed. Please run: pip install -e . inside your cloned YOLOX repository.")
        sys.exit(1)

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading experiment config: {exp_file}")
    exp = get_exp(exp_file)
    model = exp.get_model()

    print(f"Loading weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.head.decode_in_inference = True

    dummy_input = torch.randn(1, 3, 640, 640)
    print(f"Exporting model to ONNX: {output_path} (opset {opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Successfully exported {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOX PyTorch checkpoint to ONNX")
    parser.add_argument(
        "-f", "--exp-file",
        default="training/exps/yolox_nano_plate.py",
        help="Path to YOLOX experiment file (default: training/exps/yolox_nano_plate.py)",
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="YOLOX_outputs/yolox_nano_plate/best_ckpt.pth",
        help="Path to .pth checkpoint file (default: YOLOX_outputs/yolox_nano_plate/best_ckpt.pth)",
    )
    parser.add_argument(
        "-o", "--output",
        default="detector.onnx",
        help="Output ONNX filename (default: detector.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    args = parser.parse_args()

    export_onnx(
        exp_file=args.exp_file,
        ckpt_path=args.ckpt,
        output_path=args.output,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
