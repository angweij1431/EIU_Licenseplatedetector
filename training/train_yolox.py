# Master Server Training Runner for YOLOX-Nano License Plate Detector
import argparse
import os
import subprocess
import sys

# Ensure training directory is in python path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Train YOLOX-Nano on Server with Auto-Checkpointing")
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Total training epochs (default: 150)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size per GPU (default: 32)",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=int,
        default=1,
        help="Number of GPUs to train on (default: 1)",
    )
    parser.add_argument(
        "--weights",
        "-c",
        default="yolox_nano.pth",
        help="Pretrained baseline weights (default: yolox_nano.pth)",
    )
    parser.add_argument(
        "--data-dir",
        default="dataset_coco",
        help="Path to COCO dataset folder containing train/val images and annotations (default: dataset_coco)",
    )
    parser.add_argument(
        "--exp-file",
        "-f",
        default="training/exps/yolox_nano_plate.py",
        help="Experiment file path (default: training/exps/yolox_nano_plate.py)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest_ckpt.pth if previous run was interrupted",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 half-precision training",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        default=True,
        help="Automatically export best_ckpt.pth to detector.onnx when training completes (default: True)",
    )
    args = parser.parse_args()

    # Pass dynamic settings to experiment file via environment variables
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["YOLOX_EPOCHS"] = str(args.epochs)
    os.environ["YOLOX_DATA_DIR"] = os.path.abspath(args.data_dir)

    print("=" * 70)
    print("EIU LPR: YOLOX-Nano Dedicated Server Training")
    print("=" * 70)
    print(f"Target Epochs : {args.epochs}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"GPU Count     : {args.devices}")
    print(f"Dataset Path  : {os.path.abspath(args.data_dir)}")
    print(f"Experiment    : {args.exp_file}")
    print(f"Mixed Precision (FP16): {not args.no_fp16}")
    print("-" * 70)
    print("CHECKPOINT AUTOMATION:")
    print("  - Latest Save (Every Epoch) : YOLOX_outputs/yolox_nano_plate/latest_ckpt.pth")
    print("  - Best Validation Checkpoint: YOLOX_outputs/yolox_nano_plate/best_ckpt.pth")
    print("  - If interrupted, run with  : python training/train_yolox.py --resume")
    print("=" * 70)

    # Base training weights or resume checkpoint
    ckpt_target = args.weights
    if args.resume:
        resume_path = "YOLOX_outputs/yolox_nano_plate/latest_ckpt.pth"
        if os.path.exists(resume_path):
            ckpt_target = resume_path
            print(f"[RESUME] Resuming training from {resume_path}...")
        else:
            print(f"[WARNING] Resume requested but {resume_path} not found. Starting fresh from {args.weights}.")

    # Build YOLOX training command
    cmd = [
        sys.executable,
        "-m", "yolox.tools.train",
        "-f", args.exp_file,
        "-d", str(args.devices),
        "-b", str(args.batch_size),
        "-c", ckpt_target,
    ]

    if not args.no_fp16:
        cmd.append("--fp16")

    if args.resume and os.path.exists("YOLOX_outputs/yolox_nano_plate/latest_ckpt.pth"):
        cmd.append("--resume")

    print("\nExecuting command:")
    print(" ".join(cmd) + "\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with return code {e.returncode}")
        sys.exit(e.returncode)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] {args.epochs} Epochs Completed!")
    print("=" * 70)

    # Auto export to ONNX
    if args.export_onnx:
        best_ckpt = "YOLOX_outputs/yolox_nano_plate/best_ckpt.pth"
        if os.path.exists(best_ckpt):
            print("\nTriggering automated ONNX export from best_ckpt.pth...")
            from export_onnx import export_onnx
            export_onnx(
                exp_file=args.exp_file,
                ckpt_path=best_ckpt,
                output_path="detector.onnx",
            )
        else:
            print(f"[WARNING] Could not find {best_ckpt} for automated ONNX export.")


if __name__ == "__main__":
    main()
