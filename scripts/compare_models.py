# Side-by-side performance and accuracy comparison: YOLO11n vs YOLOX-Nano
import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import onnxruntime as ort


def run_benchmark(
    sess: ort.InferenceSession,
    blob: np.ndarray,
    input_name: str,
    iterations: int = 50,
) -> Tuple[float, float, float]:
    for _ in range(5):
        sess.run(None, {input_name: blob})
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess.run(None, {input_name: blob})
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.median(times)), float(np.min(times))


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO11n vs YOLOX-Nano")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark sample count")
    args = parser.parse_args()

    yolo11_path = "models/archive/v0.1_yolo11n/detector_yolo11n_v0.1.onnx"
    yolox_path = "models/detector.onnx"

    if not os.path.exists(yolo11_path):
        print(f"Error: YOLO11n model not found at {yolo11_path}")
        return
    if not os.path.exists(yolox_path):
        print(f"Error: YOLOX model not found at {yolox_path}")
        return

    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.log_severity_level = 3

    sess_yolo11 = ort.InferenceSession(yolo11_path, sess_options=sess_opt, providers=["CPUExecutionProvider"])
    sess_yolox = ort.InferenceSession(yolox_path, sess_options=sess_opt, providers=["CPUExecutionProvider"])

    yolo11_size = os.path.getsize(yolo11_path) / (1024 * 1024)
    yolox_size = os.path.getsize(yolox_path) / (1024 * 1024)

    test_img_path = "test_images/car_test_2.jpg"
    img = cv2.imread(test_img_path)
    h0, w0 = img.shape[:2]

    # Preprocessing: YOLOX-Nano (top-left aligned, BGR, [0, 255])
    r = min(640 / h0, 640 / w0)
    nh, nw = int(h0 * r), int(w0 * r)
    can_yx = np.full((640, 640, 3), 114, dtype=np.uint8)
    can_yx[:nh, :nw] = cv2.resize(img, (nw, nh))
    blob_yx = np.ascontiguousarray(can_yx.transpose((2, 0, 1))[np.newaxis, ...], dtype=np.float32)

    # Preprocessing: YOLO11n (centered, RGB, [0, 1])
    res_y11 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (nw, nh))
    can_y11 = cv2.copyMakeBorder(
        res_y11, (640 - nh) // 2, 640 - nh - (640 - nh) // 2,
        (640 - nw) // 2, 640 - nw - (640 - nw) // 2,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    blob_y11 = np.ascontiguousarray(can_y11.transpose((2, 0, 1))[np.newaxis, ...], dtype=np.float32) / 255.0

    print("=" * 75)
    print("DETECTOR PERFORMANCE COMPARISON: YOLO11n vs Megvii YOLOX-Nano")
    print("=" * 75)
    print(f"{'Attribute':<30} | {'YOLO11n (v0.1)':<18} | {'YOLOX-Nano (v0.2)':<18}")
    print("-" * 75)
    print(f"{'Software License':<30} | {'AGPL-3.0 (Copyleft)':<18} | {'Apache 2.0 (Permissive)':<18}")
    print(f"{'Model File Size':<30} | {f'{yolo11_size:.2f} MB':<18} | {f'{yolox_size:.2f} MB':<18}")
    print(f"{'Output Tensor Shape':<30} | {'[1, 5, 8400]':<18} | {'[1, 8400, 6]':<18}")
    print(f"{'Decoupled Head':<30} | {'No (Coupled)':<18} | {'Yes (Obj * Cls)':<18}")

    y11_mean, y11_p50, y11_min = run_benchmark(sess_yolo11, blob_y11, sess_yolo11.get_inputs()[0].name, args.iterations)
    yx_mean, yx_p50, yx_min = run_benchmark(sess_yolox, blob_yx, sess_yolox.get_inputs()[0].name, args.iterations)

    print(f"{'Mean CPU Latency':<30} | {f'{y11_mean:.1f} ms':<18} | {f'{yx_mean:.1f} ms':<18}")
    print(f"{'Median CPU Latency (p50)':<30} | {f'{y11_p50:.1f} ms':<18} | {f'{yx_p50:.1f} ms':<18}")
    print(f"{'Min CPU Latency':<30} | {f'{y11_min:.1f} ms':<18} | {f'{yx_min:.1f} ms':<18}")

    speedup = ((y11_mean - yx_mean) / y11_mean) * 100
    print(f"{'Latency Difference':<30} | {'Baseline':<18} | {f'{speedup:+.1f}% faster':<18}")

    # Accuracy / Confidence Comparison across Golden Dataset
    gt_path = "golden_dataset/ground_truth.json"
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)["images"]

        print("\n" + "=" * 75)
        print("GOLDEN DATASET CONFIDENCE COMPARISON (20 Images)")
        print("=" * 75)
        print(f"{'Image / Category':<35} | {'Type':<4} | {'YOLO11n':<8} | {'YOLOX':<8} | Change")
        print("-" * 75)

        for item in gt:
            p = os.path.join("test_images", item["filename"])
            if not os.path.exists(p):
                continue
            t_img = cv2.imread(p)
            th, tw = t_img.shape[:2]

            # YOLOX
            tr = min(640 / th, 640 / tw)
            tnh, tnw = int(th * tr), int(tw * tr)
            tcan_yx = np.full((640, 640, 3), 114, dtype=np.uint8)
            tcan_yx[:tnh, :tnw] = cv2.resize(t_img, (tnw, tnh))
            tb_yx = np.ascontiguousarray(tcan_yx.transpose((2, 0, 1))[np.newaxis, ...], dtype=np.float32)
            out_yx = sess_yolox.run(None, {sess_yolox.get_inputs()[0].name: tb_yx})[0][0]
            score_yx = float(np.max(out_yx[:, 4] * out_yx[:, 5]))

            # YOLO11n
            res_y11 = cv2.resize(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB), (tnw, tnh))
            tcan_y11 = cv2.copyMakeBorder(
                res_y11, (640 - tnh) // 2, 640 - tnh - (640 - tnh) // 2,
                (640 - tnw) // 2, 640 - tnw - (640 - tnw) // 2,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )
            tb_y11 = np.ascontiguousarray(tcan_y11.transpose((2, 0, 1))[np.newaxis, ...], dtype=np.float32) / 255.0
            out_y11 = sess_yolo11.run(None, {sess_yolo11.get_inputs()[0].name: tb_y11})[0][0]
            score_y11 = float(np.max(out_y11[4, :]))

            exp = "POS" if item["expected_detection"] else "NEG"
            diff = score_yx - score_y11
            note = f"{diff:+.2f}"
            if exp == "NEG" and score_yx < score_y11:
                note += " (better suppression)"
            elif exp == "POS" and score_yx > score_y11:
                note += " (higher confidence)"

            lbl = f"{item['filename']} ({item.get('category','').split('(')[0].strip()})"
            print(f"{lbl:<35} | {exp:<4} | {score_y11:<8.3f} | {score_yx:<8.3f} | {note}")


if __name__ == "__main__":
    main()

