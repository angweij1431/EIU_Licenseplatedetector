# Non-destructive side-by-side benchmark comparing baseline vs candidate detector
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import onnxruntime as ort

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))
from eiu_lpr import LPRPipeline


def benchmark_session(sess: ort.InferenceSession, blob: np.ndarray, iterations: int = 50) -> Tuple[float, float, float]:
    input_name = sess.get_inputs()[0].name
    for _ in range(5):
        sess.run(None, {input_name: blob})
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess.run(None, {input_name: blob})
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.median(times)), float(np.min(times))


def main():
    parser = argparse.ArgumentParser(description="Non-destructive detector candidate evaluation")
    parser.add_argument(
        "--candidate",
        "-c",
        default="models/detector_candidate.onnx",
        help="Path to candidate detector ONNX file (default: models/detector_candidate.onnx)",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        default="models/detector.onnx",
        help="Path to current baseline detector ONNX file (default: models/detector.onnx)",
    )
    parser.add_argument(
        "--ocr-model",
        default="models/cct_xs.onnx",
        help="Path to OCR ONNX model (default: models/cct_xs.onnx)",
    )
    parser.add_argument(
        "--ocr-config",
        default="models/cct_xs_plate_config.yaml",
        help="Path to OCR config (default: models/cct_xs_plate_config.yaml)",
    )
    parser.add_argument("--iterations", type=int, default=50, help="Latency benchmark iterations")
    args = parser.parse_args()

    if not os.path.exists(args.candidate):
        print(f"[ERROR] Candidate model not found at: {args.candidate}")
        print("Please place your newly trained model at that path or specify --candidate <path>")
        sys.exit(1)

    if not os.path.exists(args.baseline):
        print(f"[ERROR] Baseline model not found at: {args.baseline}")
        sys.exit(1)

    print("=" * 80)
    print("EIU LPR: DETECTOR CANDIDATE BENCHMARK (NON-DESTRUCTIVE)")
    print("=" * 80)
    print(f"Baseline Model  : {args.baseline} ({os.path.getsize(args.baseline) / (1024 * 1024):.2f} MB)")
    print(f"Candidate Model : {args.candidate} ({os.path.getsize(args.candidate) / (1024 * 1024):.2f} MB)")
    print("-" * 80)

    # 1. Measure Latency
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.log_severity_level = 3

    sess_base = ort.InferenceSession(args.baseline, sess_options=sess_opt, providers=["CPUExecutionProvider"])
    sess_cand = ort.InferenceSession(args.candidate, sess_options=sess_opt, providers=["CPUExecutionProvider"])

    test_img = cv2.imread("test_images/car_test_2.jpg")
    h0, w0 = test_img.shape[:2]
    r = min(640 / h0, 640 / w0)
    nh, nw = int(h0 * r), int(w0 * r)
    can = np.full((640, 640, 3), 114, dtype=np.uint8)
    can[:nh, :nw] = cv2.resize(test_img, (nw, nh))
    blob = np.ascontiguousarray(can.transpose((2, 0, 1))[np.newaxis, ...], dtype=np.float32)

    base_mean, base_p50, base_min = benchmark_session(sess_base, blob, args.iterations)
    cand_mean, cand_p50, cand_min = benchmark_session(sess_cand, blob, args.iterations)

    print("INFERENCE LATENCY (CPU Execution Provider):")
    print(f"  Baseline  (p50 / mean): {base_p50:.1f} ms / {base_mean:.1f} ms")
    print(f"  Candidate (p50 / mean): {cand_p50:.1f} ms / {cand_mean:.1f} ms")
    diff_pct = ((base_mean - cand_mean) / base_mean) * 100
    print(f"  Latency Change        : {diff_pct:+.1f}%")
    print("-" * 80)

    # 2. Golden Dataset Evaluation
    pipe_base = LPRPipeline(detector_path=args.baseline, ocr_path=args.ocr_model, ocr_config_path=args.ocr_config)
    pipe_cand = LPRPipeline(detector_path=args.candidate, ocr_path=args.ocr_model, ocr_config_path=args.ocr_config)

    with open("golden_dataset/ground_truth.json", "r", encoding="utf-8") as f:
        gt_images = json.load(f)["images"]

    print(f"{'Image':<26} | {'Exp':<4} | {'Base Conf':<9} | {'Cand Conf':<9} | {'Base Read':<10} | {'Cand Read':<10} | Status")
    print("-" * 95)

    base_correct_reads = 0
    cand_correct_reads = 0
    base_correct_rejections = 0
    cand_correct_rejections = 0
    total_pos = 0
    total_neg = 0

    for item in gt_images:
        p = os.path.join("test_images", item["filename"])
        if not os.path.exists(p):
            continue
        img = cv2.imread(p)
        is_pos = item["expected_detection"]
        gt_text = item.get("ground_truth_normalized", "")

        res_base = pipe_base.recognize(img, color_format="BGR")
        res_cand = pipe_cand.recognize(img, color_format="BGR")

        b_conf = res_base[0].detection_score if res_base else 0.0
        c_conf = res_cand[0].detection_score if res_cand else 0.0
        b_text = res_base[0].normalized_text if res_base else "-"
        c_text = res_cand[0].normalized_text if res_cand else "-"

        status = ""
        if is_pos:
            total_pos += 1
            b_ok = (b_text == gt_text) if gt_text else (len(res_base) > 0)
            c_ok = (c_text == gt_text) if gt_text else (len(res_cand) > 0)
            if b_ok:
                base_correct_reads += 1
            if c_ok:
                cand_correct_reads += 1
            if c_ok and not b_ok:
                status = "CANDIDATE RECOVERED"
            elif b_ok and not c_ok:
                status = "CANDIDATE MISSED"
            else:
                status = "MATCH" if c_ok else "BOTH MISSED"
        else:
            total_neg += 1
            b_rej = (len(res_base) == 0)
            c_rej = (len(res_cand) == 0)
            if b_rej:
                base_correct_rejections += 1
            if c_rej:
                cand_correct_rejections += 1
            if c_rej and not b_rej:
                status = "CANDIDATE SUPPRESSED"
            elif b_rej and not c_rej:
                status = "CANDIDATE FALSE POSITIVE"
            else:
                status = "SUPPRESSED" if c_rej else "BOTH FALSE POS"

        exp_tag = "POS" if is_pos else "NEG"
        print(f"{item['filename']:<26} | {exp_tag:<4} | {b_conf:<9.2f} | {c_conf:<9.2f} | {b_text:<10} | {c_text:<10} | {status}")

    print("=" * 95)
    print("FINAL SUMMARY:")
    print(f"  Positive Plates Read Accuracy : Baseline {base_correct_reads}/{total_pos} ({base_correct_reads/total_pos*100:.1f}%) vs Candidate {cand_correct_reads}/{total_pos} ({cand_correct_reads/total_pos*100:.1f}%)")
    print(f"  Negative Background Rejection: Baseline {base_correct_rejections}/{total_neg} ({base_correct_rejections/total_neg*100:.1f}%) vs Candidate {cand_correct_rejections}/{total_neg} ({cand_correct_rejections/total_neg*100:.1f}%)")
    print("=" * 95)


if __name__ == "__main__":
    main()
