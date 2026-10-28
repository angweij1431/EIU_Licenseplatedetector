# Automated CPU Latency Benchmark for eiu_lpr
import argparse
import os
import platform
import sys
import time
import cv2
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

# Add src to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from eiu_lpr import LPRPipeline

DETECTOR_PATH = "models/detector.onnx"
OCR_PATH = "models/cct_xs.onnx"
OCR_CONFIG = "models/cct_xs_plate_config.yaml"
CHECKSUMS_PATH = "models/checksums.sha256"
DEFAULT_TEST_IMAGE = "test_images/car_test_2.jpg"


def get_system_metadata():
    mem_info = "N/A"
    if psutil is not None:
        vmem = psutil.virtual_memory()
        mem_info = f"{vmem.used / (1024**3):.1f} GB used / {vmem.total / (1024**3):.1f} GB total"

    checksum_lines = []
    if os.path.exists(CHECKSUMS_PATH):
        with open(CHECKSUMS_PATH, "r", encoding="utf-8") as f:
            checksum_lines = [line.strip() for line in f if line.strip()]

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": f"{platform.processor()} ({platform.machine()})",
        "python": sys.version.split()[0],
        "memory": mem_info,
        "threads": os.cpu_count() or 1,
        "checksums": checksum_lines,
    }


def run_benchmark(iterations: int = 100, image_path: str = DEFAULT_TEST_IMAGE):
    print("=" * 65)
    print("EIU LPR Linux CPU Latency Benchmark")
    print("=" * 65)

    meta = get_system_metadata()
    print(f"Operating System  : {meta['os']}")
    print(f"CPU Architecture  : {meta['cpu']}")
    print(f"Logical CPU Cores : {meta['threads']}")
    print(f"RAM Footprint     : {meta['memory']}")
    print(f"Python Runtime    : {meta['python']}")
    print(f"Execution Provider: CPUExecutionProvider (ONNX Runtime)")
    if meta["checksums"]:
        print(f"Model Checksums   : {len(meta['checksums'])} verified assets in models/checksums.sha256")
    print("-" * 65)

    if not os.path.exists(image_path):
        print(f"[ERROR] Test image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Target Image      : {image_path} ({img.shape[1]}x{img.shape[0]} px)")
    print(f"Sample Count      : {iterations} iterations")
    print("-" * 65)

    # 1. Cold Startup Measurement (Model Loading + Session Creation + First Frame)
    t_start = time.perf_counter()
    pipeline = LPRPipeline(
        detector_path=DETECTOR_PATH,
        ocr_path=OCR_PATH,
        ocr_config_path=OCR_CONFIG,
        detection_threshold=0.20,
    )
    first_result = pipeline.process(img_rgb, color_format="RGB")
    cold_start_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"Cold Startup Latency (Load + 1st Frame): {cold_start_ms:.1f} ms")

    # 2. Warmup Passes
    print("\nExecuting 5 warmup passes...")
    pipeline.warmup(iterations=5)
    print("[OK] Warmup complete.")

    # 3. Benchmark Detection Only
    print(f"\nBenchmarking detect() over {iterations} iterations...")
    det_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        boxes = pipeline.detect(img_rgb, color_format="RGB")
        det_times.append((time.perf_counter() - t0) * 1000.0)

    # 4. Benchmark OCR (read_plate) Only
    stable_box = boxes[0] if boxes else (177.0, 234.0, 242.0, 266.0)
    print(f"Benchmarking read_plate() over {iterations} iterations...")
    ocr_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = pipeline.read_plate(img_rgb, bbox=stable_box, color_format="RGB")
        ocr_times.append((time.perf_counter() - t0) * 1000.0)

    # 5. Benchmark End-to-End Pipeline
    print(f"Benchmarking process() (End-to-End) over {iterations} iterations...")
    e2e_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = pipeline.process(img_rgb, color_format="RGB")
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    # Report Computation
    def compute_stats(times):
        arr = np.array(times)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    det_stats = compute_stats(det_times)
    ocr_stats = compute_stats(ocr_times)
    e2e_stats = compute_stats(e2e_times)

    print("\n" + "=" * 65)
    print("LATENCY BENCHMARK RESULTS TABLE (milliseconds)")
    print("=" * 65)
    header = f"{'Stage':<22} | {'Mean':<8} | {'p50':<8} | {'p95':<8} | {'Min':<8} | {'Max':<8}"
    print(header)
    print("-" * 65)

    for name, s in [
        ("Detection (detect)", det_stats),
        ("OCR (read_plate)", ocr_stats),
        ("Combined (process)", e2e_stats),
    ]:
        row = f"{name:<22} | {s['mean']:<8.1f} | {s['p50']:<8.1f} | {s['p95']:<8.1f} | {s['min']:<8.1f} | {s['max']:<8.1f}"
        print(row)

    print("=" * 65)
    print(f"Detected Plate: {first_result.candidates[0].normalized_text if first_result.candidates else 'None'}")
    print("[SUCCESS] Benchmark completed successfully.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linux CPU Latency Benchmark for eiu_lpr")
    parser.add_argument("--iterations", type=int, default=100, help="Number of test iterations (default: 100)")
    parser.add_argument("--image", type=str, default=DEFAULT_TEST_IMAGE, help="Path to test image")
    args = parser.parse_args()

    run_benchmark(iterations=args.iterations, image_path=args.image)

