# Tests verifying session warmup and cold vs warm execution
import os
import sys
import time
import unittest
import cv2

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

from eiu_lpr import LPRPipeline

DETECTOR_PATH = "models/detector.onnx"
OCR_PATH = "models/cct_xs.onnx"
OCR_CONFIG = "models/cct_xs_plate_config.yaml"
TEST_IMG_PATH = "test_images/car_test_2.jpg"


class TestWarmup(unittest.TestCase):
    def test_session_warmup(self):
        pipeline = LPRPipeline(
            detector_path=DETECTOR_PATH,
            ocr_path=OCR_PATH,
            ocr_config_path=OCR_CONFIG,
        )
        pipeline.warmup(iterations=2)

        img = cv2.imread(TEST_IMG_PATH)
        t0 = time.perf_counter()
        candidates = pipeline.recognize(img, color_format="BGR")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertGreaterEqual(len(candidates), 1)
        # Warm inference on modern CPU should be fast (typically < 500ms)
        self.assertLess(elapsed_ms, 1000.0)


if __name__ == "__main__":
    unittest.main()
