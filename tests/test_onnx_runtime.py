# Tests verifying pure ONNX Runtime execution without ultralytics imports
import os
import sys
import unittest
import cv2

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

from eiu_lpr import LPRPipeline

DETECTOR_PATH = "models/detector.onnx"
OCR_PATH = "models/cct_xs.onnx"
OCR_CONFIG = "models/cct_xs_plate_config.yaml"
TEST_IMG_PATH = "test_images/car_test_2.jpg"


class TestONNXRuntime(unittest.TestCase):
    def test_ultralytics_not_imported(self):
        # Ensure ultralytics is never imported into runtime
        if "ultralytics" in sys.modules:
            del sys.modules["ultralytics"]

        pipeline = LPRPipeline(
            detector_path=DETECTOR_PATH,
            ocr_path=OCR_PATH,
            ocr_config_path=OCR_CONFIG,
        )

        img = cv2.imread(TEST_IMG_PATH)
        pipeline.recognize(img, color_format="BGR")

        self.assertNotIn("ultralytics", sys.modules)


if __name__ == "__main__":
    unittest.main()
