# Tests verifying in-memory NumPy API, zero disk I/O, and zero stdout prints
import io
import os
import sys
import unittest
import cv2
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

from eiu_lpr import (
    ConfigurationError,
    InvalidInputError,
    LPRPipeline,
    ModelNotFoundError,
    PlateCandidate,
)

DETECTOR_PATH = "models/detector.onnx"
OCR_PATH = "models/cct_xs.onnx"
OCR_CONFIG = "models/cct_xs_plate_config.yaml"
TEST_IMG_PATH = "test_images/car_test_2.jpg"


class TestInMemoryAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = LPRPipeline(
            detector_path=DETECTOR_PATH,
            ocr_path=OCR_PATH,
            ocr_config_path=OCR_CONFIG,
            detection_threshold=0.20,
        )

    def test_in_memory_rgb_input(self):
        img_bgr = cv2.imread(TEST_IMG_PATH)
        self.assertIsNotNone(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        candidates = self.pipeline.recognize(img_rgb, color_format="RGB")
        self.assertIsInstance(candidates, list)
        self.assertGreaterEqual(len(candidates), 1)

        cand = candidates[0]
        self.assertIsInstance(cand, PlateCandidate)
        self.assertEqual(cand.normalized_text, "30A69770")
        self.assertTrue(cand.is_valid)
        self.assertGreater(cand.ocr_confidence, 0.90)

    def test_bgr_color_format_support(self):
        img_bgr = cv2.imread(TEST_IMG_PATH)
        candidates = self.pipeline.recognize(img_bgr, color_format="BGR")
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0].normalized_text, "30A69770")

    def test_zero_disk_io(self):
        img_bgr = cv2.imread(TEST_IMG_PATH)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        files_before = set(os.listdir("."))

        for _ in range(5):
            _ = self.pipeline.recognize(img_rgb)
            _ = self.pipeline.process(img_rgb)

        files_after = set(os.listdir("."))
        self.assertEqual(files_before, files_after)

    def test_zero_stdout_stderr_prints(self):
        img_bgr = cv2.imread(TEST_IMG_PATH)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            _ = self.pipeline.recognize(img_rgb)
            _ = self.pipeline.process(img_rgb)
            out = sys.stdout.getvalue()
            err = sys.stderr.getvalue()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.assertEqual(out, "")
        self.assertEqual(err, "")

    def test_invalid_inputs_raise_typed_exceptions(self):
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(None)
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(np.zeros((100, 100), dtype=np.uint8))
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(np.zeros((1, 100, 100, 3), dtype=np.uint8))
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(np.zeros((100, 100, 3), dtype=np.float32))
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(np.zeros((100, 100, 3), dtype=np.uint8), color_format="RGBA")

    def test_invalid_config_raises_typed_exceptions(self):
        with self.assertRaises(ModelNotFoundError):
            LPRPipeline(
                detector_path="nonexistent.onnx",
                ocr_path=OCR_PATH,
                ocr_config_path=OCR_CONFIG,
            )
        with self.assertRaises(ConfigurationError):
            LPRPipeline(
                detector_path=DETECTOR_PATH,
                ocr_path=OCR_PATH,
                ocr_config_path=OCR_CONFIG,
                detection_threshold=1.5,
            )


if __name__ == "__main__":
    unittest.main()
