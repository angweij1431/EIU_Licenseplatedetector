# Automated verification test suite for eiu_lpr
import os
import sys
import unittest
import cv2
import numpy as np

# Add src to pythonpath
sys.path.insert(0, os.path.abspath("src"))

from eiu_lpr import (
    BoundingBox,
    ConfigurationError,
    InvalidInputError,
    LPRPipeline,
    ModelNotFoundError,
    PlateCandidate,
)

DETECTOR_PATH = "models/detector.onnx"
OCR_PATH = "models/cct_xs.onnx"
OCR_CONFIG = "models/cct_xs_plate_config.yaml"
TEST_IMG = "test_images/car_test_2.jpg"


class TestEIULPR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = LPRPipeline(
            detector_path=DETECTOR_PATH,
            ocr_path=OCR_PATH,
            ocr_config_path=OCR_CONFIG,
            detection_threshold=0.20,
        )

    def test_01_rgb_in_memory_inference(self):
        """01: In-memory RGB array inference (reads 30A69770, valid=True, conf>0.90)"""
        img_bgr = cv2.imread(TEST_IMG)
        self.assertIsNotNone(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        candidates = self.pipeline.recognize(img_rgb, color_format="RGB")
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0].normalized_text, "30A69770")
        self.assertTrue(candidates[0].is_valid)
        self.assertGreater(candidates[0].ocr_confidence, 0.90)

    def test_02_bgr_color_format(self):
        """02: BGR color format support (seamless conversion from OpenCV)"""
        img_bgr = cv2.imread(TEST_IMG)
        candidates = self.pipeline.recognize(img_bgr, color_format="BGR")
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0].normalized_text, "30A69770")

    def test_03_zero_disk_io(self):
        """03: Zero disk I/O verification (0 new files created during inference)"""
        img = cv2.imread(TEST_IMG)
        files_before = set(os.listdir("."))
        for _ in range(3):
            _ = self.pipeline.recognize(img, color_format="BGR")
            _ = self.pipeline.process(img, color_format="BGR")
        files_after = set(os.listdir("."))
        self.assertEqual(files_before, files_after)

    def test_04_split_apis_consistency(self):
        """04: Split APIs consistency (detect() + read_plate() matches recognize())"""
        img = cv2.imread(TEST_IMG)
        combined = self.pipeline.recognize(img, color_format="BGR")
        boxes = self.pipeline.detect(img, color_format="BGR")
        self.assertEqual(len(boxes), len(combined))

        for b, comb in zip(boxes, combined):
            split_cand = self.pipeline.read_plate(img, b, color_format="BGR")
            self.assertEqual(split_cand.normalized_text, comb.normalized_text)
            self.assertEqual(split_cand.raw_text, comb.raw_text)
            self.assertAlmostEqual(split_cand.ocr_confidence, comb.ocr_confidence, places=4)

    def test_05_pure_onnx_no_ultralytics(self):
        """05: Pure ONNX runtime verification (ultralytics is not in sys.modules)"""
        self.assertNotIn("ultralytics", sys.modules)

    def test_06_invalid_input_exceptions(self):
        """06: Error handling (invalid shapes/dtypes raise InvalidInputError)"""
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(None)
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(np.zeros((50, 50), dtype=np.uint8))
        with self.assertRaises(InvalidInputError):
            self.pipeline.detect(np.zeros((50, 50, 3), dtype=np.float32))

    def test_07_invalid_config_exceptions(self):
        """07: Configuration safety (bad paths or thresholds raise custom errors)"""
        with self.assertRaises(ModelNotFoundError):
            LPRPipeline("bad_path_xyz.onnx", OCR_PATH, OCR_CONFIG)
        with self.assertRaises(ConfigurationError):
            LPRPipeline(DETECTOR_PATH, OCR_PATH, OCR_CONFIG, detection_threshold=2.0)

    def test_08_warmup(self):
        """08: Session warmup (primes execution providers for low latency)"""
        self.pipeline.warmup(iterations=2)
        res = self.pipeline.process(cv2.imread(TEST_IMG), color_format="BGR")
        self.assertEqual(res.status, "OK")
        self.assertGreaterEqual(len(res.candidates), 1)

    def test_09_custom_versioning(self):
        """09: Telemetry versioning (custom model_version and schema_version forwarded)"""
        custom_pipe = LPRPipeline(
            detector_path=DETECTOR_PATH,
            ocr_path=OCR_PATH,
            ocr_config_path=OCR_CONFIG,
            model_version="eiu-yolo11n-v2.0",
            schema_version="1.1",
        )
        res = custom_pipe.process(cv2.imread(TEST_IMG), color_format="BGR")
        self.assertEqual(res.model_version, "eiu-yolo11n-v2.0")
        self.assertEqual(res.schema_version, "1.1")
        self.assertEqual(res.to_dict()["model_version"], "eiu-yolo11n-v2.0")
        self.assertEqual(res.to_dict()["schema_version"], "1.1")

    def test_10_threshold_override(self):
        """10: Dynamic threshold override (per-call detection_threshold works)"""
        img = cv2.imread(TEST_IMG)
        boxes_default = self.pipeline.detect(img, color_format="BGR")
        self.assertGreaterEqual(len(boxes_default), 1)

        boxes_strict = self.pipeline.detect(img, color_format="BGR", detection_threshold=0.999)
        self.assertEqual(len(boxes_strict), 0)

    def test_11_in_memory_debug_images(self):
        """11: In-memory debug crops & annotated frame (retrievable on request without disk I/O)"""
        img = cv2.imread(TEST_IMG)
        result = self.pipeline.process(
            img,
            color_format="BGR",
            include_debug_crop=True,
            include_annotated_image=True,
        )
        self.assertIsNotNone(result.annotated_image)
        self.assertIsInstance(result.annotated_image, np.ndarray)
        self.assertEqual(result.annotated_image.shape, img.shape)

        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertIsNotNone(result.candidates[0].debug_crop)
        self.assertIsInstance(result.candidates[0].debug_crop, np.ndarray)

        # Confirm direct pipeline.annotate method works
        annotated = self.pipeline.annotate(img, result.candidates)
        self.assertEqual(annotated.shape, img.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
