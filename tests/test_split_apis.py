# Tests verifying split detect() and read_plate() API behavior and consistency
import os
import sys
import unittest
import cv2

# Ensure src is in python path
sys.path.insert(0, os.path.abspath("src"))

from eiu_lpr import BoundingBox, LPRPipeline, PlateCandidate

DETECTOR_PATH = "models/detector.onnx"
OCR_PATH = "models/cct_xs.onnx"
OCR_CONFIG = "models/cct_xs_plate_config.yaml"
TEST_IMG_PATH = "test_images/car_test_2.jpg"


class TestSplitAPIs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = LPRPipeline(
            detector_path=DETECTOR_PATH,
            ocr_path=OCR_PATH,
            ocr_config_path=OCR_CONFIG,
            detection_threshold=0.20,
        )

    def test_split_apis_consistency(self):
        """Verifies that recognize() produces identical candidates to detect() + read_plate()."""
        img_bgr = cv2.imread(TEST_IMG_PATH)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Combined call
        combined_candidates = self.pipeline.recognize(img_rgb)
        self.assertGreaterEqual(len(combined_candidates), 1)

        # 2. Split call
        boxes = self.pipeline.detect(img_rgb)
        self.assertEqual(len(boxes), len(combined_candidates))

        split_candidates = []
        for b in boxes:
            cand = self.pipeline.read_plate(img_rgb, b)
            split_candidates.append(cand)

        # Compare results
        for c_comb, c_split in zip(combined_candidates, split_candidates):
            self.assertEqual(c_comb.normalized_text, c_split.normalized_text)
            self.assertEqual(c_comb.raw_text, c_split.raw_text)
            self.assertEqual(c_comb.is_valid, c_split.is_valid)
            self.assertAlmostEqual(c_comb.ocr_confidence, c_split.ocr_confidence, places=4)
            self.assertAlmostEqual(c_comb.detection_score, c_split.detection_score, places=4)

    def test_read_plate_with_debug_crop(self):
        """Verifies that include_debug_crop returns the cropped ndarray."""
        img_bgr = cv2.imread(TEST_IMG_PATH)
        boxes = self.pipeline.detect(img_bgr, color_format="BGR")
        self.assertGreaterEqual(len(boxes), 1)

        cand = self.pipeline.read_plate(
            img_bgr,
            boxes[0],
            color_format="BGR",
            include_debug_crop=True,
        )
        self.assertIsNotNone(cand.debug_crop)
        self.assertEqual(len(cand.debug_crop.shape), 3)

    def test_in_memory_annotated_image(self):
        """Verifies that in-memory annotated visualizations work without disk writes."""
        img_bgr = cv2.imread(TEST_IMG_PATH)
        result = self.pipeline.process(
            img_bgr,
            color_format="BGR",
            include_debug_crop=True,
            include_annotated_image=True,
        )
        self.assertIsNotNone(result.annotated_image)
        self.assertEqual(result.annotated_image.shape, img_bgr.shape)
        self.assertIsNotNone(result.candidates[0].debug_crop)

        # Also test direct pipeline.annotate call
        annotated_direct = self.pipeline.annotate(img_bgr, result)
        self.assertEqual(annotated_direct.shape, img_bgr.shape)


if __name__ == "__main__":
    unittest.main()
