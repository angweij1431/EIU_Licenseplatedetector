# Automated regression test verifying the 20-image golden dataset
import json
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
GROUND_TRUTH_PATH = "golden_dataset/ground_truth.json"
IMAGE_DIR = "test_images"


class TestGoldenDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = LPRPipeline(
            detector_path=DETECTOR_PATH,
            ocr_path=OCR_PATH,
            ocr_config_path=OCR_CONFIG,
            detection_threshold=0.20,
        )
        cls.pipeline.warmup(iterations=1)

        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            cls.ground_truth = json.load(f)["images"]

    def test_golden_dataset_execution(self):
        passed_reads = 0
        safe_rejections = 0

        for item in self.ground_truth:
            img_path = os.path.join(IMAGE_DIR, item["filename"])
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            self.assertIsNotNone(img, f"Failed to load {img_path}")

            candidates = self.pipeline.recognize(img, color_format="BGR")

            if item["expected_detection"] and item["ground_truth_normalized"]:
                # Should find at least one candidate with matching plate
                self.assertGreaterEqual(len(candidates), 1, f"Expected detection for {item['filename']}")
                top_plate = candidates[0].normalized_text
                self.assertEqual(
                    top_plate,
                    item["ground_truth_normalized"],
                    f"Mismatch in {item['filename']}: got {top_plate}, expected {item['ground_truth_normalized']}"
                )
                passed_reads += 1
            elif not item["expected_detection"]:
                # Negative or challenging cases should either return 0 candidates or safely rejected (invalid/low-conf) candidates
                if len(candidates) == 0:
                    safe_rejections += 1
                else:
                    # If detected, all candidates must be flagged invalid or below confidence threshold
                    all_rejected = all(not cand.is_valid for cand in candidates)
                    self.assertTrue(
                        all_rejected,
                        f"False positive leak in negative sample {item['filename']}: "
                        f"detected valid candidate {[c.normalized_text for c in candidates if c.is_valid]}"
                    )
                    safe_rejections += 1

        self.assertGreaterEqual(passed_reads, 10)
        self.assertGreaterEqual(safe_rejections, 7)


if __name__ == "__main__":
    unittest.main()

