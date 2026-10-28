# In-memory CCT-XS Vision Transformer OCR Recognizer Adapter
import os
from typing import Any, Dict, List
import cv2
import numpy as np

from .exceptions import InvalidInputError, ModelInferenceError, ModelNotFoundError
from .models import CharacterDetail


# Resolves relative path against current working directory or repo root
def _resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return os.path.abspath(path)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(pkg_dir, "..", "..", path))
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(path)


class CCTXSRecognizer:
    # OCR adapter wrapping CCT-XS Compact Convolutional Transformer
    def __init__(
        self,
        model_path: str,
        config_path: str,
    ):
        resolved_model = _resolve_model_path(model_path)
        resolved_config = _resolve_model_path(config_path)

        if not os.path.exists(resolved_model):
            raise ModelNotFoundError(f"CCT-XS model weights not found: {model_path}")
        if not os.path.exists(resolved_config):
            raise ModelNotFoundError(f"CCT-XS plate config not found: {config_path}")

        self.model_path = resolved_model
        self.config_path = resolved_config

        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self.recognizer = LicensePlateRecognizer(
                onnx_model_path=self.model_path,
                plate_config_path=self.config_path,
                device="cpu",
            )
        except Exception as e:
            raise ModelInferenceError(f"Failed to initialize CCT-XS OCR engine: {e}") from e

    # Runs Optical Character Recognition on an in-memory image crop
    def recognize_crop(
        self,
        crop: np.ndarray,
        color_format: str = "RGB",
    ) -> Dict[str, Any]:
        if crop is None or not isinstance(crop, np.ndarray):
            raise InvalidInputError("Crop must be a non-null numpy.ndarray.")
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return {
                "raw_text": "",
                "ocr_confidence": 0.0,
                "ocr_min_confidence": 0.0,
                "character_details": [],
                "region": "Unknown",
                "region_prob": 0.0,
            }
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise InvalidInputError(f"Expected 3-channel crop [H, W, 3], received shape {crop.shape}.")
        if crop.dtype != np.uint8:
            raise InvalidInputError(f"Expected uint8 crop, received dtype {crop.dtype}.")

        # CCT-XS requires RGB input
        if color_format.upper() == "BGR":
            work_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        elif color_format.upper() == "RGB":
            work_crop = crop
        else:
            raise InvalidInputError(f"Unsupported color_format '{color_format}'. Must be 'RGB' or 'BGR'.")

        try:
            preds = self.recognizer.run(work_crop, return_confidence=True)
        except Exception as e:
            raise ModelInferenceError(f"CCT-XS OCR inference execution failed: {e}") from e

        if not preds or not preds[0].plate:
            return {
                "raw_text": "",
                "ocr_confidence": 0.0,
                "ocr_min_confidence": 0.0,
                "character_details": [],
                "region": "Unknown",
                "region_prob": 0.0,
            }

        pred = preds[0]
        raw_text = pred.plate
        plate_len = len(raw_text)

        char_details: List[CharacterDetail] = []
        if pred.char_probs is not None and plate_len > 0:
            plate_char_probs = pred.char_probs[:plate_len]
            mean_conf = float(np.mean(plate_char_probs))
            min_conf = float(np.min(plate_char_probs))
            char_details = [
                CharacterDetail(char=ch, confidence=float(p))
                for ch, p in zip(raw_text, plate_char_probs)
            ]
        else:
            mean_conf = 0.0
            min_conf = 0.0

        region = getattr(pred, "region", "Unknown") or "Unknown"
        region_prob = float(getattr(pred, "region_prob", 0.0) or 0.0)

        return {
            "raw_text": raw_text,
            "ocr_confidence": mean_conf,
            "ocr_min_confidence": min_conf,
            "character_details": char_details,
            "region": region,
            "region_prob": region_prob,
        }

    # Primes OCR session with dummy crop tensor
    def warmup(self, iterations: int = 2) -> None:
        dummy_crop = np.zeros((64, 128, 3), dtype=np.uint8)
        for _ in range(max(1, iterations)):
            self.recognize_crop(dummy_crop, color_format="RGB")
