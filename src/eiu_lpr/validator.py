# Vietnamese License Plate Syntax Validator (Circular 24/2023/TT-BCA)
import re
from typing import Dict, Any


class VietnamPlateValidator:
    # Automobiles (e.g. 30E69770, 29M6840, 51G74372): 2 digits + 1 letter + 4-6 digits
    AUTO_PATTERN = re.compile(r'^[1-9][0-9][A-Z][0-9]{4,6}$')

    # Motorcycles: 2 digits + (1 letter + 1 digit, or 2 letters per Circular 24/2023) + 4-6 digits
    # Examples: 59D242069, 63H90394, 77C144470, 59VA24003
    MOTO_PATTERN = re.compile(r'^[1-9][0-9]([A-Z][0-9]|[A-Z]{2})[0-9]{4,6}$')

    # Normalization: uppercase, remove non-alphanumerics, keep order, no ambiguous substitution
    @staticmethod
    def normalize_text(raw_text: str) -> str:
        if not raw_text:
            return ""
        return re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    # Check plate against official regex standards
    @classmethod
    def validate(cls, raw_text: str) -> Dict[str, Any]:
        normalized = cls.normalize_text(raw_text)
        if not normalized:
            return {
                "raw_text": raw_text,
                "normalized_text": "",
                "is_valid": False,
                "vehicle_type": "unknown",
            }

        is_auto = bool(cls.AUTO_PATTERN.match(normalized))
        is_moto = bool(cls.MOTO_PATTERN.match(normalized))
        is_valid = is_auto or is_moto

        vehicle_type = "unknown"
        if is_auto:
            vehicle_type = "automobile"
        elif is_moto:
            vehicle_type = "motorcycle"

        return {
            "raw_text": raw_text,
            "normalized_text": normalized,
            "is_valid": is_valid,
            "vehicle_type": vehicle_type,
        }
