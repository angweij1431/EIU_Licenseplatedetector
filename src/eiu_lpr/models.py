# Data models for eiu_lpr candidates, detections, and pipeline outputs
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Default schema and model version identifiers
SCHEMA_VERSION = "1.0"
DEFAULT_MODEL_VERSION = "alpr-v0.1"


@dataclass(frozen=True)
class BoundingBox:
    # Detected bounding box in original image pixel coordinates
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str = "plate"

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_int_xyxy(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class CharacterDetail:
    # Individual character and its softmax probability from OCR
    char: str
    confidence: float


@dataclass
class PlateCandidate:
    # Structured candidate result
    bbox: Tuple[float, float, float, float]
    detection_score: float
    raw_text: str
    normalized_text: str
    is_valid: bool
    ocr_confidence: float
    ocr_min_confidence: float
    character_details: List[CharacterDetail] = field(default_factory=list)
    region: str = "Vietnam"
    region_prob: float = 0.0
    pass_level: str = "base_640px"
    debug_crop: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        # Protects vehicle PII in console/telemetry logs by masking plate text.
        # Full plate text remains accessible via self.normalized_text, self.raw_text, and self.to_dict().
        if len(self.normalized_text) >= 4:
            masked = f"{self.normalized_text[:3]}****{self.normalized_text[-1:]}"
        elif self.normalized_text:
            masked = "***"
        else:
            masked = ""
        box = [int(round(c)) for c in self.bbox]
        return (
            f"PlateCandidate(bbox={box}, plate='{masked}', "
            f"conf={self.ocr_confidence:.2f}, is_valid={self.is_valid})"
        )

    # Convert candidate to dictionary matching CerberusEdge JSON schema
    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "plate": self.normalized_text,
            "raw_plate": self.raw_text,
            "confidence": round(self.ocr_confidence, 4),
            "detection_confidence": round(self.detection_score, 4),
            "bbox": [int(round(coord)) for coord in self.bbox],
        }
        if not minimal:
            data["is_valid"] = self.is_valid
            data["ocr_min_confidence"] = round(self.ocr_min_confidence, 4)
            data["character_details"] = [
                {"char": cd.char, "confidence": round(cd.confidence, 4)}
                for cd in self.character_details
            ]
            data["region"] = self.region
            data["region_prob"] = round(self.region_prob, 4)
            data["pass_level"] = self.pass_level
        return data


@dataclass
class PipelineResult:
    # High-level pipeline result container matching CerberusEdge schema
    candidates: List[PlateCandidate]
    processing_time_ms: float
    status: str = "OK"
    schema_version: str = SCHEMA_VERSION
    model_version: str = DEFAULT_MODEL_VERSION
    request_id: Optional[str] = None
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_dict(self, minimal: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": self.schema_version,
        }
        if self.request_id is not None:
            data["request_id"] = self.request_id
        data["status"] = self.status
        data["candidates"] = [c.to_dict(minimal=minimal) for c in self.candidates]
        data["model_version"] = self.model_version
        data["processing_time_ms"] = round(self.processing_time_ms, 2)
        return data
