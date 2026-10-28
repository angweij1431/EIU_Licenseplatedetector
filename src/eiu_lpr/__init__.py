# EIU License Plate Recognition Library (Production Vision Engine)

from .exceptions import (
    ConfigurationError,
    EIULPRError,
    InvalidInputError,
    ModelInferenceError,
    ModelNotFoundError,
)
from .models import (
    DEFAULT_MODEL_VERSION,
    SCHEMA_VERSION,
    BoundingBox,
    CharacterDetail,
    PipelineResult,
    PlateCandidate,
)
from .pipeline import LPRPipeline
from .detector import ONNXYoloDetector
from .recognizer import CCTXSRecognizer
from .preprocessor import PlatePreprocessor
from .validator import VietnamPlateValidator

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "DEFAULT_MODEL_VERSION",
    "LPRPipeline",
    "ONNXYoloDetector",
    "CCTXSRecognizer",
    "PlatePreprocessor",
    "VietnamPlateValidator",
    "BoundingBox",
    "CharacterDetail",
    "PlateCandidate",
    "PipelineResult",
    "EIULPRError",
    "InvalidInputError",
    "ModelNotFoundError",
    "ModelInferenceError",
    "ConfigurationError",
]
