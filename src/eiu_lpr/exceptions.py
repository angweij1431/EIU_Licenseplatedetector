# Custom typed exceptions for eiu_lpr

class EIULPRError(Exception):
    # Base exception for all errors raised by eiu_lpr
    pass


class InvalidInputError(EIULPRError):
    # Raised when input image array or bounding box format is invalid
    pass


class ModelNotFoundError(EIULPRError):
    # Raised when model weights or YAML configs are missing
    pass


class ModelInferenceError(EIULPRError):
    # Raised when ONNX Runtime session execution fails
    pass


class ConfigurationError(EIULPRError):
    # Raised when threshold or configuration parameters are invalid
    pass
