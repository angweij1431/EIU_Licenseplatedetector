# Core ALPR Pipeline orchestrating detect, read_plate, recognize, and warmup
import time
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from .detector import ONNXYoloDetector
from .exceptions import ConfigurationError, InvalidInputError
from .models import (
    DEFAULT_MODEL_VERSION,
    SCHEMA_VERSION,
    BoundingBox,
    PlateCandidate,
    PipelineResult,
)
from .preprocessor import PlatePreprocessor
from .recognizer import CCTXSRecognizer
from .validator import VietnamPlateValidator


class LPRPipeline:
    # In-memory ALPR pipeline with pure ONNX Runtime inference
    def __init__(
        self,
        detector_path: str,
        ocr_path: str,
        ocr_config_path: str,
        detection_threshold: float = 0.20,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 10,
        ocr_mean_threshold: float = 0.70,
        ocr_min_threshold: float = 0.50,
        enable_adaptive_enhancement: bool = True,
        intra_op_threads: Optional[int] = None,
        model_version: str = DEFAULT_MODEL_VERSION,
        schema_version: str = SCHEMA_VERSION,
    ):
        if not (0.0 <= detection_threshold <= 1.0):
            raise ConfigurationError(f"detection_threshold must be in [0.0, 1.0], got {detection_threshold}")
        if not (0.0 <= nms_iou_threshold <= 1.0):
            raise ConfigurationError(f"nms_iou_threshold must be in [0.0, 1.0], got {nms_iou_threshold}")
        if max_detections <= 0:
            raise ConfigurationError(f"max_detections must be a positive integer, got {max_detections}")

        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections = max_detections
        self.ocr_mean_threshold = ocr_mean_threshold
        self.ocr_min_threshold = ocr_min_threshold
        self.enable_adaptive_enhancement = enable_adaptive_enhancement
        self.model_version = model_version
        self.schema_version = schema_version

        self.detector = ONNXYoloDetector(
            model_path=detector_path,
            confidence_threshold=detection_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            intra_op_threads=intra_op_threads,
        )

        self.recognizer = CCTXSRecognizer(
            model_path=ocr_path,
            config_path=ocr_config_path,
        )

    # Detects plate bounding boxes
    def detect(
        self,
        image: np.ndarray,
        color_format: str = "RGB",
        detection_threshold: Optional[float] = None,
    ) -> List[BoundingBox]:
        return self.detector.detect(
            image,
            color_format=color_format,
            confidence_threshold=detection_threshold,
        )

    # Crops, enhances, and reads a single plate region from a bounding box
    def read_plate(
        self,
        image: np.ndarray,
        bbox: Union[BoundingBox, Tuple[float, float, float, float], List[float]],
        color_format: str = "RGB",
        include_debug_crop: bool = False,
    ) -> PlateCandidate:
        if isinstance(bbox, BoundingBox):
            box_coords = bbox.to_xyxy()
            det_score = bbox.confidence
        elif isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            box_coords = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            det_score = 1.0
        else:
            raise InvalidInputError(f"Invalid bbox format: expected BoundingBox or 4-element sequence, got {bbox}.")

        # Crop plate with dynamic 4% padding (3px to 12px)
        crop = PlatePreprocessor.crop_plate(
            image=image,
            bbox=box_coords,
            padding_percent=0.04,
            min_pad_px=3,
            max_pad_px=12,
        )

        if crop.size == 0:
            return PlateCandidate(
                bbox=box_coords,
                detection_score=det_score,
                raw_text="",
                normalized_text="",
                is_valid=False,
                ocr_confidence=0.0,
                ocr_min_confidence=0.0,
                character_details=[],
                region="Unknown",
                region_prob=0.0,
                pass_level="empty_crop",
                debug_crop=crop if include_debug_crop else None,
            )

        # Mild adaptive enhancement for blurry or low-res crops
        pass_level = "base_640px"
        work_crop = crop
        if self.enable_adaptive_enhancement:
            crop_h, crop_w = crop.shape[:2]
            blur_score = PlatePreprocessor.compute_blur_score(crop)
            if crop_w < 110 or crop_h < 60 or blur_score < 150.0:
                work_crop = PlatePreprocessor.mild_enhance_for_transformer(crop)
                pass_level = "enhanced_crop"

        # Optical character recognition
        ocr_result = self.recognizer.recognize_crop(work_crop, color_format=color_format)
        raw_text = ocr_result["raw_text"]

        # Circular 24/2023 normalization & validation
        validation = VietnamPlateValidator.validate(raw_text)

        return PlateCandidate(
            bbox=box_coords,
            detection_score=det_score,
            raw_text=raw_text,
            normalized_text=validation["normalized_text"],
            is_valid=validation["is_valid"],
            ocr_confidence=ocr_result["ocr_confidence"],
            ocr_min_confidence=ocr_result["ocr_min_confidence"],
            character_details=ocr_result["character_details"],
            region=ocr_result["region"],
            region_prob=ocr_result["region_prob"],
            pass_level=pass_level,
            debug_crop=work_crop if include_debug_crop else None,
        )

    # Convenience pipeline: runs detection then reads all detected plates
    def recognize(
        self,
        image: np.ndarray,
        color_format: str = "RGB",
        detection_threshold: Optional[float] = None,
        include_debug_crop: bool = False,
    ) -> List[PlateCandidate]:
        boxes = self.detect(
            image,
            color_format=color_format,
            detection_threshold=detection_threshold,
        )
        candidates: List[PlateCandidate] = []
        for box in boxes:
            cand = self.read_plate(
                image=image,
                bbox=box,
                color_format=color_format,
                include_debug_crop=include_debug_crop,
            )
            candidates.append(cand)
        return candidates

    # Generates in-memory annotated visualization with bounding boxes and text labels
    def annotate(
        self,
        image: np.ndarray,
        candidates: Union[List[PlateCandidate], PipelineResult],
    ) -> np.ndarray:
        PlatePreprocessor.validate_image(image)
        cands = candidates.candidates if isinstance(candidates, PipelineResult) else candidates
        annotated = image.copy()

        for cand in cands:
            x1, y1, x2, y2 = [int(round(c)) for c in cand.bbox]
            color = (0, 255, 0) if cand.is_valid else (0, 165, 255)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{cand.normalized_text} ({cand.ocr_confidence * 100:.0f}%)" if cand.normalized_text else "Plate"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = y1
            cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 4, label_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 1),
                font,
                font_scale,
                (0, 0, 0),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated

    # High-level processing method returning timed, structured PipelineResult
    def process(
        self,
        image: np.ndarray,
        color_format: str = "RGB",
        detection_threshold: Optional[float] = None,
        request_id: Optional[str] = None,
        include_debug_crop: bool = False,
        include_annotated_image: bool = False,
    ) -> PipelineResult:
        t0 = time.perf_counter()
        candidates = self.recognize(
            image=image,
            color_format=color_format,
            detection_threshold=detection_threshold,
            include_debug_crop=include_debug_crop,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        annotated_img = None
        if include_annotated_image:
            annotated_img = self.annotate(image, candidates)

        return PipelineResult(
            candidates=candidates,
            processing_time_ms=elapsed_ms,
            status="OK",
            schema_version=self.schema_version,
            model_version=self.model_version,
            request_id=request_id,
            annotated_image=annotated_img,
        )

    # Primes detector and OCR ONNX sessions with dummy inference passes
    def warmup(self, iterations: int = 2) -> None:
        self.detector.warmup(iterations=iterations)
        self.recognizer.warmup(iterations=iterations)
