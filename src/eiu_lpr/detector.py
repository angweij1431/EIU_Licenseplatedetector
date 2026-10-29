# Direct ONNX Runtime YOLO Detector with pure NumPy preprocessing and NMS
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from .exceptions import InvalidInputError, ModelInferenceError, ModelNotFoundError
from .models import BoundingBox


# Resolves relative path against current working directory or repo root
def _resolve_model_path(path: str) -> str:
    if os.path.exists(path):
        return os.path.abspath(path)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(pkg_dir, "..", "..", path))
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(path)


class ONNXYoloDetector:
    # Pure ONNX Runtime YOLO object detector without ultralytics imports
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.20,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 10,
        input_size: Tuple[int, int] = (640, 640),
        intra_op_threads: Optional[int] = None,
    ):
        resolved_path = _resolve_model_path(model_path)
        if not os.path.exists(resolved_path):
            raise ModelNotFoundError(f"Detector weights file not found: {model_path}")

        self.model_path = resolved_path
        self.confidence_threshold = float(confidence_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_detections = int(max_detections)
        self.input_size = input_size

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        if intra_op_threads is not None and intra_op_threads > 0:
            sess_options.intra_op_num_threads = intra_op_threads

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise ModelInferenceError(f"Failed to initialize ONNX session for {model_path}: {e}") from e

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    # Resizes and letterboxes image to target resolution while preserving aspect ratio 
    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        fill_value: int = 114,
    ) -> Tuple[np.ndarray, float, float, float]:
        h, w = image.shape[:2]
        tw, th = target_size
        scale = min(th / h, tw / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_w = (tw - nw) / 2.0
        pad_h = (th - nh) / 2.0
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(fill_value, fill_value, fill_value)
        )
        return canvas, scale, float(left), float(top)

    # Pure NumPy vectorized Non-Maximum Suppression
    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_detections: int = 10) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0 and len(keep) < max_detections:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    # Detects plate bounding boxes on an in-memory frame
    def detect(
        self,
        image: np.ndarray,
        color_format: str = "RGB",
        confidence_threshold: Optional[float] = None,
    ) -> List[BoundingBox]:
        if image is None or not isinstance(image, np.ndarray):
            raise InvalidInputError("Input image must be a non-null numpy.ndarray.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputError(f"Expected 3-channel image array [H, W, 3], received shape {image.shape}.")
        if image.dtype != np.uint8:
            raise InvalidInputError(f"Expected uint8 image array, received dtype {image.dtype}.")

        h0, w0 = image.shape[:2]
        if h0 == 0 or w0 == 0:
            return []

        # Convert BGR to RGB if passed from OpenCV
        if color_format.upper() == "BGR":
            work_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_format.upper() == "RGB":
            work_img = image
        else:
            raise InvalidInputError(f"Unsupported color_format '{color_format}'. Must be 'RGB' or 'BGR'.")

        canvas, scale, pad_x, pad_y = self._letterbox(work_img, target_size=self.input_size)

        # Preprocessing: [H, W, C] -> [1, C, H, W] normalized to [0, 1]
        blob = np.ascontiguousarray(canvas, dtype=np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        try:
            outputs = self.session.run([self.output_name], {self.input_name: blob})
        except Exception as e:
            raise ModelInferenceError(f"Inference session execution failed: {e}") from e

        # YOLO output is [batch, 5, 8400] -> transpose to [8400, 5]
        preds = outputs[0][0].T
        cx = preds[:, 0]
        cy = preds[:, 1]
        bw = preds[:, 2]
        bh = preds[:, 3]
        scores = preds[:, 4]

        # Filter by confidence threshold (uses instance default or per-call override)
        conf_thresh = self.confidence_threshold if confidence_threshold is None else float(confidence_threshold)
        mask = scores >= conf_thresh
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        bw = bw[mask]
        bh = bh[mask]
        scores = scores[mask]

        # Map coordinates back to original image space
        x1 = (cx - bw / 2.0 - pad_x) / scale
        y1 = (cy - bh / 2.0 - pad_y) / scale
        x2 = (cx + bw / 2.0 - pad_x) / scale
        y2 = (cy + bh / 2.0 - pad_y) / scale

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)

        keep_indices = self._nms(boxes, scores, self.nms_iou_threshold, max_detections=self.max_detections)

        results: List[BoundingBox] = []
        for idx in keep_indices:
            bx1, by1, bx2, by2 = boxes[idx]
            conf = float(scores[idx])
            results.append(
                BoundingBox(
                    x1=float(bx1),
                    y1=float(by1),
                    x2=float(bx2),
                    y2=float(by2),
                    confidence=conf,
                    class_name="plate",
                )
            )

        return results

    # Primes ONNX session execution provider with dummy tensors
    def warmup(self, iterations: int = 2) -> None:
        dummy_img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        for _ in range(max(1, iterations)):
            self.detect(dummy_img, color_format="RGB")
