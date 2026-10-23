from abc import ABC, abstractmethod
import time
import os
import sys
import re
import cv2
import numpy as np

# Import preprocessing and validation modules
from lpr_core.preprocessor import PlatePreprocessor
from lpr_core.validator import VietnamPlateValidator

# Import-level fallback for Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Output directories for organized storage
OUTPUTS_DIR = "Outputs"
BASELINE_DIR = os.path.join(OUTPUTS_DIR, "baseline_output")
BENCH_DIR = os.path.join(OUTPUTS_DIR, "bench_test_output")
REFINED_DIR = os.path.join(OUTPUTS_DIR, "refined_output")
BALANCED_DIR = os.path.join(OUTPUTS_DIR, "balanced_output")

for d in [OUTPUTS_DIR, BASELINE_DIR, BENCH_DIR, REFINED_DIR, BALANCED_DIR]:
    os.makedirs(d, exist_ok=True)

# Helper to determine target output folder based on model name
def get_target_output_dir(model_name: str) -> str:
    m_lower = model_name.lower().replace("\\", "/")
    bname = os.path.basename(m_lower)
    if "balanced" in bname:
        return BALANCED_DIR
    elif "refined" in bname:
        return REFINED_DIR
    elif bname in ["best.pt", "best.onnx"]:
        return BASELINE_DIR
    else:
        return BENCH_DIR

# Helper to generate unique, non-overwriting filenames (cleanly appends: _1 -> _2 -> _3 without mangling image names)
def get_unique_filepath(directory: str, filename: str) -> str:
    candidate = os.path.join(directory, filename)
    if not os.path.exists(candidate):
        return candidate
        
    stem, ext = os.path.splitext(filename)
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1

# Vehicle target classes for object filtering
VEHICLE_CLASSES = ['license_plate', 'plate'] # cleaned plate


# Abstract base class for detection models
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str):
        pass


# Abstract base class for optical character recognition models
class BaseOCR(ABC):
    @abstractmethod
    def recognize(self, image_crop: np.ndarray, debug_prefix: str = None):
        pass


# Ultralytics YOLO detector adapter (supports ONNX runtime exports and custom weights)
class YOLODetector(BaseDetector):
    # Initializes detector with model weights, target class filter, and confidence threshold
    def __init__(self, model_weights: str = "Models_yolo/best.onnx", 
                 target_classes: list = VEHICLE_CLASSES, 
                 confidence_threshold: float = 0.45):
        from ultralytics import YOLO
        if not os.path.exists(model_weights):
            if model_weights.endswith(".onnx"):
                pt_alt = model_weights[:-5] + ".pt"
                if os.path.exists(pt_alt):
                    print(f"ONNX model '{model_weights}' not found. Loading PyTorch weights '{pt_alt}' instead.")
                    model_weights = pt_alt
                else:
                    raise FileNotFoundError(f"Detector weights not found: {model_weights}")
            else:
                raise FileNotFoundError(f"Detector weights not found: {model_weights}")
                
        self.model_name = model_weights
        self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
        print(f"Loading detector weights: {model_weights} (confidence threshold: {confidence_threshold})")
        self.model = YOLO(model_weights)
        
    # Detects objects in image path and returns bounding boxes
    def detect(self, image_path: str, imgsz: int = 640):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Detection image path not found: {image_path}")
            
        results = self.model(image_path, conf=self.confidence_threshold, imgsz=imgsz, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                cls_id = int(b.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(b.conf[0])
                
                # Filter out detections not in target classes if target_classes is set
                if self.target_classes is not None and cls_name not in self.target_classes:
                    continue
                    
                xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'box': xyxy
                })
        return detections


# PaddleOCR model adapter
class PaddleOCRAdapter(BaseOCR):
    # Initializes PaddleOCR with streamlined settings
    def __init__(self, lang: str = 'en', mode: str = 'full'):
        import logging
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Suppress Paddle and PaddleX internal logging bloat
        os.environ["FLAGS_allocator_strategy"] = "auto_growth"
        os.environ["GLOG_minloglevel"] = "2"
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        
        import torch
        self.mode = mode.lower()
        if self.mode in ['no_dbnet', 'rec_only', 'paddle_no_dbnet']:
            from paddleocr import TextRecognition
            print(f"Initializing PaddleOCR (language: {lang}, mode: {self.mode} [Recognition Only / Zero DBNet])")
            self.rec_model = TextRecognition()
        else:
            from paddleocr import PaddleOCR
            print(f"Initializing PaddleOCR (language: {lang}, mode: {self.mode})")
            self.ocr_model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                lang=lang
            )
        
    # Performs text recognition on a cropped plate image
    def recognize(self, image_crop: np.ndarray, debug_prefix: str = None):
        if image_crop is None or image_crop.size == 0:
            raise ValueError("Input image crop to PaddleOCR is empty")
            
        if self.mode in ['no_dbnet', 'rec_only', 'paddle_no_dbnet']:
            return self._recognize_no_dbnet(image_crop)
        elif self.mode in ['split_lines', 'split', 'fast_split', 'geometric']:
            return self._recognize_split_lines(image_crop)
        else:
            return self._recognize_full(image_crop)

    # Recognition-only pipeline bypassing DBNet text detection
    def _recognize_no_dbnet(self, image_crop: np.ndarray):
        h, w = image_crop.shape[:2]
        aspect_ratio = w / float(h) if h > 0 else 1.0

        
        # Slicing logic: square 2-line plates (<= 2.2) vs long 1-line plates
        slices = [image_crop[0:int(h * 0.55), :], image_crop[int(h * 0.45):h, :]] if aspect_ratio <= 2.2 else [image_crop]
        
        ocr_entries = []
        for s in slices:
            if s.size == 0:
                continue
            predict_res = list(self.rec_model.predict(s))
            for item in predict_res:
                if isinstance(item, dict):
                    txt = item.get('rec_text', '')
                    score = float(item.get('rec_score', 0.0))
                    if txt:
                        ocr_entries.append({'text': txt, 'confidence': score})
        return ocr_entries

    # Standard full neural detection + recognition pipeline
    def _recognize_full(self, image_crop: np.ndarray):
        temp_crop_path = "_temp_crop.jpg"
        cv2.imwrite(temp_crop_path, image_crop)
        
        try:
            predict_res = list(self.ocr_model.predict(temp_crop_path))
        except Exception as e:
            if os.path.exists(temp_crop_path):
                os.remove(temp_crop_path)
            raise RuntimeError(f"PaddleOCR recognition failed: {e}")
            
        ocr_entries = []
        for item in predict_res:
            if isinstance(item, dict):
                texts = item.get('rec_texts', [])
                scores = item.get('rec_scores', [])
                boxes = item.get('rec_boxes', [])
                
                # Pair text, score, and box coordinates together
                paired = []
                for idx, (txt, score) in enumerate(zip(texts, scores)):
                    # Calculate vertical center (y_center) for reading order sorting
                    if idx < len(boxes) and len(boxes[idx]) >= 4:
                        y_center = (boxes[idx][1] + boxes[idx][3]) / 2.0
                    else:
                        y_center = float(idx)
                    paired.append({'text': txt, 'confidence': float(score), 'y_center': y_center})
                
                # Sort top-to-bottom so Line 1 always precedes Line 2 on 2-line plates
                paired = sorted(paired, key=lambda x: x['y_center'])
                for p in paired:
                    ocr_entries.append({'text': p['text'], 'confidence': p['confidence']})
        
        if os.path.exists(temp_crop_path):
            os.remove(temp_crop_path)
            
        return ocr_entries

    # Fast line-splitting mode using aspect ratio and horizontal slicing
    def _recognize_split_lines(self, image_crop: np.ndarray):
        h, w = image_crop.shape[:2]
        aspect_ratio = w / float(h) if h > 0 else 1.0

        
        slices = []
        if aspect_ratio <= 2.5:
            # 2-line square plate: slice top and bottom with 10% overlap
            h_split_top = int(h * 0.55)
            h_split_bot = int(h * 0.45)
            slice1 = image_crop[0:h_split_top, :]
            slice2 = image_crop[h_split_bot:h, :]
            slices = [slice1, slice2]
        else:
            # 1-line long plate
            slices = [image_crop]
            
        ocr_entries = []
        for s in slices:
            if s.size == 0:
                continue
            temp_s = "_temp_slice.jpg"
            cv2.imwrite(temp_s, s)
            try:
                predict_res = list(self.ocr_model.predict(temp_s))
                for item in predict_res:
                    if isinstance(item, dict):
                        texts = item.get('rec_texts', [])
                        scores = item.get('rec_scores', [])
                        for t, sc in zip(texts, scores):
                            ocr_entries.append({'text': t, 'confidence': float(sc)})
            finally:
                if os.path.exists(temp_s):
                    os.remove(temp_s)
                    
        return ocr_entries


# Orders 4 quadrilateral vertices clockwise starting from Top-Left (TL, TR, BR, BL)
def order_quad_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left has minimum (x + y) sum
    rect[2] = pts[np.argmax(s)]  # Bottom-Right has maximum (x + y) sum
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right has minimum (y - x) difference
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left has maximum (y - x) difference
    return rect


# Evaluates candidate license plate contours and calculates a deterministic contour confidence score [0.0, 1.0]
def compute_plate_quad_and_confidence(image_crop: np.ndarray) -> tuple:
    if image_crop is None or image_crop.size == 0:
        return None, 0.0

    h, w = image_crop.shape[:2]
    crop_area = float(h * w)
    if crop_area < 50:
        return None, 0.0

    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dual edge and threshold extraction
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours_canny, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in list(contours_thresh) + list(contours_canny):
        area = cv2.contourArea(cnt)
        if area < 0.25 * crop_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            rect = cv2.minAreaRect(cnt)
            pts = cv2.boxPoints(rect).astype("float32")

        ordered_pts = order_quad_points(pts)

        # Metric 1: Solidity / Area Fill Ratio
        fill_ratio = min(1.0, area / crop_area)
        if 0.40 <= fill_ratio <= 0.98:
            s_fill = 1.0
        elif fill_ratio < 0.40:
            s_fill = fill_ratio / 0.40
        else:
            s_fill = 0.8

        # Metric 2: Aspect Ratio Consistency (Motorbikes ~ 1.1-1.5, Automobiles ~ 2.0-4.5)
        tl, tr, br, bl = ordered_pts
        w_top = np.linalg.norm(tr - tl)
        w_bot = np.linalg.norm(br - bl)
        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)
        cw = (w_top + w_bot) / 2.0
        ch = (h_left + h_right) / 2.0
        if ch <= 0:
            continue
        ar = cw / ch

        if 1.0 <= ar <= 1.6 or 1.8 <= ar <= 4.8:
            s_ar = 1.0
        elif 0.8 <= ar < 1.0 or 1.6 < ar < 1.8:
            s_ar = 0.6
        else:
            s_ar = 0.2

        # Metric 3: Corner Angle Orthogonality (deviation from 90 degrees)
        v0 = tr - tl
        v1 = br - tr
        v2 = bl - br
        v3 = tl - bl

        def angle_deg(va, vb):
            dot = np.dot(va, vb)
            norm = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-6
            return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))

        a0 = angle_deg(v0, -v3)
        a1 = angle_deg(v1, -v0)
        a2 = angle_deg(v2, -v1)
        a3 = angle_deg(v3, -v2)
        dev = np.mean([abs(a0 - 90), abs(a1 - 90), abs(a2 - 90), abs(a3 - 90)])
        s_angle = max(0.0, min(1.0, 1.0 - (dev / 25.0)))

        # Metric 4: Luminance Contrast Differential
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [ordered_pts.astype(np.int32)], -1, 255, -1)
        mean_inside = cv2.mean(gray, mask=mask)[0]
        inv_mask = cv2.bitwise_not(mask)
        mean_outside = cv2.mean(gray, mask=inv_mask)[0]
        lum_diff = mean_inside - mean_outside

        if lum_diff >= 30:
            s_lum = 1.0
        elif lum_diff > 0:
            s_lum = lum_diff / 30.0
        else:
            s_lum = 0.5

        total_conf = (0.25 * s_fill) + (0.30 * s_ar) + (0.30 * s_angle) + (0.15 * s_lum)
        candidates.append((total_conf, ordered_pts))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], round(candidates[0][0], 3)


# Projects quadrilateral crop to canonical orthogonal coordinate frame
def canonical_perspective_unwarp(image_crop: np.ndarray, pts: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = pts
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    max_w = max(10, int(max(w_top, w_bot)))

    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    max_h = max(10, int(max(h_left, h_right)))

    dst_pts = np.array([
        [0, 0],
        [max_w - 1, 0],
        [max_w - 1, max_h - 1],
        [0, max_h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst_pts)
    return cv2.warpPerspective(image_crop, M, (max_w, max_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# Normalizes plate geometry via canonical perspective unwarp (or affine fallback) and slices dual-row motorbike plates
def deskew_and_slice_plate(image_crop: np.ndarray, debug_prefix: str = None, save_debug: bool = True) -> list:
    """
    Normalizes license plate geometry using 4-corner Canonical Geometry Normalizer
    when contour confidence >= 0.70, falls back to Hough affine deskewing if low,
    and slices dual-row motorbike plates into top and bottom strips.
    """
    if image_crop is None or image_crop.size == 0:
        return []

    h, w = image_crop.shape[:2]
    aspect_ratio = w / float(h) if h > 0 else 1.0

    # Step 1: Attempt Canonical 4-Point Normalization
    quad_pts, contour_conf = compute_plate_quad_and_confidence(image_crop)

    if contour_conf >= 0.70 and quad_pts is not None:
        straight = canonical_perspective_unwarp(image_crop, quad_pts)
        sh, sw = straight.shape[:2]
        aspect_ratio = sw / float(sh) if sh > 0 else aspect_ratio
    else:
        # Step 1b: Fallback to Hough-line tilt detection & affine rotation
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        min_line_len = max(12, int(w * 0.22))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=min_line_len, maxLineGap=8)

        skew_angle = 0.0
        if lines is not None:
            valid_angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    continue
                deg = np.degrees(np.arctan2(dy, dx))
                if -25.0 <= deg <= 25.0:
                    valid_angles.append(deg)
            if valid_angles:
                skew_angle = float(np.median(valid_angles))

        if abs(skew_angle) > 1.5:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            if aspect_ratio > 1.5:
                straight = cv2.warpAffine(image_crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            else:
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                M[0, 2] += (nW / 2) - center[0]
                M[1, 2] += (nH / 2) - center[1]
                straight = cv2.warpAffine(image_crop, M, (nW, nH), borderMode=cv2.BORDER_REPLICATE)
        else:
            straight = image_crop

    # Aspect ratio > 1.5 indicates a single-row rectangular automobile plate (no horizontal slicing needed)
    if aspect_ratio > 1.5:
        if save_debug and debug_prefix:
            for debug_dir in ["Outputs/deskewed_slices", "debug_slices"]:
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_full_crop.jpg"), image_crop)
                cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_deskewed.jpg"), straight)
        return [straight]

    # Step 2: For dual-row square plates, isolate bright plate surface (discard dark mudguard / bumper)
    sgray = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY) if len(straight.shape) == 3 else straight
    sh, sw = sgray.shape
    row_means = np.mean(sgray, axis=1)
    plate_thresh = max(80.0, float(np.mean(row_means) * 0.75))
    bright_rows = np.where(row_means >= plate_thresh)[0]

    if len(bright_rows) > int(sh * 0.40):
        y_start = max(0, int(bright_rows[0]))
        y_end = min(sh, int(bright_rows[-1]) + 1)
        plate_clean = straight[y_start:y_end, :]
    else:
        plate_clean = straight

    # Step 3: Horizontal line slicing with 4% vertical overlap
    ph, pw = plate_clean.shape[:2]
    top_slice = plate_clean[0:int(ph * 0.52), :]
    bot_slice = plate_clean[int(ph * 0.48):ph, :]

    # Step 4: Persist intermediate debug crops for inspection
    if save_debug and debug_prefix:
        for debug_dir in ["Outputs/deskewed_slices", "debug_slices"]:
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_full_crop.jpg"), image_crop)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_deskewed.jpg"), straight)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_top_slice.jpg"), top_slice)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_bot_slice.jpg"), bot_slice)

    return [top_slice, bot_slice]

# # CRNN model adapter (Convolutional Recurrent Neural Network for scene and plate text)
# deprecated for now
#  
# 
# 
# 
# 
# class CRNNAdapter(BaseOCR):
#     def __init__(self, model_path: str = "Models_ocr/crnn.onnx", 
#                  chars: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"):
#         self.model_path = model_path
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"CRNN model weights not found: {model_path}. Please place weights in Models_ocr/.")
#         import onnxruntime as ort
#         self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
#         self.input_name = self.session.get_inputs()[0].name
#         self.chars = list(chars)

#     def recognize(self, image_crop: np.ndarray, debug_prefix: str = None):
#         if image_crop is None or image_crop.size == 0:
#             raise ValueError("Input image crop to CRNN is empty")
        
#         # Standard CRNN input dimensions: 100x32 grayscale or RGB
#         slices = deskew_and_slice_plate(image_crop, debug_prefix=debug_prefix, save_debug=True)
        
#         results = []
#         for s in slices:
#             gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) if len(s.shape) == 3 else s
#             resized = cv2.resize(gray, (100, 32)).astype(np.float32)
#             normalized = (resized / 127.5) - 1.0
#             tensor = normalized[np.newaxis, np.newaxis, ...]
            
#             logits = self.session.run(None, {self.input_name: tensor})[0]
#             if len(logits.shape) == 3:
#                 logits_2d = logits[0] # (T, C)
#                 exp_logits = np.exp(logits_2d - np.max(logits_2d, axis=-1, keepdims=True))
#                 probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
#                 preds = np.argmax(probs, axis=-1)
#                 char_confs = []
#                 decoded = []
#                 prev = -1
#                 for i, idx in enumerate(preds):
#                     if idx != 0 and idx != prev:
#                         if idx < len(self.chars):
#                             decoded.append(self.chars[idx])
#                             char_confs.append(float(probs[i, idx]))
#                     prev = idx
#                 text = "".join(decoded)
#                 mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
#             else:
#                 preds = np.argmax(logits, axis=1)
#                 decoded = []
#                 prev = -1
#                 for idx in preds:
#                     if idx != 0 and idx != prev:
#                         if idx < len(self.chars):
#                             decoded.append(self.chars[idx])
#                     prev = idx
#                 text = "".join(decoded)
#                 mean_conf = 0.0
#             if text:
#                 results.append({'text': text, 'confidence': round(mean_conf, 4)})
#         return results if results else [{'text': '', 'confidence': 0.0}]

# DEPRECATED lprnet adapter for now.
# # LPRNet model adapter (lightweight spatial CNN for license plates)
# class LPRNetAdapter(BaseOCR):
#     def __init__(self, model_path: str = "Models_ocr/lprnet.onnx"):
#         self.model_path = model_path
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"LPRNet model weights not found: {model_path}.")
#         import onnxruntime as ort
#         self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
#         self.input_name = self.session.get_inputs()[0].name # input tensor name software can feed data correctly
#         self.chars = ['_'] + list("0123456789ABCDEFGHKLMNPRSTUVXYZ-") # CTC blank token at index 0 followed by 32 characters

#     def recognize(self, image_crop: np.ndarray, debug_prefix: str = None):
#         if image_crop is None or image_crop.size == 0:
#             raise ValueError("Input image crop to LPRNet is empty")

#         slices = deskew_and_slice_plate(image_crop, debug_prefix=debug_prefix, save_debug=True)

#         results = []
#         for s in slices:
#             if s.size == 0:
#                 continue
#             resized = cv2.resize(s, (94, 24)).astype(np.float32)
#             normalized = (resized - 127.5) * 0.0078125
#             tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
#             logits = self.session.run(None, {self.input_name: tensor})[0]
#             logits_2d = logits[0] # (num_classes, seq_len)

#             # Compute numerically stable softmax probabilities along class dimension
#             exp_logits = np.exp(logits_2d - np.max(logits_2d, axis=0, keepdims=True))
#             probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

#             best_path = np.argmax(probs, axis=0)
#             decoded = []
#             char_confs = []
#             for i, idx in enumerate(best_path):
#                 if idx != 0 and (i == 0 or idx != best_path[i - 1]):
#                     if idx < len(self.chars):
#                         decoded.append(self.chars[idx])
#                         char_confs.append(float(probs[idx, i]))
#             text = "".join(decoded)
#             if text:
#                 mean_conf = float(np.mean(char_confs)) if char_confs else 0.0
#                 results.append({'text': text, 'confidence': round(mean_conf, 4)})
#         return results if results else [{'text': '', 'confidence': 0.0}]


# CCT-XS compact convolutional transformer adapter (Fast Plate OCR)
class CCTXSRecognizerAdapter(BaseOCR):
    def __init__(self, model_path: str = "Models_ocr/cct_xs.onnx", config_path: str = "Models_ocr/cct_xs_plate_config.yaml"):
        from fast_plate_ocr import LicensePlateRecognizer
        if os.path.exists(model_path) and os.path.exists(config_path):
            self.recognizer = LicensePlateRecognizer(
                onnx_model_path=model_path,
                plate_config_path=config_path,
                device='cpu'
            )
        else:
            self.recognizer = LicensePlateRecognizer(
                hub_ocr_model='cct-xs-v2-global-model',
                device='cpu'
            )

    def recognize(self, image_crop: np.ndarray, debug_prefix: str = None):
        #  adding new fields.

        if image_crop is None or image_crop.size == 0:
            return [{'text': '', 'confidence': 0.0, 'min_confidence': 0.0, 'region': 'Unknown', 'region_prob':0.0}]

        preds = self.recognizer.run(image_crop, return_confidence=True)
        if not preds or not preds[0].plate:
            return [{'text': '', 'confidence': 0.0, 'min_confidence': 0.0, 'region': 'Unknown', 'region_prob':0.0}]
            

        pred = preds[0]
        plate_text = pred.plate
        plate_len = len(plate_text)

        # slice to each char
        if pred.char_probs is not None and plate_len >0:
            plate_char_probs = pred.char_probs[:plate_len]
            mean_conf = float(np.mean(plate_char_probs))
            min_conf = float(np.min(plate_char_probs))
        else:
            mean_conf=0.0
            min_conf=0.0

        region = getattr(pred, 'region', 'Unknown') or 'Unknown'
        region_prob = float(getattr(pred, 'region_prob', 0.0) or 0.0)

        return [{
            'text':plate_text,
            'confidence': round(mean_conf,4),
            'min_confidence': round(min_conf,4),
            'region': region,
            'region_prob': round(region_prob,4)
        }]


# Modular OCR factory loader
def get_ocr_adapter(name: str = "cct_xs", **kwargs) -> BaseOCR:
    name_clean = name.lower().strip()
    if name_clean in ["cct_xs", "cct", "fast_plate", "fast_plate_ocr"]:
        return CCTXSRecognizerAdapter(**kwargs)
    elif name_clean in ["paddleocr", "paddle", "paddle_full", "full"]:
        return PaddleOCRAdapter(mode="full", **kwargs)
    elif name_clean in ["paddle_no_dbnet", "no_dbnet", "rec_only"]:
        return PaddleOCRAdapter(mode="no_dbnet", **kwargs)
    elif name_clean in ["paddle_split", "split_lines", "split", "paddle_fast", "fast_split", "paddle_geometric", "geometric"]:
        return PaddleOCRAdapter(mode="split_lines", **kwargs)
    elif name_clean in ["crnn", "crnn_onnx"]:
        raise ValueError("crnn has been deprecated for now please use cct_xs for production or paddleocr for baseline")
        # return CRNNAdapter(**kwargs)
    elif name_clean in ["lprnet"]:
        raise ValueError("lprnet has been permanently deprecated (its buns for this case;) please use cct_xs for production or paddleocr for baseline")
        # return LPRNetAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown OCR adapter type: '{name}'. Supported: cct_xs, paddleocr, paddle_no_dbnet, paddle_split, crnn, lprnet")


# Unified License Plate Recognition pipeline
class LPRPipeline:
    # Composes detector, OCR adapter, preprocessing flag, and optional fallback detector
    def __init__(self, detector: BaseDetector, ocr: BaseOCR, enable_preprocessing: bool = True, fallback_detector: BaseDetector = None):
        self.detector = detector
        self.ocr = ocr
        self.enable_preprocessing = enable_preprocessing
        self.fallback_detector = fallback_detector
        self._is_first_run = True
    # ==============================================================================================
    # MULTI-VEHICLE / MULTI-PLATE SELECTION & CAMERA PERSPECTIVE LOGIC
    # ==============================================================================================
    # When multiple vehicles or plates appear in a single frame (e.g. queuing at a gate barrier),
    # YOLO detects ALL bounding boxes. This function filters and picks the TARGET plate for OCR.
    #
    # CUSTOMIZE BASED ON YOUR CAMERA SETUP:
    # 1. Overhead Boom Barrier Camera (Pointing down at approaching lane):
    #    -> Use sort_by='front': The vehicle closest to the boom gate is at the bottom of the
    #       image frame (highest y2 pixel coordinate). Detections are sorted by y2 descending.
    #
    # 2. Side-Mounted Toll Booth Camera:
    #    -> Use sort_by='area': The vehicle closest to the booth has the largest bounding box area.
    #
    # 3. High-Angle Distance Camera:
    #    -> Use sort_by='confidence': Selects the cleanest, most confident plate detected.
    #
    # To process ALL queuing vehicles simultaneously:
    #    -> Set primary_only=False (returns the full sorted list of all plates in frame).
    # ==============================================================================================
    def _select_primary(self, detections, sort_by='front', primary_only=True):
        if not detections:
            return []
            
        # Strategy A: Bottom-most bounding box in frame (closest foreground vehicle)
        if sort_by == 'front':
            detections = sorted(detections, key=lambda d: d['box'][3], reverse=True)
            
        # Strategy B: Largest bounding box area (closest vehicle by physical size)
        elif sort_by == 'area':
            detections = sorted(detections, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]), reverse=True)
            
        # Strategy C: Highest neural detector confidence
        elif sort_by == 'confidence':
            detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
            
        # If primary_only is True, gate controller only triggers on the single primary vehicle
        if primary_only and detections:
            return [detections[0]]
            
        return detections

    # Checks whether extracted OCR results strictly match Vietnamese plate syntax and region
    @staticmethod
    def _is_strictly_valid(ocr_results): 
        if not ocr_results:
            return False
        
        r0 = ocr_results[0]
        reg_name = r0.get('region', 'Unknown') or 'Unknown'
        reg_prob = float(r0.get('region_prob', 0.0) or 0.0)

        if reg_name not in ['Vietnam', 'Unknown'] and reg_prob >= 0.50: # auto remv
            return False

        if reg_name != 'Vietnam' and reg_name != "Unknown": # THIS ONE removes all plate that is not vietnamese however if the model needs to track itnernational cars then we can adjust
            return False

        if len(ocr_results) == 1:
            return ocr_results[0].get('is_valid', False)
        val_lines = [r['validated_text'] for r in ocr_results if r.get('validated_text')]
        joined = "-".join(val_lines)
        return VietnamPlateValidator.clean_and_correct(joined).get('is_valid', False)

    # Crops plate bounding boxes and runs OCR (natural or enhanced)
    def _crop_and_ocr(self, img, detections, crop_identifier, padding_percent, apply_enhancement=False, target_dir=OUTPUTS_DIR, model_tag="model", save_crop=True):
        ocr_results = []
        was_enhanced = False
        if not detections or img is None:
            return ocr_results, was_enhanced
            
        # is_lprnet = isinstance(self.ocr, LPRNetAdapter) # old deprecated lwk.
        is_cct_xs = isinstance(self.ocr, CCTXSRecognizerAdapter)

        for idx, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d['box'])
            h, w, _ = img.shape
            if is_cct_xs: # i removed is_lprnet.
                pad_w = max(3, min(12, int((x2 -x1) * 0.04))) # hard coded 4 pixel padding replaced w/ dynamic scale of boudning box.
                pad_h = max(3, min(12, int((y2 -y1) * 0.04))) # find distance between two points x2-x1 and get 4% of that.  output is never less than 3.
            else:
                pad_w = max(20, int((x2 - x1) * padding_percent))
                pad_h = max(15, int((y2 - y1) * padding_percent))
            
            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(w, x2 + pad_w)
            crop_y2 = min(h, y2 + pad_h)
            
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size > 0:
                slice_prefix = f"{crop_identifier}_{idx+1}" if len(detections) > 1 else crop_identifier
                final_crop = crop
                crop_h, crop_w = crop.shape[:2]
                blur_score = PlatePreprocessor.compute_blur_score(crop)
                
                # Adaptive Trigger: Trigger enhancement if crop is small/low-res (< 110x60) OR blurry (< 150)
                is_low_res = (crop_w < 110 or crop_h < 60) # MANUALLY INPUTTED PIXEL THAT IS RELEVANT TO THE 640X640 
                is_blurry = (blur_score < 150.0) or is_low_res
                
                if not apply_enhancement:
                    if is_blurry and self.enable_preprocessing and not (is_cct_xs): # removed is_lprnet
                        reason = f"low resolution ({crop_w}x{crop_h})" if is_low_res else f"sharpness {blur_score:.1f} < 150.0"
                        print(f"  [Adaptive Recovery] Plate needs enhancement ({reason}). Auto-applying stroke & resolution recovery...")
                        enhanced_crop = PlatePreprocessor.enhance_for_ocr(crop.copy())
                        raw_text_results = self.ocr.recognize(enhanced_crop, debug_prefix=slice_prefix)
                        final_crop = enhanced_crop
                        was_enhanced = True
                    else:
                        print(f"  [1st try - Fast Pass] Natural crop evaluated (size: {crop_w}x{crop_h}, sharpness: {blur_score:.1f})")
                        raw_text_results = self.ocr.recognize(crop, debug_prefix=slice_prefix)
                else:
                    # 2nd try: Explicit escalation or forced enhancement
                    if is_cct_xs:
                        # LPRNet expects tightly bounded crops matching its 94x24 input geometry
                        print(f"  [2nd try - Transformer Mild Enhancement] Applying mild sharpening and stroke recovery (sharpness: {blur_score:.1f})...")
                        enhanced_crop = PlatePreprocessor.mild_enhance_for_transformer(crop.copy())

                        raw_text_results = self.ocr.recognize(enhanced_crop, debug_prefix=slice_prefix) # part of 2nd try.
                        final_crop=enhanced_crop
                        was_enhanced = True
                    else:
                        print(f" NOTE - is_cct_xs NOT detected; this is default send: [2nd try - Enhancement Pass] Applying adaptive image enhancement (sharpness: {blur_score:.1f})...") # ts is general.
                        enhanced_crop = PlatePreprocessor.enhance_for_ocr(crop.copy())
                        raw_text_results = self.ocr.recognize(enhanced_crop, debug_prefix=slice_prefix)
                        final_crop = enhanced_crop
                        was_enhanced = True
                    
                if save_crop:
                    crop_filename = f"{model_tag}_crop_{crop_identifier}_{idx+1}.jpg"
                    crop_path = get_unique_filepath(target_dir, crop_filename)
                    cv2.imwrite(crop_path, final_crop)
                
                for res in raw_text_results:
                    val_res = VietnamPlateValidator.clean_and_correct(res['text'])
                    res['validated_text'] = val_res['text']
                    res['is_valid'] = val_res['is_valid']
                    res['format_similarity'] = val_res.get('format_similarity', 0.0)
                    res['is_format_similar'] = val_res.get('is_format_similar', False)
                    ocr_results.append(res)
        return ocr_results, was_enhanced

    # Executes detection, cropping, preprocessing, OCR, and validation
    def process(self, image_path: str, save_annotated_name: str = "output_annotated.jpg", 
                sort_by: str = 'front', primary_only: bool = True, padding_percent: float = 0.20,
                force_second_try: bool = False, force_third_try: bool = False,
                measure_warm: bool = False, output_dir: str = None, save_crops: bool = False):
        start_time = time.time()
        is_cold_start = getattr(self, '_is_first_run', True)
        self._is_first_run = False
        
        # Determine model tag and target output directory
        model_name = getattr(self.detector, 'model_name', 'yolo')
        model_stem = os.path.splitext(os.path.basename(model_name))[0]
        target_dir = output_dir if output_dir is not None else get_target_output_dir(model_name)
        os.makedirs(target_dir, exist_ok=True)
        
        img_stem = os.path.splitext(os.path.basename(image_path))[0]
        annotated_base = f"{model_stem}_output_{img_stem}.jpg"
        output_file_path = get_unique_filepath(target_dir, annotated_base)
        crop_identifier = img_stem
        
        print(f"Processing image: {image_path}")
        print(f"Detector: {self.detector.__class__.__name__} ({model_stem}) | OCR: {self.ocr.__class__.__name__}")
        print(f"Adaptive enhancement capability: {'Enabled' if self.enable_preprocessing else 'Disabled'}")
        if force_third_try:
            print("Mode: FORCED 3rd try (1024px Deep Zoom + OCR)")
        elif force_second_try:
            print("Mode: FORCED 2nd try (Adaptive Enhancement)")
        print(f"Output directory: {target_dir}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at path: {image_path}")

        # 1st try: Fastest pass at standard resolution (640px) with 100% natural crop (zero enhancement)
        det_start = time.time()
        try_level = "1st try"
        resolution = "640px"
        print(f"  [1st try - 640px] Running base detector...")
        detections = self.detector.detect(image_path, imgsz=640)
        detections = self._select_primary(detections, sort_by, primary_only)
        det_latency_ms = (time.time() - det_start) * 1000
        print(f"  [1st try - 640px] Found {len(detections)} detection(s) in {det_latency_ms:.1f}ms")
        
        ocr_start = time.time()
        ocr_results, was_enhanced = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=False, target_dir=target_dir, model_tag=model_stem, save_crop=save_crops)
        ocr_latency_ms = (time.time() - ocr_start) * 1000
        
        # If blur enhancement was needed to read the plate, mark it honestly as 2nd try
        if was_enhanced:
            try_level = "2nd try"
        
        # Check if 1st try succeeded: confident detection + strictly valid syntax + high character confidence.
        # checking charcter confidence before deciding when to escalate.

        ocr_conf = ocr_results[0].get('confidence', 0.0) if ocr_results else 0.0
        ocr_min_conf = ocr_results[0].get('min_confidence', 0.0) if ocr_results else 0.0
        # check region
        r0 = ocr_results[0] if ocr_results else {}
        reg_name = r0.get('region', 'Unknown') or 'Unknown'
        reg_prob = float(r0.get('region_prob', 0.0) or 0.0)
        is_region_valid = (reg_name in ['Vietnam', 'Unknown'] or reg_prob < 0.50)

                
        first_try_ok = (
            bool(detections) and 
            detections[0]['confidence'] >= self.detector.confidence_threshold and 
            self._is_strictly_valid(ocr_results) and 
            ocr_conf >= 0.70 and # considered first try only if ocr conf and min conf is above 70, 60 %.
            ocr_min_conf >= 0.60 and
            is_region_valid and
            not force_second_try and
            not force_third_try
        )
        
        # 2nd try: Adaptive enhancement pass on the existing crop
        # Triggered if 1st try detected a box (>= 0.25) but OCR syntax was invalid, or force_second_try is requested
        if (not first_try_ok or force_second_try) and not force_third_try and detections and detections[0]['confidence'] >= self.detector.confidence_threshold and self.enable_preprocessing:
            reason = "Forced 2nd try requested" if force_second_try else "1st try unverified"
            print(f"  [2nd try - Adaptive Enhancement] {reason}. Running enhancement on existing crop...")
            ocr_start_enh = time.time()
            enh_results, _ = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=True, target_dir=target_dir, model_tag=model_stem, save_crop=save_crops)
            ocr_latency_ms += (time.time() - ocr_start_enh) * 1000

            enh_conf = enh_results[0].get('confidence',0.0) if enh_results else 0.0
            enh_valid = self._is_strictly_valid(enh_results)

            if force_second_try or (enh_valid and enh_conf >= ocr_conf) or (enh_valid and not self._is_strictly_valid(ocr_results)):
                ocr_results=enh_results
                try_level="2nd try"
                print(f'    [2nd try - succceded] enhancement improved OCR | Conf: {enh_conf*100:.1f}')
        elif not detections:
            print("  [2nd try] Skipped (no plate detected in 1st try to enhance).")

        # 3rd try: High-resolution zoom pass (1024px)
        # Triggered if 1st/2nd try still have 0 detections or unverified syntax, or force_third_try is requested
        if not self._is_strictly_valid(ocr_results) or force_third_try:
            reason = "Forced 3rd try requested" if force_third_try else "Syntax unverified or 0 detections"
            print(f"Notice: {reason}. Escalating to 3rd try (1024px Deep Zoom)...")
            resolution = "1024px"
            det_start_2 = time.time()
            zoom_detections = self.detector.detect(image_path, imgsz=1024)
            zoom_detections = self._select_primary(zoom_detections, sort_by, primary_only)
            det_latency_ms += (time.time() - det_start_2) * 1000
            
            if zoom_detections:
                detections = zoom_detections
                try_level = "3rd try"
                ocr_start_2 = time.time()
                ocr_results, _ = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=self.enable_preprocessing, target_dir=target_dir, model_tag=model_stem, save_crop=save_crops)
                ocr_latency_ms += (time.time() - ocr_start_2) * 1000
                print(f"3rd try succeeded: Found plate at 1024px zoom | Conf: {detections[0]['confidence']:.2f}")
            else:
                try_level = "3rd try (exhausted)"
                print("Notice: 3rd try (1024px zoom) found 0 detections. All tries exhausted.")
                if not detections:
                    resolution = "1024px"

        total_time_s = time.time() - start_time
        total_time_ms = total_time_s * 1000

        # Execute warm pass to measure steady-state if requested
        warm_det_ms = None
        warm_ocr_ms = None
        warm_total_ms = None
        if measure_warm:
            print("  [Steady-State Measurement] Running warm pass (without cold start)...")
            warm_t0 = time.time()
            warm_det_t0 = time.time()
            warm_res_imgsz = 1024 if resolution == "1024px" else 640
            warm_dets = self.detector.detect(image_path, imgsz=warm_res_imgsz)
            warm_dets = self._select_primary(warm_dets, sort_by, primary_only)
            warm_det_ms = (time.time() - warm_det_t0) * 1000.0
            
            warm_ocr_ms = 0.0
            if warm_dets:
                warm_ocr_t0 = time.time()
                self._crop_and_ocr(
                    img, warm_dets, crop_identifier, padding_percent,
                    apply_enhancement=(try_level == "2nd try" or was_enhanced),
                    target_dir=target_dir, model_tag=model_stem, save_crop=False
                )
                warm_ocr_ms = (time.time() - warm_ocr_t0) * 1000.0
            warm_total_ms = (time.time() - warm_t0) * 1000.0

        if measure_warm:
            cold_det_ms = det_latency_ms
            cold_ocr_ms = ocr_latency_ms
            cold_total_ms = total_time_ms
        else:
            cold_det_ms = det_latency_ms if is_cold_start else None
            cold_ocr_ms = ocr_latency_ms if is_cold_start else None
            cold_total_ms = total_time_ms if is_cold_start else None
            warm_det_ms = det_latency_ms if not is_cold_start else None
            warm_ocr_ms = ocr_latency_ms if not is_cold_start else None
            warm_total_ms = total_time_ms if not is_cold_start else None

        # Extract primary OCR plate text for visualization
        plate_str = "NONE"
        is_plate_valid = False
        if ocr_results:
            val_lines = [r.get('validated_text', r.get('text', '')) for r in ocr_results if r.get('text')]
            joined = "".join(val_lines)
            val_out = VietnamPlateValidator.clean_and_correct(joined)
            plate_str = val_out['text']
            is_plate_valid = self._is_strictly_valid(ocr_results) # to not draw box around incorrect plates.
 
        # Determine run tag from filename (e.g. "Run #1", "Run #2")
        stem_out = os.path.splitext(os.path.basename(output_file_path))[0]
        orig_stem = os.path.splitext(annotated_base)[0]
        if stem_out == orig_stem:
            run_tag = "Run #1"
        elif stem_out.startswith(orig_stem + "_"):
            suffix_num = stem_out[len(orig_stem) + 1:]
            run_tag = f"Run #{int(suffix_num) + 1}" if suffix_num.isdigit() else "Run #1"
        else:
            run_tag = "Run #1"

        # Generate annotated visualization
        annotated_img = img.copy()
        img_h, img_w = annotated_img.shape[:2]

        # Formulate HUD text lines
        ocr_tag = self.ocr.__class__.__name__.replace("Adapter", "").replace("Recognizer", "")
        hud_line1 = f"Model: {model_stem} + {ocr_tag} | {run_tag} | Status: {try_level} ({resolution})"
        if measure_warm:
            hud_line2 = f"Cold: {cold_total_ms:.0f}ms (Det: {cold_det_ms:.0f}ms) | Warm: {warm_total_ms:.0f}ms (Det: {warm_det_ms:.0f}ms)"
        elif is_cold_start:
            hud_line2 = f"Latency (Cold-Start): {total_time_ms:.0f}ms (Det: {det_latency_ms:.0f}ms, OCR: {ocr_latency_ms:.0f}ms)"
        else:
            hud_line2 = f"Latency (Steady-State): {total_time_ms:.0f}ms (Det: {det_latency_ms:.0f}ms, OCR: {ocr_latency_ms:.0f}ms)"

        # Draw top HUD metadata bar (2 lines, slight transparent background, green text)
        font_scale = max(0.38, min(0.62, img_w / 1400.0))
        thickness = 1 if font_scale < 0.6 else 2

        # Ensure HUD lines never exceed image width
        while font_scale > 0.28:
            (w1, _), _ = cv2.getTextSize(hud_line1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            (w2, _), _ = cv2.getTextSize(hud_line2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            if max(w1, w2) <= img_w - 24:
                break
            font_scale -= 0.02

        (test_w, text_h), _ = cv2.getTextSize("Model: Sample", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        line_gap = int(text_h * 0.75)
        hud_h = int((text_h * 2) + (line_gap * 3) + 8)
        y_line1 = int(line_gap + text_h + 2)
        y_line2 = int(y_line1 + text_h + line_gap + 2)

        # Slight transparent dark background
        overlay = annotated_img.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, hud_h), (12, 18, 12), -1)
        cv2.addWeighted(overlay, 0.55, annotated_img, 0.45, 0, annotated_img)

        # HUD 2-line green text
        hud_color = (0, 255, 0) # High-contrast green
        cv2.putText(annotated_img, hud_line1, (12, y_line1), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, hud_color, thickness, cv2.LINE_AA)
        cv2.putText(annotated_img, hud_line2, (12, y_line2), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, hud_color, thickness, cv2.LINE_AA)

        # Draw bounding boxes and plate text labels
        if detections:
            for idx, d in enumerate(detections):
                x1, y1, x2, y2 = map(int, d['box'])
                box_color = (0, 220, 0) if is_plate_valid else (0, 165, 255) # Green if valid, Orange if unverified
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
                
                # Bounding box tag: "[PLATE] | [CONF]%"
                tag_text = f"{plate_str} ({d['confidence'] * 100:.1f}%)" if (idx == 0 and plate_str != "NONE") else f"{d['class']} ({d['confidence'] * 100:.1f}%)"
                (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                
                # Tag background rectangle
                tag_y1 = max(hud_h + 5, y1 - th - 8)
                cv2.rectangle(annotated_img, (x1, tag_y1), (x1 + tw + 8, tag_y1 + th + 6), box_color, -1)
                cv2.putText(annotated_img, tag_text, (x1 + 4, tag_y1 + th + 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                            
        cv2.imwrite(output_file_path, annotated_img)
        print(f"Visualization saved to: {output_file_path}")
        
        return {
            'detector': self.detector.__class__.__name__,
            'try_level': try_level,
            'resolution': resolution,
            'ocr': self.ocr.__class__.__name__,
            'detections': detections,
            'ocr_results': ocr_results,
            'detection_latency_ms': warm_det_ms if measure_warm else det_latency_ms,
            'ocr_latency_ms': warm_ocr_ms if measure_warm else ocr_latency_ms,
            'total_time_s': (warm_total_ms / 1000.0) if measure_warm else total_time_s,
            'cold_detection_latency_ms': cold_det_ms,
            'cold_ocr_latency_ms': cold_ocr_ms,
            'cold_total_latency_ms': cold_total_ms,
            'warm_detection_latency_ms': warm_det_ms,
            'warm_ocr_latency_ms': warm_ocr_ms,
            'warm_total_latency_ms': warm_total_ms,
            'annotated_path': output_file_path
        }
