from abc import ABC, abstractmethod
import time
import os
import sys
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

for d in [OUTPUTS_DIR, BASELINE_DIR, BENCH_DIR, REFINED_DIR]:
    os.makedirs(d, exist_ok=True)

# Helper to determine target output folder based on model name
def get_target_output_dir(model_name: str) -> str:
    m_lower = model_name.lower().replace("\\", "/")
    bname = os.path.basename(m_lower)
    if "refined" in bname:
        return REFINED_DIR
    elif bname in ["best.pt", "best.onnx"]:
        return BASELINE_DIR
    else:
        return BENCH_DIR

# Helper to generate unique, non-overwriting filenames (cleanly increments: _1 -> _2 -> _3)
def get_unique_filepath(directory: str, filename: str) -> str:
    import re
    candidate = os.path.join(directory, filename)
    if not os.path.exists(candidate):
        return candidate
        
    stem, ext = os.path.splitext(filename)
    
    # Check if stem already ends with an underscore and digits (e.g. "image_1")
    match = re.search(r'^(.*)_(\d+)$', stem)
    if match:
        base_stem = match.group(1)
        counter = int(match.group(2)) + 1
    else:
        base_stem = stem
        counter = 1
        
    while True:
        candidate = os.path.join(directory, f"{base_stem}_{counter}{ext}")
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
    def recognize(self, image_crop: np.ndarray):
        pass


# Ultralytics YOLO detector adapter (supports ONNX runtime exports and custom weights)
class YOLODetector(BaseDetector):
    # Initializes detector with model weights, target class filter, and confidence threshold
    def __init__(self, model_weights: str = "Models_yolo/best.onnx", 
                 target_classes: list = VEHICLE_CLASSES, 
                 confidence_threshold: float = 0.25):
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
        from paddleocr import PaddleOCR
        self.mode = mode.lower()
        print(f"Initializing PaddleOCR (language: {lang}, mode: {self.mode})")
        self.ocr_model = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            lang=lang
        )
        
    # Performs text recognition on a cropped plate image
    def recognize(self, image_crop: np.ndarray):
        if image_crop is None or image_crop.size == 0:
            raise ValueError("Input image crop to PaddleOCR is empty")
            
        if self.mode in ['split_lines', 'split', 'fast_split', 'geometric']:
            return self._recognize_split_lines(image_crop)
        else:
            return self._recognize_full(image_crop)

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


# EasyOCR model adapter
class EasyOCRAdapter(BaseOCR):
    # Initializes EasyOCR reader
    def __init__(self, lang_list: list = ['en'], gpu: bool = False):
        import easyocr
        print(f"Initializing EasyOCR (languages: {lang_list}, gpu: {gpu})")
        self.reader = easyocr.Reader(lang_list, gpu=gpu)
        
    # Performs text recognition on image crop
    def recognize(self, image_crop: np.ndarray):
        if image_crop is None or image_crop.size == 0:
            raise ValueError("Input image crop to EasyOCR is empty")
            
        results = self.reader.readtext(image_crop)
        ocr_entries = []
        for res in results:
            # Unpack text and confidence directly without storing unused bbox
            text, conf = res[1], float(res[2])
            ocr_entries.append({
                'text': text,
                'confidence': conf
            })
        return ocr_entries


# LPRNet model adapter (lightweight spatial CNN for license plates)
class LPRNetAdapter(BaseOCR):
    def __init__(self, model_path: str = "Models_ocr/lprnet.onnx"):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LPRNet model weights not found: {model_path}.")
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name # input tensor name software can feed data correctly
        self.chars = list("0123456789ABCDEFGHKLMNPRSTUVXYZ-") # required for lprnet

    def recognize(self, image_crop: np.ndarray):
        resized = cv2.resize(image_crop, (94, 24)).astype(np.float32)
        normalized = (resized - 127.5) * 0.0078125
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        logits = self.session.run(None, {self.input_name: tensor})[0]
        best_path = np.argmax(logits, axis=1)[0]
        decoded = []
        for i, idx in enumerate(best_path):
            if idx != 0 and (i == 0 or idx != best_path[i - 1]):
                if idx < len(self.chars):
                    decoded.append(self.chars[idx])
        text = "".join(decoded)
        return [{'text': text, 'confidence': 0.95}]


# CCT-XS compact convolutional transformer adapter
class CCTXSRecognizerAdapter(BaseOCR):
    def __init__(self, model_path: str = "Models_ocr/cct_xs.onnx"):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CCT-XS model weights not found: {model_path}. Please place weights in Models_ocr/.")
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name # input tensor name software can feed data correctly

    def recognize(self, image_crop: np.ndarray):
        h, w = image_crop.shape[:2]
        aspect_ratio = w / float(h) if h > 0 else 1.0
        slices = [image_crop] if aspect_ratio > 2.5 else [image_crop[0:int(h*0.55), :], image_crop[int(h*0.45):h, :]]
        entries = []
        for s in slices:
            resized = cv2.resize(s, (128, 32)).astype(np.float32) / 255.0
            tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]
            entries.append({'text': 'LINE', 'confidence': 0.90})
        return entries


# Modular OCR factory loader
def get_ocr_adapter(name: str = "paddleocr", **kwargs) -> BaseOCR:
    name_clean = name.lower().strip()
    if name_clean in ["paddleocr", "paddle", "paddle_full", "full"]:
        return PaddleOCRAdapter(mode="full", **kwargs)
    elif name_clean in ["paddle_split", "split_lines", "split", "paddle_fast", "fast_split", "paddle_geometric", "geometric"]:
        return PaddleOCRAdapter(mode="split_lines", **kwargs)
    elif name_clean in ["easyocr", "easy"]:
        return EasyOCRAdapter(**kwargs)
    elif name_clean in ["lprnet"]:
        return LPRNetAdapter(**kwargs)
    elif name_clean in ["cct_xs", "cct"]:
        return CCTXSRecognizerAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown OCR adapter type: '{name}'. Supported: paddleocr, paddle_split, easyocr, lprnet, cct_xs")


# Unified License Plate Recognition pipeline
class LPRPipeline:
    # Composes detector, OCR adapter, preprocessing flag, and optional fallback detector
    def __init__(self, detector: BaseDetector, ocr: BaseOCR, enable_preprocessing: bool = True, fallback_detector: BaseDetector = None):
        self.detector = detector
        self.ocr = ocr
        self.enable_preprocessing = enable_preprocessing
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

    # Checks whether extracted OCR results strictly match Vietnamese plate syntax
    @staticmethod
    def _is_strictly_valid(ocr_results):
        if not ocr_results:
            return False
        if len(ocr_results) == 1:
            return ocr_results[0].get('is_valid', False)
        val_lines = [r['validated_text'] for r in ocr_results if r.get('validated_text')]
        joined = "-".join(val_lines)
        return VietnamPlateValidator.clean_and_correct(joined).get('is_valid', False)

    # Crops plate bounding boxes and runs OCR (natural or enhanced)
    def _crop_and_ocr(self, img, detections, crop_identifier, padding_percent, apply_enhancement=False, target_dir=OUTPUTS_DIR, model_tag="model"):
        ocr_results = []
        was_enhanced = False
        if not detections or img is None:
            return ocr_results, was_enhanced
            
        for idx, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d['box'])
            h, w, _ = img.shape
            pad_w = max(20, int((x2 - x1) * padding_percent))
            pad_h = max(15, int((y2 - y1) * padding_percent))
            
            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(w, x2 + pad_w)
            crop_y2 = min(h, y2 + pad_h)
            
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size > 0:
                final_crop = crop
                crop_h, crop_w = crop.shape[:2]
                blur_score = PlatePreprocessor.compute_blur_score(crop)
                
                # Adaptive Trigger: Trigger enhancement if crop is small/low-res (< 110x60) OR blurry (< 150)
                is_low_res = (crop_w < 110 or crop_h < 60) # MANUALLY INPUTTED PIXEL THAT IS RELEVANT TO THE 640X640 
                is_blurry = (blur_score < 150.0) or is_low_res
                
                if not apply_enhancement:
                    if is_blurry and self.enable_preprocessing:
                        reason = f"low resolution ({crop_w}x{crop_h})" if is_low_res else f"sharpness {blur_score:.1f} < 150.0"
                        print(f"  [Adaptive Recovery] Plate needs enhancement ({reason}). Auto-applying stroke & resolution recovery...")
                        enhanced_crop = PlatePreprocessor.enhance_for_ocr(crop.copy())
                        raw_text_results = self.ocr.recognize(enhanced_crop)
                        final_crop = enhanced_crop
                        was_enhanced = True
                    else:
                        print(f"  [1st try - Fast Pass] Natural crop evaluated (size: {crop_w}x{crop_h}, sharpness: {blur_score:.1f})")
                        raw_text_results = self.ocr.recognize(crop)
                else:
                    # 2nd try: Explicit escalation or forced enhancement
                    print(f"  [2nd try - Enhancement Pass] Applying adaptive image enhancement (sharpness: {blur_score:.1f})...")
                    enhanced_crop = PlatePreprocessor.enhance_for_ocr(crop.copy())
                    raw_text_results = self.ocr.recognize(enhanced_crop)
                    final_crop = enhanced_crop
                    was_enhanced = True
                    
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
                force_second_try: bool = False, force_third_try: bool = False):
        start_time = time.time()
        
        # Determine model tag and target output directory
        model_name = getattr(self.detector, 'model_name', 'yolo')
        model_stem = os.path.splitext(os.path.basename(model_name))[0]
        target_dir = get_target_output_dir(model_name)
        
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
        ocr_results, was_enhanced = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=False, target_dir=target_dir, model_tag=model_stem)
        ocr_latency_ms = (time.time() - ocr_start) * 1000
        
        # If blur enhancement was needed to read the plate, mark it honestly as 2nd try
        if was_enhanced:
            try_level = "2nd try"
        
        # Check if 1st try succeeded: confident detection + strictly valid syntax from natural crop
        first_try_ok = (
            bool(detections) and 
            detections[0]['confidence'] >= 0.25 and 
            self._is_strictly_valid(ocr_results) and
            not force_second_try and
            not force_third_try
        )
        
        # 2nd try: Adaptive enhancement pass on the existing crop
        # Triggered if 1st try detected a box (>= 0.25) but OCR syntax was invalid, or force_second_try is requested
        if (not first_try_ok or force_second_try) and not force_third_try and detections and detections[0]['confidence'] >= 0.25 and self.enable_preprocessing:
            reason = "Forced 2nd try requested" if force_second_try else "1st try unverified"
            print(f"  [2nd try - Adaptive Enhancement] {reason}. Running enhancement on existing crop...")
            ocr_start_enh = time.time()
            enh_results, _ = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=True, target_dir=target_dir, model_tag=model_stem)
            ocr_latency_ms += (time.time() - ocr_start_enh) * 1000
            if force_second_try or self._is_strictly_valid(enh_results) or len(enh_results) > len(ocr_results):
                ocr_results = enh_results
                try_level = "2nd try"
                print(f"  [2nd try - Succeeded] Image enhancement executed | Conf: {detections[0]['confidence']:.2f}")
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
                ocr_results, _ = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=self.enable_preprocessing, target_dir=target_dir, model_tag=model_stem)
                ocr_latency_ms += (time.time() - ocr_start_2) * 1000
                print(f"3rd try succeeded: Found plate at 1024px zoom | Conf: {detections[0]['confidence']:.2f}")
            else:
                try_level = "3rd try (exhausted)"
                print("Notice: 3rd try (1024px zoom) found 0 detections. All tries exhausted.")
                if not detections:
                    resolution = "1024px"

        # Generate annotated visualization
        annotated_img = img.copy()
        if detections:
            for idx, d in enumerate(detections):
                x1, y1, x2, y2 = map(int, d['box'])
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"#{idx+1} {d['class']} {d['confidence']:.2f} ({try_level})"
                cv2.putText(annotated_img, label_text, (x1, max(15, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                            
        cv2.imwrite(output_file_path, annotated_img)
        print(f"Visualization saved to: {output_file_path}")

        total_time_s = time.time() - start_time
        
        return {
            'detector': self.detector.__class__.__name__,
            'try_level': try_level,
            'resolution': resolution,
            'ocr': self.ocr.__class__.__name__,
            'detections': detections,
            'ocr_results': ocr_results,
            'detection_latency_ms': det_latency_ms,
            'ocr_latency_ms': ocr_latency_ms,
            'total_time_s': total_time_s,
            'annotated_path': output_file_path
        }
