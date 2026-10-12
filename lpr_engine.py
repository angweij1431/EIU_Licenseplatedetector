from abc import ABC, abstractmethod
import time
import os
import sys
import psutil
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

# Output directory for saving visualizations and preprocessed crops
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Memory tracking helper (measures resident set size in megabytes)
def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Vehicle target classes for object filtering
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'license_plate', 'plate']


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


# Ultralytics YOLO detector adapter (supports YOLOv8, YOLO11, and custom fine-tuned weights)
class YOLODetector(BaseDetector):
    # Initializes detector with model weights, target class filter, and confidence threshold
    def __init__(self, model_weights: str = "Models_yolo/best.pt", 
                 target_classes: list = VEHICLE_CLASSES, 
                 confidence_threshold: float = 0.25):
        from ultralytics import YOLO
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


# Backward-compatible alias
YOLOv8Detector = YOLODetector


# PaddleOCR model adapter
class PaddleOCRAdapter(BaseOCR):
    # Initializes PaddleOCR with streamlined settings (disabling doc unwarping and rotation overhead)
    def __init__(self, lang: str = 'en'):
        import torch
        from paddleocr import PaddleOCR
        print(f"Initializing PaddleOCR (language: {lang}, streamlined: True)")
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


# Unified License Plate Recognition pipeline
class LPRPipeline:
    # Composes detector, OCR adapter, preprocessing flag, and optional fallback detector
    def __init__(self, detector: BaseDetector, ocr: BaseOCR, enable_preprocessing: bool = True, fallback_detector: BaseDetector = None):
        self.detector = detector
        self.ocr = ocr
        self.enable_preprocessing = enable_preprocessing
        self.fallback_detector = fallback_detector
        
    # Selects target vehicle / plate based on sorting criteria
    def _select_primary(self, detections, sort_by='front', primary_only=True):
        if not detections:
            return []
        if sort_by == 'front':
            detections = sorted(detections, key=lambda d: d['box'][3], reverse=True)
        elif sort_by == 'area':
            detections = sorted(detections, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]), reverse=True)
        elif sort_by == 'confidence':
            detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
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
    def _crop_and_ocr(self, img, detections, crop_identifier, padding_percent, apply_enhancement=False):
        ocr_results = []
        if not detections or img is None:
            return ocr_results
            
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
                if not apply_enhancement:
                    # 1st try: 100% natural crop (zero filters to preserve clean character curves)
                    print("  [1st try - Fast Pass] Natural crop evaluated with zero enhancement filters")
                    raw_text_results = self.ocr.recognize(crop)
                else:
                    # 2nd try: Image enhancement fallback (CLAHE, bilateral filter, dynamic upscale)
                    print("  [2nd try - Enhancement Pass] Applying adaptive image enhancement (CLAHE + upscale)...")
                    enhanced_crop = PlatePreprocessor.enhance_for_ocr(crop.copy())
                    raw_text_results = self.ocr.recognize(enhanced_crop)
                    final_crop = enhanced_crop
                    
                crop_filename = f"crop_{crop_identifier}_{idx+1}.jpg"
                crop_path = os.path.join(OUTPUTS_DIR, crop_filename)
                cv2.imwrite(crop_path, final_crop)
                
                for res in raw_text_results:
                    val_res = VietnamPlateValidator.clean_and_correct(res['text'])
                    res['validated_text'] = val_res['text']
                    res['is_valid'] = val_res['is_valid']
                    res['format_similarity'] = val_res.get('format_similarity', 0.0)
                    res['is_format_similar'] = val_res.get('is_format_similar', False)
                    ocr_results.append(res)
        return ocr_results

    # Executes detection, cropping, preprocessing, OCR, and validation
    def process(self, image_path: str, save_annotated_name: str = "output_annotated.jpg", 
                sort_by: str = 'front', primary_only: bool = True, padding_percent: float = 0.20):
        start_time = time.time()
        output_file_path = os.path.join(OUTPUTS_DIR, os.path.basename(save_annotated_name))
        crop_identifier = os.path.splitext(os.path.basename(save_annotated_name))[0]
        
        print(f"Processing image: {image_path}")
        print(f"Detector: {self.detector.__class__.__name__} | OCR: {self.ocr.__class__.__name__}")
        print(f"Adaptive enhancement capability: {'Enabled' if self.enable_preprocessing else 'Disabled'}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at path: {image_path}")

        # 1st try: Fastest pass at standard resolution (640px) with 100% natural crop (zero enhancement)
        det_start = time.time()
        try_level = "1st try"
        resolution = "640px"
        detections = self.detector.detect(image_path, imgsz=640)
        detections = self._select_primary(detections, sort_by, primary_only)
        det_latency_ms = (time.time() - det_start) * 1000
        
        ocr_start = time.time()
        ocr_results = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=False)
        ocr_latency_ms = (time.time() - ocr_start) * 1000
        
        # Check if 1st try succeeded: confident detection + strictly valid syntax from natural crop
        first_try_ok = (
            bool(detections) and 
            detections[0]['confidence'] >= 0.25 and 
            self._is_strictly_valid(ocr_results)
        )
        
        # 2nd try: Escalation pass with image enhancement (CLAHE/upscale) and/or 1024px zoom
        # Triggered if 1st try had: 0 detections, low confidence, or unverified natural crop
        if not first_try_ok:
            print("Notice: 1st try unverified. Escalating to 2nd try (Enhancement & Zoom)...")
            try_level = "2nd try"
            
            # Sub-step 2A: If plate was detected at 640px but text was faint, apply crop enhancement
            if detections and detections[0]['confidence'] >= 0.25 and self.enable_preprocessing:
                ocr_start_enh = time.time()
                enh_results = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=True)
                ocr_latency_ms += (time.time() - ocr_start_enh) * 1000
                if self._is_strictly_valid(enh_results) or len(enh_results) > len(ocr_results):
                    ocr_results = enh_results
                    print(f"2nd try succeeded: Image enhancement recovered valid plate | Conf: {detections[0]['confidence']:.2f}")
            
            # Sub-step 2B: If still unverified or 0 detections, escalate to 1024px zoom
            if not self._is_strictly_valid(ocr_results):
                resolution = "1024px"
                det_start_2 = time.time()
                zoom_detections = self.detector.detect(image_path, imgsz=1024)
                zoom_detections = self._select_primary(zoom_detections, sort_by, primary_only)
                det_latency_ms += (time.time() - det_start_2) * 1000
                
                if zoom_detections:
                    detections = zoom_detections
                    ocr_start_2 = time.time()
                    ocr_results = self._crop_and_ocr(img, detections, crop_identifier, padding_percent, apply_enhancement=self.enable_preprocessing)
                    ocr_latency_ms += (time.time() - ocr_start_2) * 1000
                    print(f"2nd try succeeded: Found plate at 1024px zoom | Conf: {detections[0]['confidence']:.2f}")
                else:
                    print("Notice: 2nd try (1024px zoom) found 0 detections.")
                    if not detections:
                        resolution = "none"

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
        end_ram = get_ram_usage_mb()
        
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
            'ram_usage_mb': end_ram,
            'annotated_path': output_file_path
        }
