from abc import ABC, abstractmethod
import time
import os
import sys
import psutil
import cv2
import numpy as np

# Import our image preprocessor & validator
from lpr_core.preprocessor import PlatePreprocessor
from lpr_core.validator import VietnamPlateValidator

# Ensure Windows UTF-8 console output
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure 'Outputs' folder exists for storing output images
OUTPUTS_DIR = "Outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def get_ram_usage_mb():
    """Utility function to measure current process memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Default vehicle classes to focus ONLY on vehicles & plates
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'license_plate']


# =====================================================================
# 1. ABSTRACT BASE CLASSES (Modular Interfaces)
# =====================================================================

class BaseDetector(ABC):
    """Abstract interface for Vehicle / License Plate Detectors."""
    
    @abstractmethod
    def detect(self, image_path: str):
        """
        Returns:
            list[dict]: List of detections:
                        { 'class': str, 'confidence': float, 'box': [x1, y1, x2, y2] }
        """
        pass


class BaseOCR(ABC):
    """Abstract interface for Character Recognition engines."""
    
    @abstractmethod
    def recognize(self, image_crop: np.ndarray):
        """
        Returns:
            list[dict]: List of recognized text entries:
                        { 'text': str, 'confidence': float }
        """
        pass


# =====================================================================
# 2. CONCRETE DETECTION MODEL ADAPTERS
# =====================================================================

class YOLOv8Detector(BaseDetector):
    """
    Adapter for Ultralytics YOLOv8 / YOLO11 models.
    Supports filtering by target classes and confidence thresholding.
    """
    
    def __init__(self, model_weights: str = "yolov8n.pt", 
                 target_classes: list = VEHICLE_CLASSES, 
                 confidence_threshold: float = 0.35):
        from ultralytics import YOLO
        self.model_name = model_weights
        self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
        print(f"[YOLOv8Detector] Loading '{model_weights}' (Filter: {target_classes}, Conf >= {confidence_threshold})...")
        self.model = YOLO(model_weights)
        
    def detect(self, image_path: str):
        results = self.model(image_path, conf=self.confidence_threshold, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for b in boxes:
                cls_id = int(b.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(b.conf[0])
                
                # Filter by target classes if specified
                if self.target_classes is not None and cls_name not in self.target_classes:
                    continue
                    
                xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'box': xyxy
                })
        return detections


# =====================================================================
# 3. CONCRETE OCR MODEL ADAPTERS
# =====================================================================

class PaddleOCRAdapter(BaseOCR):
    """Adapter for PaddlePaddle-native PaddleOCR engine."""
    
    def __init__(self, lang: str = 'en'):
        import torch
        from paddleocr import PaddleOCR
        print(f"[PaddleOCRAdapter] Initializing PaddleOCR (lang={lang}, enable_mkldnn=False)...")
        self.ocr_model = PaddleOCR(enable_mkldnn=False, lang=lang)
        
    def recognize(self, image_crop: np.ndarray):
        temp_crop_path = "_temp_crop.jpg"
        cv2.imwrite(temp_crop_path, image_crop)
        
        predict_res = list(self.ocr_model.predict(temp_crop_path))
        ocr_entries = []
        for item in predict_res:
            if isinstance(item, dict):
                texts = item.get('rec_texts', [])
                scores = item.get('rec_scores', [])
                for txt, score in zip(texts, scores):
                    ocr_entries.append({
                        'text': txt,
                        'confidence': float(score)
                    })
        
        if os.path.exists(temp_crop_path):
            try:
                os.remove(temp_crop_path)
            except Exception:
                pass
            
        return ocr_entries


class EasyOCRAdapter(BaseOCR):
    """Adapter for PyTorch-native EasyOCR engine."""
    
    def __init__(self, lang_list: list = ['en'], gpu: bool = False):
        import easyocr
        print(f"[EasyOCRAdapter] Initializing EasyOCR (langs={lang_list}, gpu={gpu})...")
        self.reader = easyocr.Reader(lang_list, gpu=gpu)
        
    def recognize(self, image_crop: np.ndarray):
        results = self.reader.readtext(image_crop)
        ocr_entries = []
        for res in results:
            bbox, text, conf = res[0], res[1], float(res[2])
            ocr_entries.append({
                'text': text,
                'confidence': conf
            })
        return ocr_entries


# =====================================================================
# 4. UNIFIED LPR PIPELINE RUNNER (WITH AUTOMATIC PREPROCESSING)
# =====================================================================

class LPRPipeline:
    """Unified LPR Engine combining any BaseDetector and any BaseOCR implementation."""
    
    def __init__(self, detector: BaseDetector, ocr: BaseOCR, enable_preprocessing: bool = True):
        self.detector = detector
        self.ocr = ocr
        self.enable_preprocessing = enable_preprocessing
        
    def process(self, image_path: str, save_annotated_name: str = "output_annotated.jpg", 
                sort_by: str = 'front', primary_only: bool = True, padding_percent: float = 0.05):
        """
        Processes any input image path through Detection -> CLAHE/Bilateral Preprocessing -> OCR -> RegEx Validation.
        """
        start_time = time.time()
        start_ram = get_ram_usage_mb()
        
        output_file_path = os.path.join(OUTPUTS_DIR, os.path.basename(save_annotated_name))
        
        print("\n" + "=" * 60)
        print(f"RUNNING LPR PIPELINE ON IMAGE: '{image_path}'")
        print(f"Detector: {self.detector.__class__.__name__} | OCR: {self.ocr.__class__.__name__}")
        print(f"Preprocess Enchancement: {'ENABLED (CLAHE+Bilateral)' if self.enable_preprocessing else 'DISABLED'}")
        print("=" * 60)
        
        # Step 1: Detect objects / ROIs
        det_start = time.time()
        detections = self.detector.detect(image_path)
        det_latency_ms = (time.time() - det_start) * 1000
        
        print(f"  [1. Detection] Found {len(detections)} vehicle(s) in {det_latency_ms:.2f} ms")
        
        # Proximity sorting (bottom y2 coordinate)
        if detections:
            if sort_by == 'front':
                detections = sorted(detections, key=lambda d: d['box'][3], reverse=True)
            elif sort_by == 'area':
                detections = sorted(detections, key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]), reverse=True)
            elif sort_by == 'confidence':
                detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)

        if primary_only and detections:
            primary_vehicle = detections[0]
            detections = [primary_vehicle]
            print(f"  [Proximity Priority] Selected Closest Vehicle in Front: Class '{primary_vehicle['class']}' | Bottom y2={primary_vehicle['box'][3]}")

        for idx, d in enumerate(detections):
            print(f"      ({idx+1}) Class: '{d['class']}' | Conf: {d['confidence']:.2f} | BBox: {d['box']}")
            
        # Step 2: Read image and prepare annotated visualization
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: '{image_path}'")
            
        annotated_img = img.copy()
        ocr_results = []
        ocr_start = time.time()
        
        if detections:
            for idx, d in enumerate(detections):
                x1, y1, x2, y2 = map(int, d['box'])
                
                h, w, _ = img.shape
                pad_w = int((x2 - x1) * padding_percent)
                pad_h = int((y2 - y1) * padding_percent)
                
                crop_x1 = max(0, x1 - pad_w)
                crop_y1 = max(0, y1 - pad_h)
                crop_x2 = min(w, x2 + pad_w)
                crop_y2 = min(h, y2 + pad_h)
                
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"#{idx+1} {d['class']} {d['confidence']:.2f}"
                cv2.putText(annotated_img, label_text, (x1, max(15, y1 - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Crop region
                crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size > 0:
                    # 📚 ENHANCED CROP INTEGRATION: Pass crop through CLAHE + Bilateral Denoising
                    if self.enable_preprocessing:
                        crop = PlatePreprocessor.enhance_for_ocr(crop)
                        
                    raw_text_results = self.ocr.recognize(crop)
                    
                    # Pass through Vietnamese RegEx Validator
                    for res in raw_text_results:
                        val_res = VietnamPlateValidator.clean_and_correct(res['text'])
                        res['validated_text'] = val_res['text']
                        res['is_valid'] = val_res['is_valid']
                        ocr_results.append(res)
        else:
            print("  [Notice] No vehicle bounding boxes detected. Running OCR on entire image...")
            crop = img
            if self.enable_preprocessing:
                crop = PlatePreprocessor.enhance_for_ocr(crop)
            raw_text_results = self.ocr.recognize(crop)
            for res in raw_text_results:
                val_res = VietnamPlateValidator.clean_and_correct(res['text'])
                res['validated_text'] = val_res['text']
                res['is_valid'] = val_res['is_valid']
                ocr_results.append(res)
            
        ocr_latency_ms = (time.time() - ocr_start) * 1000
        
        print(f"  [2. OCR Recognition] Extracted {len(ocr_results)} text string(s) in {ocr_latency_ms:.2f} ms")
        for res in ocr_results:
            print(f"      * Raw: '{res['text']}' -> Validated: '{res['validated_text']}' | Confidence: {res['confidence']:.4f} ({res['confidence']*100:.2f}%)")
            
        if output_file_path:
            cv2.imwrite(output_file_path, annotated_img)
            print(f"\n  [Visual Output] Saved visualization to '{output_file_path}'")

        total_time_s = time.time() - start_time
        end_ram = get_ram_usage_mb()
        
        return {
            'detector': self.detector.__class__.__name__,
            'ocr': self.ocr.__class__.__name__,
            'detections': detections,
            'ocr_results': ocr_results,
            'detection_latency_ms': det_latency_ms,
            'ocr_latency_ms': ocr_latency_ms,
            'total_time_s': total_time_s,
            'ram_usage_mb': end_ram,
            'annotated_path': output_file_path
        }
