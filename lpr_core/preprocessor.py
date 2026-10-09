import cv2
import numpy as np

class PlatePreprocessor:
    """
    Advanced Computer Vision Preprocessing Pipeline to reinforce OCR accuracy against:
    - Mud & dirt smears across characters
    - Harsh sunlight glare, low light, and deep shadows
    - Pinhole paint scratches and stroke noise
    """
    
    @staticmethod
    def enhance_for_ocr(crop_bgr: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
        """
        Applies a 3-pass OpenCV enhancement pipeline optimized for Deep Learning OCR:
        1. Grayscale Conversion
        2. CLAHE (Contrast Limited Adaptive Histogram Equalization) -> Equalizes glare/shadows
        3. Bilateral Filter -> Edge-preserving denoising to smooth out dirt specks
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        # Pass 1: Convert to Grayscale
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        
        # Pass 2: CLAHE Contrast Equalization
        # Fixes dark shadows and bright sun glare across the plate
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)
        
        # Pass 3: Bilateral Filtering (Edge-preserving noise reduction)
        # Smooths dirt specks while preserving sharp character boundaries
        denoised = cv2.bilateralFilter(contrast_enhanced, d=7, sigmaColor=50, sigmaSpace=50)
        
        # Convert back to 3-channel BGR image expected by PaddleOCR / EasyOCR
        enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

