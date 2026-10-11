import cv2
import numpy as np

# Computer Vision Preprocessing Pipeline for License Plate OCR
class PlatePreprocessor:
    @staticmethod
    def enhance_for_ocr(crop_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr

        # Step 1: Dynamic bicubic upscaling for small crops to reconstruct character strokes
        h, w = crop_bgr.shape[:2]
        if w < 160 or h < 80:
            scale_factor = max(2.5, 160.0 / max(1, w))
            crop_bgr = cv2.resize(crop_bgr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Step 2: Add white border padding so perimeter characters are not clipped by DBNet
        bordered = cv2.copyMakeBorder(crop_bgr, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # Step 3: Convert BGR to Grayscale for luminescence processing
        gray = cv2.cvtColor(bordered, cv2.COLOR_BGR2GRAY)

        # Step 4: CLAHE contrast equalization to pull dark text out of shadows
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # Step 5: Bilateral filter to smooth background noise while preserving character edges
        denoised = cv2.bilateralFilter(contrast_enhanced, d=7, sigmaColor=50, sigmaSpace=50)

        # Step 6: High-pass unsharp masking kernel to sharpen character boundaries
        sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(denoised, -1, sharpening_kernel)

        # Step 7: Convert back to 3-channel BGR format expected by PaddleOCR
        enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr
