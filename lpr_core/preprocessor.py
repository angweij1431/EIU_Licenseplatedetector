import cv2
import numpy as np

# Computer Vision Preprocessing Pipeline for License Plate OCR
class PlatePreprocessor:
    @staticmethod
    def compute_blur_score(crop_bgr: np.ndarray) -> float:
        """Computes Laplacian variance sharpness score (lower = blurrier, < 150 = blurred)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

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

        # Step 6: Morphological reconstruction to close faint or broken loops on digits like 8, 0, B
        # In a light background plate, text is dark. Morphological erosion on grayscale darkens and thickens strokes.
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thickened = cv2.erode(denoised, morph_kernel, iterations=1)

        # Step 7: High-pass unsharp masking kernel to sharpen character boundaries
        sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(thickened, -1, sharpening_kernel)

        # Step 8: Convert back to 3-channel BGR format expected by PaddleOCR
        enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        return enhanced_bgr

    # adding a glare filter
    @staticmethod
    def attenuate_glare(crop_bgr: np.ndarray, glare_threshold: int = 225) -> np.ndarray:
        """
        Attenuates harsh specular reflections and tropical sun glare on license plate surfaces
        by estimating localized background illumination and normalizing luminance.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        # Convert to YCrCb to isolate pure luminance (Y)

        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Detect specular glare mask (overexposed regions above glare_threshold)

        _, glare_mask = cv2.threshold(y, glare_threshold, 255, cv2.THRESH_BINARY)
        glare_ratio = np.sum(glare_mask > 0) / float(y.size)
        # Apply attenuation only if localized glare is detected (between 0.5% and 40% of crop)

        if 0.005 <= glare_ratio <= 0.40:
            # Estimate background illumination using localized Gaussian blur
            kernel_size = max(9, int(crop_bgr.shape[1] * 0.15) | 1)
            illumination = cv2.GaussianBlur(y, (kernel_size, kernel_size), 0)
            # Retinex illumination division: compresses specular hot spots back down
            y_float = y.astype(np.float32) + 1.0
            illum_float = illumination.astype(np.float32) + 1.0
            normalized_y = np.clip((y_float / illum_float) * 128.0, 0, 255).astype(np.uint8)
            # Softly blend normalized luminance specifically inside glare regions
            glare_mask_soft = cv2.GaussianBlur(glare_mask, (7, 7), 0).astype(np.float32) / 255.0
            y_balanced = (normalized_y * glare_mask_soft + y * (1.0 - glare_mask_soft)).astype(np.uint8)
            ycrcb = cv2.merge([y_balanced, cr, cb])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return crop_bgr

    @staticmethod
    def mild_enhance_for_transformer(crop_bgr: np.ndarray) -> np.ndarray: 
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        # Step 0: Attenuate localized specular reflections and sun glare
        crop_bgr = PlatePreprocessor.attenuate_glare(crop_bgr) # the glare is inside of this fucntion 
        # Step 1: Upscale small crops so strokes have enough pixel density (native CCT-XS size)
        h, w = crop_bgr.shape[:2]
        if w < 128 or h < 64:
            scale = max(128.0 / max(1, w), 64.0 / max(1, h))
            crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Step 2: Mild CLAHE for shadow and contrast balancing (clipLimit 1.5, zero white borders)

        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

        contrast = clahe.apply(gray)

        # Step 3: Subtle morphological stroke reinforcement (2x2 kernel, 1 pass)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thickened = cv2.erode(contrast, kernel, iterations=1)

        # Step 4: Mild unsharp mask to crisp up edges without halo artifacts
        blurred = cv2.GaussianBlur(thickened, (3, 3), 0)
        sharpened = cv2.addWeighted(thickened, 1.3, blurred, -0.3, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)