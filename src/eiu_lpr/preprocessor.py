# In-memory image preprocessing utilities for license plate crops (pure RGB)
from typing import Tuple
import cv2
import numpy as np

from .exceptions import InvalidInputError


class PlatePreprocessor:
    # Validates that image is a 3-channel uint8 numpy array
    @staticmethod
    def validate_image(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise InvalidInputError("Input image must be a non-null numpy.ndarray.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidInputError(f"Expected 3-channel image array [H, W, 3], received shape {getattr(image, 'shape', None)}.")
        if image.dtype != np.uint8:
            raise InvalidInputError(f"Expected uint8 image array, received dtype {image.dtype}.")

    # Extracts plate crop with proportional 4% padding (clamped between 3 and 12 px)
    @staticmethod
    def crop_plate(
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        padding_percent: float = 0.04,
        min_pad_px: int = 3,
        max_pad_px: int = 12,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)

        pad_w = max(min_pad_px, min(max_pad_px, int(bw * padding_percent)))
        pad_h = max(min_pad_px, min(max_pad_px, int(bh * padding_percent)))

        cx1 = max(0, int(round(x1)) - pad_w)
        cy1 = max(0, int(round(y1)) - pad_h)
        cx2 = min(w, int(round(x2)) + pad_w)
        cy2 = min(h, int(round(y2)) + pad_h)

        if cx2 <= cx1 or cy2 <= cy1:
            return np.empty((0, 0, 3), dtype=image.dtype)

        return image[cy1:cy2, cx1:cx2].copy()

    # Computes Laplacian variance sharpness score (lower indicates blur, < 150 considered blurred)
    @staticmethod
    def compute_blur_score(crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Attenuates specular reflections and sun glare via Retinex luminance normalization
    @staticmethod
    def attenuate_glare(crop_rgb: np.ndarray, glare_threshold: int = 225) -> np.ndarray:
        if crop_rgb is None or crop_rgb.size == 0:
            return crop_rgb

        # Convert RGB to YCrCb to isolate luminance (Y)
        ycrcb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # Extract binary mask of overexposed pixels
        _, glare_mask = cv2.threshold(y, glare_threshold, 255, cv2.THRESH_BINARY)
        glare_ratio = np.sum(glare_mask > 0) / float(y.size)

        # Apply Retinex illumination normalization only if glare is between 0.5% and 40% of crop
        if 0.005 <= glare_ratio <= 0.40:
            kernel_size = max(9, int(crop_rgb.shape[1] * 0.15) | 1)
            illumination = cv2.GaussianBlur(y, (kernel_size, kernel_size), 0)
            y_float = y.astype(np.float32) + 1.0
            illum_float = illumination.astype(np.float32) + 1.0
            normalized_y = np.clip((y_float / illum_float) * 128.0, 0, 255).astype(np.uint8)

            # Soft blend normalized luminance into glare regions
            glare_mask_soft = cv2.GaussianBlur(glare_mask, (7, 7), 0).astype(np.float32) / 255.0
            y_balanced = (normalized_y * glare_mask_soft + y * (1.0 - glare_mask_soft)).astype(np.uint8)
            ycrcb = cv2.merge([y_balanced, cr, cb])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        return crop_rgb

    # Mild contrast and edge enhancement tailored for CCT-XS Vision Transformer (RGB)
    @staticmethod
    def mild_enhance_for_transformer(crop_rgb: np.ndarray) -> np.ndarray:
        if crop_rgb is None or crop_rgb.size == 0:
            return crop_rgb

        # Attenuate localized specular reflections
        enhanced = PlatePreprocessor.attenuate_glare(crop_rgb)

        # Upscale small crops to ensure stroke pixel density matches CCT-XS tensor (128x64)
        h, w = enhanced.shape[:2]
        if w < 128 or h < 64:
            scale = max(128.0 / max(1, w), 64.0 / max(1, h))
            enhanced = cv2.resize(enhanced, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Mild CLAHE contrast balancing (clipLimit 1.5)
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # Subtle morphological stroke reinforcement (2x2 kernel, 1 pass)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thickened = cv2.erode(contrast, kernel, iterations=1)

        # Mild unsharp mask to sharpen stroke boundaries
        blurred = cv2.GaussianBlur(thickened, (3, 3), 0)
        sharpened = cv2.addWeighted(thickened, 1.3, blurred, -0.3, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
