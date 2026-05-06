"""Watermark detection — visible (edge-density) + invisible (FFT frequency-domain).
Detects both human-visible watermarks in image corners and invisible steganographic
patterns embedded as periodic frequency-domain artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class WatermarkResult:
    watermark_present: bool = False
    watermark_type: str = ""
    watermark_confidence: float = 0.0


def detect_watermark(image_path: str) -> WatermarkResult:
    """Detect visible and frequency-domain watermarks in an image.

    Two detection strategies:
    1. Visible watermark: edge density analysis in corner regions (10% margins).
       High edge density in corners vs centre = visible watermark pattern.
    2. Frequency watermark: FFT outer ring vs inner ring peak ratio.
       AI watermarking tools embed signals as high-frequency periodic patterns.

    Parameters
    ----------
    image_path : Path to the image file.

    Returns
    -------
    WatermarkResult with detection result, type, and confidence score.
    """
    result = WatermarkResult()
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        gray = img.convert("L")
        img_arr = np.array(img)
        gray_arr = np.array(gray).astype(float)

        h, w = gray_arr.shape

        # --- Strategy 1: Visible watermark via corner edge density ---
        margin = max(1, h // 10), max(1, w // 10)
        # Corner region: top-left and bottom-right combined
        corners = np.concatenate([
            gray_arr[:margin[0], :margin[1]].ravel(),
            gray_arr[-margin[0]:, -margin[1]:].ravel(),
        ])
        center = gray_arr[
            margin[0]: h - margin[0],
            margin[1]: w - margin[1],
        ].ravel()

        # Edge proxy: absolute gradient (diff between adjacent pixels)
        corner_edges = float(np.mean(np.abs(np.diff(corners)))) if len(corners) > 1 else 0.0
        center_edges = float(np.mean(np.abs(np.diff(center)))) if len(center) > 1 else 1e-9

        # High corner-to-center edge ratio suggests text/logo overlay in corners
        visible_flag = (corner_edges / (center_edges + 1e-9)) > 4.0
        visible_confidence = min(1.0, corner_edges / (center_edges + 1e-9) / 15.0)

        # --- Strategy 2: Frequency-domain watermark via FFT ---
        fft = np.fft.fft2(gray_arr)
        fft_shift = np.fft.fftshift(fft)
        outer = np.abs(fft_shift)
        cy, cx = h // 2, w // 2
        # Mask out center (DC + low-frequency)
        inner_mask = np.zeros((h, w), dtype=bool)
        inner_r = min(h, w) // 4
        inner_mask[cy - inner_r:cy + inner_r, cx - inner_r:cx + inner_r] = True
        outer_copy = outer.copy()
        outer_copy[inner_mask] = 0
        peak_ratio = float(outer_copy.max() / (outer.max() + 1e-9))
        freq_flag = peak_ratio > 0.3
        freq_confidence = min(1.0, peak_ratio * 2.0)

        # --- Combine signals ---
        if visible_flag and freq_flag:
            result.watermark_present = True
            result.watermark_type = "visible+frequency"
            result.watermark_confidence = round(min(1.0, (visible_confidence + freq_confidence) / 2), 4)
        elif freq_flag:
            result.watermark_present = True
            result.watermark_type = "frequency"
            result.watermark_confidence = round(freq_confidence, 4)
        elif visible_flag and visible_confidence > 30.0:
            result.watermark_present = True
            result.watermark_type = "visible"
            result.watermark_confidence = round(min(1.0, visible_confidence), 4)
        else:
            result.watermark_type = "none"

    except Exception:
        pass

    return result
