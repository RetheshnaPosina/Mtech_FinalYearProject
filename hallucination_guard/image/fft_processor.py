"""FFT spectral analysis — detect periodic artifacts from GAN/diffusion grid patterns."""
from __future__ import annotations

import numpy as np
from PIL import Image

from hallucination_guard.config import settings


def compute_fft(image_path: str) -> dict:
    """Compute FFT on grayscale image. Flag periodic artifacts using 6-sigma threshold.
    Returns: spectral_score, has_periodic.

    GAN-generated images often exhibit periodic artifacts in the frequency domain
    due to upsampling operations (checkerboard patterns). These appear as
    high-energy peaks in the peripheral FFT spectrum beyond the natural roll-off.

    Parameters
    ----------
    image_path : Path to the image file.

    Returns
    -------
    dict with keys:
        spectral_score      : float — normalized peripheral peak strength [0, 1]
        has_periodic        : bool  — True if peaks exceed sigma_threshold
        peripheral_peaks    : int   — count of statistically significant peaks
    """
    try:
        img = Image.open(image_path).convert("L")
        w, h = settings.ela_resize_w, settings.ela_resize_h
        img = img.resize((w, h))

        arr = np.array(img, dtype=np.float64)
        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))

        # Build peripheral mask (exclude 5-pixel center which holds DC component)
        center_y, center_x = h // 2, w // 2
        mask = np.ones_like(magnitude, dtype=bool)
        mask[center_y - 5:center_y + 5, center_x - 5:center_x + 5] = False
        peripheral = magnitude[mask]

        mean_p = np.mean(peripheral)
        std_p = np.std(peripheral)
        threshold = mean_p + settings.fft_sigma_threshold * std_p

        peaks = int(np.sum(peripheral > threshold))
        spectral_score = float(peaks) / max(1, len(peripheral))
        has_periodic = peaks > 0

        return {
            "spectral_score": spectral_score,
            "has_periodic": has_periodic,
            "peripheral_peaks": peaks,
        }

    except Exception as e:
        return {
            "spectral_score": 0.0,
            "has_periodic": False,
            "peripheral_peaks": 0,
            "error": str(e),
        }
