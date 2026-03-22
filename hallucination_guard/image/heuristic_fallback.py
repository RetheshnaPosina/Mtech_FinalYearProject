"""Heuristic fallback when CNN model is not available. Uses ELA + FFT only."""
from __future__ import annotations

from hallucination_guard.image.ela_processor import compute_ela
from hallucination_guard.image.fft_processor import compute_fft
from hallucination_guard.config import settings


def heuristic_score(image_path: str) -> float:
    """Return AI-generated probability using ELA+FFT heuristics only.

    Used when the trained CNN model is unavailable (no GPU, missing weights).
    Combines ELA mean_energy and FFT spectral_score using configured weights.

    Parameters
    ----------
    image_path : Path to the image file.

    Returns
    -------
    float in [0, 1] — probability image is AI-generated or manipulated.
    """
    ela = compute_ela(image_path)
    fft = compute_fft(image_path)

    ela_score = min(1.0, ela.get("mean_energy", 0.0) / 20.0)
    fft_score = fft.get("spectral_score", 0.0)

    total_w = settings.ela_weight + settings.fft_weight
    fusion = (
        settings.ela_weight * ela_score +
        settings.fft_weight * fft_score
    ) / max(total_w, 1e-6)

    return float(min(1.0, max(0.0, fusion)))
