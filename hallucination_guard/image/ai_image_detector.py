"""AI-generated image detector — HuggingFace model + heuristic fallback.

Primary: fine-tuned ViT classifier (umm-maybe/AI-image-detector) loaded via
model_hub, cached to ./model_cache/.

Fallback (when model unavailable): multi-signal heuristic approach using:
  1. EXIF metadata absence — AI images have no camera Make/Model/GPS
  2. Pixel channel std deviation — AI images have unnaturally uniform channels
  3. Color histogram entropy — AI images have smoother, less varied histograms

Returns a probability in [0, 1] where:
  1.0 = definitely AI-generated
  0.0 = definitely real photograph
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AIImageResult:
    ai_probability: float        # 0.0 = real, 1.0 = AI-generated
    label: str                   # "artificial" or "real" or "uncertain"
    model_available: bool
    signals: dict                # breakdown of signals for debugging


# ---------------------------------------------------------------------------
# HuggingFace model path
# ---------------------------------------------------------------------------

def _run_hf_model(image_path: str) -> AIImageResult | None:
    """Try the HuggingFace ViT classifier. Returns None on any failure."""
    try:
        from hallucination_guard.models.model_hub import get_ai_detector
        from PIL import Image

        pipeline = get_ai_detector()
        if pipeline is None:
            return None

        img = Image.open(image_path).convert("RGB")
        results = pipeline(img)

        # Pipeline returns list of {"label": ..., "score": ...}
        # Labels from umm-maybe/AI-image-detector: "artificial" / "real"
        scores = {r["label"].lower(): r["score"] for r in results}
        ai_prob = float(scores.get("artificial", 1.0 - scores.get("real", 0.5)))
        ai_prob = round(min(1.0, max(0.0, ai_prob)), 4)
        label = "artificial" if ai_prob > 0.6 else ("real" if ai_prob < 0.4 else "uncertain")

        logger.info(
            "HF AI detector: prob=%.3f label=%s raw=%s",
            ai_prob, label, results,
        )
        return AIImageResult(
            ai_probability=ai_prob,
            label=label,
            model_available=True,
            signals={"hf_scores": scores},
        )
    except Exception as e:
        logger.debug("HF AI detector failed, using heuristics: %s", e)
        return None


# ---------------------------------------------------------------------------
# Heuristic fallback signals
# ---------------------------------------------------------------------------

def _exif_score(image_path: str) -> float:
    """Returns 1.0 if image has NO camera EXIF (AI signal), 0.0 if camera found."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(image_path)
        exif_data = img._getexif() if hasattr(img, "_getexif") else None
        if not exif_data:
            return 0.8  # No EXIF at all — strong AI signal

        tags = {TAGS.get(k, k): v for k, v in exif_data.items()}
        has_camera = bool(tags.get("Make") or tags.get("Model"))
        has_gps = "GPSInfo" in tags
        has_datetime = bool(tags.get("DateTime") or tags.get("DateTimeOriginal"))

        if has_camera:
            return 0.1   # Real camera metadata — likely authentic
        if has_datetime and not has_camera:
            return 0.5   # Has date but no camera — neutral
        return 0.7       # No camera info — likely AI
    except Exception:
        return 0.5  # Cannot read EXIF — neutral


def _pixel_uniformity_score(image_path: str) -> float:
    """AI images have smoother pixel distributions than real photos.
    Returns 1.0 for very uniform (AI-like), 0.0 for noisy (real-like).
    """
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB").resize((128, 128))
        arr = np.array(img, dtype=np.float32)

        # Real photos have higher per-channel std due to sensor noise
        channel_stds = [arr[:, :, c].std() for c in range(3)]
        avg_std = sum(channel_stds) / 3.0

        # Real photos: avg_std typically 40-80; AI images: 20-50
        # Normalize: low std → high AI score
        ai_score = max(0.0, min(1.0, 1.0 - (avg_std - 20.0) / 60.0))
        return round(ai_score, 3)
    except Exception:
        return 0.5


def _color_entropy_score(image_path: str) -> float:
    """Real photos have higher color entropy than AI images.
    Returns 1.0 for low entropy (AI-like), 0.0 for high entropy (real-like).
    """
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("L").resize((128, 128))
        arr = np.array(img, dtype=np.float32)

        # Compute histogram and entropy
        hist, _ = np.histogram(arr.flatten(), bins=64, range=(0, 256))
        hist = hist / hist.sum() + 1e-10
        entropy = -np.sum(hist * np.log2(hist))

        # Max entropy for 64 bins = log2(64) = 6.0
        # Real photos: entropy 4.5-6.0; AI images: often 3.5-5.0
        ai_score = max(0.0, min(1.0, 1.0 - (entropy - 3.0) / 3.0))
        return round(ai_score, 3)
    except Exception:
        return 0.5


def _heuristic_detect(image_path: str) -> AIImageResult:
    """Heuristic-only detection (EXIF + pixel stats)."""
    exif_s = _exif_score(image_path)
    pixel_s = _pixel_uniformity_score(image_path)
    entropy_s = _color_entropy_score(image_path)

    # Weighted combination: EXIF is most reliable signal
    ai_prob = 0.6 * exif_s + 0.2 * pixel_s + 0.2 * entropy_s
    ai_prob = round(min(1.0, max(0.0, ai_prob)), 4)

    label = "artificial" if ai_prob > 0.6 else ("real" if ai_prob < 0.4 else "uncertain")

    signals = {
        "exif_score": exif_s,
        "pixel_uniformity": pixel_s,
        "color_entropy": entropy_s,
    }

    logger.info(
        "Heuristic AI detector: prob=%.3f label=%s exif=%.2f pixel=%.2f entropy=%.2f",
        ai_prob, label, exif_s, pixel_s, entropy_s,
    )

    return AIImageResult(
        ai_probability=ai_prob,
        label=label,
        model_available=False,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_ai_generated(image_path: str) -> AIImageResult:
    """Detect whether an image is AI-generated.

    Tries the HuggingFace ViT classifier first (umm-maybe/AI-image-detector,
    cached to model_cache/). Falls back to EXIF + pixel-statistics heuristics
    if the model is unavailable or fails to load.
    """
    result = _run_hf_model(image_path)
    if result is not None:
        return result
    return _heuristic_detect(image_path)
