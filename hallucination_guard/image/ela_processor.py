"""Error Level Analysis — detect compression inconsistencies indicating manipulation."""
from __future__ import annotations

import io
import numpy as np
from PIL import Image

from hallucination_guard.config import settings


def compute_ela(image_path: str) -> dict:
    """Re-compress at quality=90, compute pixel difference map.
    Returns: mean_energy, anomaly_regions, dominant_colors.

    ELA (Error Level Analysis) detects regions that were added to or modified
    in an image after original JPEG compression. Re-compressed authentic images
    show uniform error distribution; manipulated regions show elevated residuals.

    Parameters
    ----------
    image_path : Path to the image file.

    Returns
    -------
    dict with keys:
        mean_energy     : float — mean absolute pixel difference (0=authentic, higher=suspicious)
        anomaly_regions : list  — (x, y, dominant_color) tuples for high-energy blocks
        has_anomaly     : bool  — True if mean_energy > 20.0
    """
    try:
        orig = Image.open(image_path).convert("RGB")
        w, h = settings.ela_resize_w, settings.ela_resize_h
        orig = orig.resize((w, h))

        # Re-compress at target quality
        buf = io.BytesIO()
        orig.save(buf, format="JPEG", quality=settings.ela_quality)
        buf.seek(0)
        recompressed = Image.open(buf)

        # Compute pixel difference map
        orig_arr = np.array(orig, dtype=np.float32)
        recomp_arr = np.array(recompressed, dtype=np.float32)
        diff = np.abs(orig_arr - recomp_arr)
        mean_energy = float(np.mean(diff))

        # Find anomaly blocks (8x8 grid, threshold=2.5*mean)
        block_size = 64
        threshold = 2.5
        anomaly_regions = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = diff[y:y+block_size, x:x+block_size]
                if block.size == 0:
                    continue
                if np.mean(block) > threshold * mean_energy:
                    # Detect dominant color channel in this block
                    colors = orig_arr[y:y+block_size, x:x+block_size]
                    ch_names = ["R", "G", "B"]
                    hist = [int(np.mean(colors[:, :, ch])) for ch in range(3)]
                    peak_bin = int(np.argmax(hist))
                    ch_name = ch_names[peak_bin]
                    dominant_val = hist[peak_bin]
                    anomaly_regions.append((x, y, f"{ch_name}:{dominant_val}"))

        return {
            "mean_energy": mean_energy,
            "anomaly_regions": anomaly_regions[:32],
            "has_anomaly": mean_energy > 20.0,
        }

    except Exception as e:
        return {
            "mean_energy": 0.0,
            "anomaly_regions": [],
            "has_anomaly": False,
            "error": str(e),
        }
