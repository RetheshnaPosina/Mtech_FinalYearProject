"""CLIP cosine similarity — baseline signal for CMCD."""
from __future__ import annotations

from PIL import Image


def clip_similarity(image_path: str, text: str) -> float:
    """Return CLIP cosine similarity between image and text.

    Uses the hub's CLIP model (ViT-L/14 by default — improved relationship
    sensitivity over ViT-B/32, see config.py novelty note).

    Parameters
    ----------
    image_path : Path to the image file.
    text       : Text string to compare against the image.

    Returns
    -------
    Cosine similarity in [0, 1]. Returns 0.5 on error (neutral).
    """
    import hallucination_guard.models.model_hub as hub_module
    hub = hub_module.hub
    try:
        img = Image.open(image_path).convert("RGB")
        return hub.clip.similarity(img, text)
    except Exception:
        return 0.5
