"""Extract visual facts from image: BLIP caption + color histogram + scene + object count."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

from PIL import Image

from hallucination_guard.consistency.text_fact_extractor import TextFact, extract_text_facts


# Color map: name -> (low_rgb, high_rgb) tuples in [0, 255]
_COLOR_MAP = {
    "red":    ([150, 0, 0],   [255, 80, 80]),
    "green":  ([0, 100, 0],   [80, 200, 80]),
    "blue":   ([0, 0, 100],   [80, 80, 255]),
    "yellow": ([180, 180, 0], [255, 255, 100]),
    "orange": ([180, 80, 0],  [255, 160, 80]),
    "white":  ([200, 200, 200], [255, 255, 255]),
    "black":  ([0, 0, 0],     [60, 60, 60]),
    "gray":   ([80, 80, 80],  [160, 160, 160]),
    "brown":  ([80, 40, 0],   [160, 100, 60]),
    "purple": ([80, 0, 80],   [180, 80, 180]),
}


@dataclass
class VisualFacts:
    caption: str
    dominant_colors: list[str] = field(default_factory=list)
    scene_type: str = "unknown"
    object_count_estimate: int = 0
    structured_facts: list[TextFact] = field(default_factory=list)


def _detect_dominant_colors(image: Image.Image) -> List[str]:
    """Detect dominant colors from a PIL image using mean RGB analysis."""
    arr = np.array(image.resize((64, 64))).astype(np.float32)
    found: List[str] = []
    for color_name, (low, high) in _COLOR_MAP.items():
        low_arr = np.array(low)
        high_arr = np.array(high)
        mask = np.all((arr >= low_arr) & (arr <= high_arr), axis=-1)
        if mask.mean() > 0.08:
            found.append(color_name)
    return found


def _detect_scene(image: Image.Image) -> str:
    """Heuristic scene type detection: outdoor / indoor / unknown."""
    arr = np.array(image.resize((64, 64))).astype(np.float32) / 255.0
    # Sky region: top 1/8 rows — high blue channel = outdoor
    sky_region = arr[:8, :, :]
    # Green region: middle rows — high green channel = outdoor vegetation
    green_region = arr[16:48, :, :]
    brightness = np.mean(arr)

    if sky_region[:, :, 2].mean() > 0.55:  # strong blue sky
        return "outdoor"
    if green_region[:, :, 1].mean() > 0.45 and green_region[:, :, 0].mean() < 0.35:
        return "outdoor"
    if brightness < 0.35:
        return "indoor"
    return "unknown"


def extract_visual_facts(image_path: str, blip_caption: str = "") -> VisualFacts:
    """Extract structured visual facts from an image file + optional BLIP caption.

    Uses:
    - PIL for color histogram and scene analysis
    - BLIP model (via hub) for caption generation if blip_caption not provided
    - text_fact_extractor for structured fact parsing of the caption

    Parameters
    ----------
    image_path   : Path to the image file.
    blip_caption : Optional pre-generated BLIP caption.

    Returns
    -------
    VisualFacts dataclass with caption, dominant_colors, scene_type,
    object_count_estimate, and structured_facts.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return VisualFacts(caption=blip_caption)

    # Generate caption via BLIP if not provided
    if not blip_caption:
        try:
            import hallucination_guard.models.model_hub as hub_module
            hub = hub_module.hub
            blip_caption = hub.blip.caption(img)
        except Exception:
            blip_caption = ""

    dominant_colors = _detect_dominant_colors(img)
    scene_type = _detect_scene(img)

    # Parse caption into structured facts
    structured: List[TextFact] = extract_text_facts(blip_caption)

    # Add visual-derived facts
    for color in dominant_colors:
        structured.append(TextFact(entity="scene", attribute="color", value=color, confidence=0.8))
    if scene_type != "unknown":
        structured.append(TextFact(entity="scene", attribute="location", value=scene_type, confidence=0.7))

    return VisualFacts(
        caption=blip_caption,
        dominant_colors=dominant_colors,
        scene_type=scene_type,
        object_count_estimate=0,  # object count from Florence OD is handled separately
        structured_facts=structured,
    )
