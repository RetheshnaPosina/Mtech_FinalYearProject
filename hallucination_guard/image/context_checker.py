"""Out-of-Context Image Detection — COSMOS/COVE inspired.
Detects when a real image is used with a false/misleading caption or context.
Reference: COSMOS (2021), COVE NAACL 2025, AVerImaTeC NeurIPS 2025.

Two signals:
  1. CLIP cross-modal similarity: image embedding vs OCR/caption text
     -> very low similarity = image doesn't match what text claims
  2. Entity mismatch: named entities in OCR text vs scene objects
     -> "Tesla" in text but image shows "Toyota" in OD labels
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ContextCheckResult:
    clip_score: float = 0.5
    entity_overlap: float = 1.0
    is_out_of_context: bool = False
    mismatched_entities: List[str] = field(default_factory=list)
    context_trust: float = 1.0
    reasoning: str = ""


# Brand/entity recognition pattern (common entities in news/social media)
_BRAND_RE = re.compile(
    r"\b(Apple|Google|Microsoft|Tesla|Amazon|Facebook|Meta|Twitter|Samsung|Nike|BMW|Toyota|Ford"
    r"|Pfizer|Moderna|NASA|WHO|UN|FBI|CIA"
    r"|India|China|USA|UK|Russia|Ukraine|Israel|Iran|Pakistan"
    r"|Biden|Trump|Modi)\b",
    re.IGNORECASE,
)


def check_context(
    image_path: str,
    ocr_text: str,
    caption: str,
    detected_objects: List[str],
    dense_captions: List[str],
) -> ContextCheckResult:
    """Check if image is used in correct context.

    Parameters
    ----------
    image_path       : Path to the image file.
    ocr_text         : Text extracted from the image via OCR.
    caption          : Text claim or caption being verified.
    detected_objects : Object labels from Florence OD.
    dense_captions   : Dense region captions from Florence DRC.

    Returns
    -------
    ContextCheckResult with context trust score and mismatch details.
    """
    result = ContextCheckResult()

    # Combine all visual text signals for CLIP comparison
    visual_text = " ".join([ocr_text, " ".join(dense_captions[:3])]).strip()
    claim_text = caption.strip()[:200]

    if not visual_text and not claim_text:
        result.reasoning = "no_text_to_compare"
        return result

    # 1. CLIP similarity between claim text and image
    try:
        from hallucination_guard.consistency.clip_scorer import clip_similarity as _clip
        clip_s = _clip(image_path, claim_text)
        result.clip_score = clip_s
    except Exception:
        clip_s = 0.5
        result.clip_score = 0.5

    # 2. Entity mismatch analysis
    text_entities = set(
        m.group(0).lower()
        for m in _BRAND_RE.finditer(claim_text + " " + caption)
    )
    visual_entities = set(
        m.group(0).lower()
        for m in _BRAND_RE.finditer(visual_text + " " + " ".join(detected_objects))
    )

    if text_entities and visual_entities:
        overlap = len(text_entities & visual_entities) / max(len(text_entities), len(visual_entities))
    elif not text_entities:
        overlap = 1.0  # no entities to mismatch
    else:
        overlap = 0.0  # entities in text not found visually
    result.entity_overlap = overlap

    # Identify specifically mismatched entities
    result.mismatched_entities = list(text_entities - visual_entities)

    # 3. Out-of-context decision
    clip_flag = clip_s < 0.15
    entity_flag = overlap < 0.2 and bool(text_entities)

    # 4. Context trust score
    clip_trust = clip_s / 0.6 if clip_s < 0.6 else 1.0
    entity_trust = min(1.0, overlap + 0.4)
    result.context_trust = min(1.0, max(0.0, (clip_trust + entity_trust) / 2.0))
    result.is_out_of_context = clip_flag or entity_flag

    result.reasoning = (
        f"clip={clip_s:.3f}, entity_overlap={overlap:.2f}, "
        f"mismatched={result.mismatched_entities}, "
        f"out_of_context={result.is_out_of_context}"
    )
    return result
