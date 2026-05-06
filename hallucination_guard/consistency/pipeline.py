"""Full CMCD pipeline orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from hallucination_guard.consistency.visual_fact_extractor import extract_visual_facts
from hallucination_guard.consistency.text_fact_extractor import extract_text_facts
from hallucination_guard.consistency.contradiction_detector import Contradiction, detect_contradictions
from hallucination_guard.consistency.clip_scorer import clip_similarity
from hallucination_guard.config import settings


@dataclass
class CMCDResult:
    clip_similarity: float
    contradictions: List[Contradiction] = field(default_factory=list)
    cross_modal_trust: float = 1.0
    image_description: str = ""
    matched_elements: List[str] = field(default_factory=list)
    contradiction_count: int = 0
    avg_severity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "clip_similarity": round(self.clip_similarity, 4),
            "contradiction_count": self.contradiction_count,
            "avg_severity": round(self.avg_severity, 4),
            "cross_modal_trust": round(self.cross_modal_trust, 4),
            "image_description": self.image_description,
            "matched_elements": self.matched_elements,
            "contradictions": [
                {
                    "entity": c.entity,
                    "attribute": c.attribute,
                    "text_value": c.text_value,
                    "visual_value": c.visual_value,
                    "severity": round(c.severity, 4),
                }
                for c in self.contradictions
            ],
        }


def run_cmcd(image_path: str, caption: str, blip_caption: str = "") -> CMCDResult:
    """Extract visual + text facts, detect contradictions, compute cross-modal trust.

    CMCD (Cross-Modal Contradiction Detection) pipeline:
    1. Extract visual facts from image (color, scene, caption)
    2. Extract text facts from caption / claim
    3. Detect attribute-level contradictions
    4. Compute CLIP cosine similarity
    5. Fuse into cross_modal_trust score

    Parameters
    ----------
    image_path   : Path to the image file.
    caption      : Text claim or caption to compare against the image.
    blip_caption : Optional pre-generated BLIP caption (passed to VisualFacts).

    Returns
    -------
    CMCDResult with all CMCD signals.
    """
    # 1. Extract visual facts
    visual = extract_visual_facts(image_path, blip_caption=blip_caption)

    # 2. Extract text facts from caption
    text_facts = extract_text_facts(caption)

    # 3. Cross-modal contradiction detection
    contradictions = detect_contradictions(text_facts, visual.structured_facts)

    # 4. CLIP similarity
    clip_sim = clip_similarity(image_path, caption)

    # 5. Build matched elements (values that appear in both text and visual facts)
    text_vals = {f"{f.entity}:{f.attribute}:{f.value}" for f in text_facts}
    visual_vals = {f"{f.entity}:{f.attribute}:{f.value}" for f in visual.structured_facts}
    matched = list(text_vals & visual_vals)

    # 6. Compute cross-modal trust
    avg_severity = sum(c.severity for c in contradictions) / len(contradictions) if contradictions else 0.0
    trust = clip_sim - settings.contradiction_penalty_weight * avg_severity
    trust = max(0.0, min(1.0, trust))

    return CMCDResult(
        clip_similarity=clip_sim,
        contradictions=contradictions,
        cross_modal_trust=trust,
        image_description=visual.caption,
        matched_elements=matched,
        contradiction_count=len(contradictions),
        avg_severity=avg_severity,
    )
