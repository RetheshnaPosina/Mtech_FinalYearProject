"""CMCD — Cross-Modal Contradiction Detection.
Attribute-level mismatch detection: same entity + same attribute + different value = CONTRADICTION.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from hallucination_guard.consistency.text_fact_extractor import TextFact


@dataclass
class Contradiction:
    entity: str
    attribute: str
    text_value: str
    visual_value: str
    severity: float


def _embedding_distance(a: str, b: str) -> float:
    """Semantic distance between two short strings using MiniLM sentence model.

    Returns a distance in [0, 1] where 0 = identical meaning, 1 = opposite.
    Falls back to simple word-overlap Jaccard distance when model unavailable.
    """
    import hallucination_guard.models.model_hub as hub_module
    hub = hub_module.hub
    try:
        sim = hub.sentence.similarity(a, b)
        # Convert similarity to distance
        return max(0.0, 1.0 - float(sim))
    except Exception:
        # Fallback: Jaccard distance on word sets
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 1.0
        overlap = len(set_a & set_b) / len(set_a | set_b)
        return 1.0 - overlap


def detect_contradictions(
    text_facts: List[TextFact],
    visual_facts: List[TextFact],
) -> List[Contradiction]:
    """Detect attribute-level contradictions between text and visual facts.

    For every (text_fact, visual_fact) pair that shares the same entity and
    attribute, compute semantic distance between their values.  A pair with
    distance > 0.25 is flagged as a contradiction.

    Parameters
    ----------
    text_facts   : Facts extracted from the text/claim.
    visual_facts : Facts extracted from the image (via VisualFacts.structured_facts).

    Returns
    -------
    List of Contradiction dataclass instances.
    """
    contradictions: List[Contradiction] = []
    for tf in text_facts:
        for vf in visual_facts:
            if tf.entity.lower() != vf.entity.lower():
                continue
            if tf.attribute.lower() != vf.attribute.lower():
                continue
            if tf.value.lower() == vf.value.lower():
                continue
            severity = _embedding_distance(tf.value, vf.value)
            if severity > 0.25:
                contradictions.append(Contradiction(
                    entity=tf.entity,
                    attribute=tf.attribute,
                    text_value=tf.value,
                    visual_value=vf.value,
                    severity=severity,
                ))
    return contradictions
