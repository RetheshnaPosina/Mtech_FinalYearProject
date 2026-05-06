"""Per-claim CLIP scoring — scores each extracted claim against the image.
Returns a list of {claim, clip_score} dicts for downstream analysis.
"""
from __future__ import annotations

from typing import List, Dict

from hallucination_guard.consistency.clip_scorer import clip_similarity


def score_claims_against_image(image_path: str, claims: List[str]) -> List[Dict]:
    """Score each claim against the image using CLIP cosine similarity.

    Parameters
    ----------
    image_path : Path to the image file.
    claims     : List of claim text strings to evaluate.

    Returns
    -------
    List of dicts with keys: 'claim' (str), 'clip_score' (float).
    """
    results: List[Dict] = []
    for claim in claims:
        score = clip_similarity(image_path, claim)
        results.append({
            "claim": claim,
            "clip_score": round(score, 4),
        })
    return results
