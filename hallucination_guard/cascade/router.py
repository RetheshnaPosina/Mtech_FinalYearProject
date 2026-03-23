"""Adaptive risk-based tier selection."""
from __future__ import annotations

from hallucination_guard.sentinel.risk_estimator import RiskEstimate


def route(risk: RiskEstimate, has_image: bool = False) -> int:
    """Return the minimum tier to use for this request."""
    # Only hard-reject to tier 0 when text is invalid AND no image to process
    if not risk.valid and not has_image:
        return 0
    tier = risk.tier
    if has_image and tier < 3:
        tier = 3
    return tier
