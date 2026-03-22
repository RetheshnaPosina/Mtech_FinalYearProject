"""Adaptive risk-based tier selection."""
from __future__ import annotations

from hallucination_guard.sentinel.risk_estimator import RiskEstimate


def route(risk: RiskEstimate, has_image: bool = False) -> int:
    """Return the minimum tier to use for this request."""
    tier = risk.tier
    if has_image and tier < 2:
        tier = 2
    if not risk.valid:
        tier = 0
    return tier
