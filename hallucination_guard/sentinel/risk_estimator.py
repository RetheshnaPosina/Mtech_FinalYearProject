"""Compute risk score -> tier routing decision (0-3)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from hallucination_guard.sentinel.input_validator import validate
from hallucination_guard.sentinel.query_classifier import classify, QueryType
from hallucination_guard.sentinel.confidence_heuristics import detect_hedging


@dataclass
class RiskEstimate:
    risk_score: float
    tier: int
    reason: str
    query_type: QueryType
    valid: bool = True


def estimate_risk(text: str) -> RiskEstimate:
    """Estimate verification risk and suggest a processing tier.

    Parameters
    ----------
    text:
        The claim or passage to evaluate.

    Returns
    -------
    RiskEstimate
        risk_score in [0, 1], tier in {0, 1, 2, 3}.

    Tier mapping
    ------------
    0  — trivially invalid or non-factual; skip verification
    1  — low-risk; heuristic-only pass
    2  — medium-risk; local model verification
    3  — high-risk; full debate + optional API judge
    """
    # Tier 0: reject invalid input immediately
    v = validate(text)
    if not v.valid:
        return RiskEstimate(
            risk_score=0.0,
            tier=0,
            reason=f"invalid: {v.reason}",
            query_type=QueryType.UNKNOWN,
            valid=False,
        )

    qtype = classify(text)

    # Non-factual queries do not need expensive verification
    if qtype in (QueryType.UNKNOWN, QueryType.CODE, QueryType.CREATIVE):
        return RiskEstimate(
            risk_score=0.1,
            tier=0,
            reason=f"non_factual: {qtype.value}",
            query_type=qtype,
            valid=True,
        )

    hedge = detect_hedging(text)

    # Numeric density (numbers, percentages, magnitudes)
    num_count = len(
        re.findall(
            r"\b\d[\d,\.]*\s*(k|m|b|%|million|billion|thousand|percent)?\b",
            text,
            re.IGNORECASE,
        )
    )

    # Temporal references
    temporal_hits = len(
        re.findall(
            r"\b(20\d\d|19\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
            text,
            re.IGNORECASE,
        )
    )

    # Base risk from query type
    if qtype == QueryType.FACTUAL:
        risk = 0.5
    else:  # OPINION — lower base risk
        risk = 0.3

    # Adjust for numeric complexity
    risk += min(0.3, num_count * 0.06)

    # Adjust for temporal sensitivity
    risk += min(0.2, temporal_hits * 0.05)

    # Hedged language slightly lowers risk (hedges signal acknowledged uncertainty)
    risk -= hedge.hedging_score * 0.1

    risk = max(0.0, min(1.0, risk))

    # Tier assignment
    if risk < 0.35:
        tier = 1
    elif risk < 0.65:
        tier = 2
    else:
        tier = 3

    reason = (
        f"type={qtype.value}"
        f",nums={num_count}"
        f",temporal={temporal_hits}"
        f",hedge={hedge.hedging_score:.2f}"
    )

    return RiskEstimate(
        risk_score=risk,
        tier=tier,
        reason=reason,
        query_type=qtype,
        valid=True,
    )
