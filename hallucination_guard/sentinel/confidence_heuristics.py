"""Detect hedging language that separates linguistic confidence from factual accuracy."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Patterns that indicate epistemic uncertainty / hedging
_HEDGES = [
    r"\bstudies (show|suggest|indicate)\b",
    r"\bexperts (say|believe|claim)\b",
    r"\bsome (people|sources|reports)\b",
    r"\bapparently\b",
    r"\bit is (believed|said|reported) that\b",
    r"\baccording to (some|many|various)\b",
    r"\bmany (people|experts|scientists)\b",
    r"\bwidely (known|accepted|believed)\b",
]

# Patterns that signal false confidence ("scientifically proven", "100%", etc.)
_CONFIDENT_FALSE = [
    r"\bscientifically proven\b",
    r"\bdefinitively\b",
    r"\b100%\b",
    r"\bguaranteed\b",
    r"\babsolutely (true|false|certain)\b",
]


@dataclass
class HedgeResult:
    hedge_count: int
    confident_falsehood_count: int
    hedging_score: float
    patterns_found: list[str] = field(default_factory=list)


def detect_hedging(text: str) -> HedgeResult:
    """Analyse *text* for hedging and overconfident-false patterns.

    Returns a :class:`HedgeResult` with a normalised hedging score in [0, 1].
    Higher scores indicate more hedged / uncertain language.
    """
    t = text.lower()
    found: list[str] = []
    hedge_count = 0

    for p in _HEDGES:
        hits = re.findall(p, t)
        if hits:
            hedge_count += len(hits)
            found.append(p)

    conf_count = sum(1 for p in _CONFIDENT_FALSE if re.search(p, t))
    words = max(len(t.split()), 1)
    score = min(
        1.0,
        (hedge_count * 0.15 + conf_count * 0.1) / (words / 50),
    )

    return HedgeResult(
        hedge_count=hedge_count,
        confident_falsehood_count=conf_count,
        hedging_score=score,
        patterns_found=found,
    )


# Public alias expected by risk_estimator
def compute_heuristic_confidence(claim_text: str) -> float:
    """Return a heuristic confidence score in [0, 1].

    A lower score means the claim is more hedged (less confident).
    1.0 - hedging_score is used so that well-hedged text lowers confidence.
    """
    result = detect_hedging(claim_text)
    return max(0.0, 1.0 - result.hedging_score)
