"""Classify query as factual/creative/code/opinion for tier routing."""
from __future__ import annotations

import re
from enum import Enum, auto


class QueryType(str, Enum):
    FACTUAL = "factual"
    CREATIVE = "creative"
    CODE = "code"
    OPINION = "opinion"
    UNKNOWN = "unknown"


# Pattern lists — ordered by priority (first match wins in classify())
_CODE = [
    r"\bdef\b",
    r"\bclass\b",
    r"import\s+\w+",
    r"```",
    r"#!/",
    r"\bfunction\b",
    r"<html>",
    r"\bvar\b",
    r"\bconst\b",
]

_CREATIVE = [
    r"\bwrite\s+a\b",
    r"\bstory\b",
    r"\bpoem\b",
    r"\bfiction\b",
    r"\bimagine\b",
]

_OPINION = [
    r"\bdo you think\b",
    r"\bopinion\b",
    r"\bshould\b.*\?",
    r"\bwhat do you\b",
]

_FACTUAL = [
    r"\b\d{4}\b",
    r"\bpercent\b",
    r"\b%\b",
    r"\baccording to\b",
    r"\breported\b",
    r"\bstudies show\b",
    r"\bscientists\b",
    r"\bgovernment\b",
    r"\bmillion\b",
    r"\bbillion\b",
    r"\bresearch\b",
    r"\bstatistics\b",
]


def classify(text: str) -> QueryType:
    """Return the :class:`QueryType` that best describes *text*.

    Priority order: CODE > CREATIVE > OPINION > FACTUAL > UNKNOWN.
    """
    t = text.lower()

    for p in _CODE:
        if re.search(p, t):
            return QueryType.CODE

    for p in _CREATIVE:
        if re.search(p, t):
            return QueryType.CREATIVE

    for p in _OPINION:
        if re.search(p, t):
            return QueryType.OPINION

    factual_hits = sum(1 for p in _FACTUAL if re.search(p, t))
    if factual_hits:
        return QueryType.FACTUAL

    return QueryType.UNKNOWN


# Public alias matching the described interface
def classify_query(text: str) -> dict:
    """Classify *text* and return a feature dict.

    Keys
    ----
    has_number : bool
    has_entity : bool  (approximated by named-entity-like capitalisation)
    has_temporal : bool
    is_citation : bool
    complexity_score : float  (word count normalised to [0, 1])
    query_type : QueryType
    """
    import re as _re

    qtype = classify(text)
    words = text.split()
    has_number = bool(_re.search(r"\b\d[\d,\.]*\b", text))
    has_temporal = bool(
        _re.search(
            r"\b(20\d\d|19\d\d|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
            text,
            _re.IGNORECASE,
        )
    )
    is_citation = bool(
        _re.search(r"\baccording to\b|\bcited\b|\bsource:\b", text, _re.IGNORECASE)
    )
    # Rough entity heuristic: words starting with uppercase that are not the
    # first word and are not all-caps acronyms of length 1.
    has_entity = any(
        w[0].isupper() and not w.isupper()
        for w in words[1:]
        if w.isalpha()
    )
    complexity_score = min(1.0, len(words) / 50.0)

    return {
        "has_number": has_number,
        "has_entity": has_entity,
        "has_temporal": has_temporal,
        "is_citation": is_citation,
        "complexity_score": complexity_score,
        "query_type": qtype,
    }
