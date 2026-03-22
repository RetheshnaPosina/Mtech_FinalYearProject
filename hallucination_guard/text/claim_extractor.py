"""Extract atomic verifiable claims using spaCy NER + regex. Deterministic, no LLM."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Fix #13: limit input to prevent memory exhaustion on huge pastes
_MAX_INPUT_CHARS = 10000

# Regex patterns for claim feature detection
_NUM_RE = re.compile(
    r"\b\d[\d,\.]*\s*(%|million|billion|thousand|percent|USD|EUR|GBP|km|mph|kg|lbs|MW|GB|TB)?\b"
)
_TEMPORAL_RE = re.compile(
    r"\b(in\s+\d{4}|on\s+\w+\s+\d{1,2}|since\s+\d{4}|by\s+\d{4}|\d{4}\s*[-\u2013]\s*\d{4}"
    r"|last (year|month|week)|currently|recently|today|as of \d{4})\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"(according to|reported by|study by|published in|cited in|source:)",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


@dataclass
class Claim:
    text: str
    has_number: bool
    has_entity: bool
    has_temporal: bool
    is_citation: bool
    entities: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text)
    return [s.strip() for s in parts if len(s.strip()) >= 12]


def extract_claims(text: str, max_claims: int = 20) -> List[Claim]:
    """Extract verifiable claims from text using spaCy NER + regex.

    Parameters
    ----------
    text       : Input text (truncated to _MAX_INPUT_CHARS chars — Fix #13).
    max_claims : Maximum number of claims to return.

    Returns
    -------
    List of Claim dataclass instances.
    """
    # Fix #13: truncate input to avoid memory issues on large pastes
    text = text[:_MAX_INPUT_CHARS]

    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text.strip()]

    # Try spaCy for NER
    spacy = None
    nlp = None
    use_spacy = False
    try:
        import spacy as _spacy
        spacy = _spacy
        nlp = spacy.load("en_core_web_sm")
        use_spacy = True
    except Exception:
        use_spacy = False

    results: List[Claim] = []
    for sent in sentences:
        if len(results) >= max_claims:
            break
        sent = sent.strip()
        if not sent:
            continue

        nums = [m.group(0) for m in _NUM_RE.finditer(sent)]
        temporal = bool(_TEMPORAL_RE.search(sent))
        citation = bool(_CITATION_RE.search(sent))

        entities: list[str] = []
        if use_spacy and nlp is not None:
            try:
                doc = nlp(sent)
                entities = [ent.text for ent in doc.ents]
            except Exception:
                entities = []

        results.append(Claim(
            text=sent,
            has_number=bool(nums),
            has_entity=bool(entities),
            has_temporal=temporal,
            is_citation=citation,
            entities=entities,
            numbers=nums,
        ))

    return results[:max_claims]
