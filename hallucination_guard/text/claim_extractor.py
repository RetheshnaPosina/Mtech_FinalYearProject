"""Extract atomic verifiable claims using spaCy NER + regex. Deterministic, no LLM."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Fix #13: limit input to prevent memory exhaustion on huge pastes
_MAX_INPUT_CHARS = 10000

# --- OCR artifact stripping ---
# Patterns that appear in phone-screenshot OCR but carry no factual content.
# Applied in order before sentence splitting.
# Note: no trailing \b — OCR often glues UI tokens to adjacent words.
_OCR_STRIP_PATTERNS = [
    # 1. Status-bar time at very start: ":45" or "9:45"
    re.compile(r'^\s*\d{0,2}:\d{2}\s*'),
    # 2. Twitter/app UI buttons
    re.compile(r'Add\s+languages', re.IGNORECASE),
    re.compile(r'Show\s+More', re.IGNORECASE),
    # 3. Pinned tweet UI footer
    re.compile(r'[.·]?\s*\+?\s*Pinned\s+people\s+follow.*$', re.IGNORECASE),
    # 4. "From domain.com123" metadata prefix — MUST run before @handle pattern
    #    so the domain's "." doesn't block the source-header regex below.
    re.compile(r'^\s*From\s+\S+\s*', re.IGNORECASE),
    # 5. File size artifacts: "173KB", "₹ 173KB" — run before @handle pattern
    re.compile(r'\d*\s*[\u20b9$€£]?\s*\d+\s*[KMG]B\w*', re.IGNORECASE),
    # 6. Time-ago artifact: "6hX", "2dX", "3mX" (no trailing \b — fused e.g. "6hXTrump")
    re.compile(r'\d+[hmd]X', re.IGNORECASE),
    # 7. Engagement stats: "106T2 9697411" (retweet-index + like count)
    re.compile(r'\d+T\d+\s+[\d,]+'),
    # 8. Social media source header: "Name @handle . " — runs after "From" and file-size
    #    are stripped so no domain "." blocks the [^.!?]* match.
    re.compile(r'^[^.!?]*@\w[\w\s]*\.\s*', re.IGNORECASE),
    # 9. Box office table separators: "Day ~", "~℮", "~₹+"
    re.compile(r'\bDay\s*[~\u212e\u2179]+\s*', re.IGNORECASE),
    re.compile(r'[~\u212e\u2179]+\s*[\u20b9$€£]\+?\s*'),
    # 10. Engagement footer fused with content: "...moreWORD31℃ №.6K"
    re.compile(r'\.{2,}\s*more[A-Za-z0-9\u2103\u2116.\s]*$', re.IGNORECASE),
    # 11. Unicode metric/engagement symbols: ℃ № in social counts
    re.compile(r'\d+[\u2103\u2109°]\s*[\u2116#]\s*[\d.]+[KkMm]?\w*'),
    # 12. Remaining unicode separator symbols used as table/list dividers
    re.compile(r'[\u212e\u2179\u2103\u2116~]+'),
    # 13. Orphaned currency symbol with no preceding number: "₹+" alone
    re.compile(r'(?<!\d)\s*[\u20b9$€£]\+?\s*(?=[A-Za-z\s]|$)'),
    # 14. Orphaned trailing punctuation artifacts (e.g. ".+", "·+")
    re.compile(r'[\s.·+]+$'),
]


def _clean_ocr_text(text: str) -> str:
    """Strip common phone-screenshot OCR artifacts that are not factual claims."""
    for pat in _OCR_STRIP_PATTERNS:
        text = pat.sub(' ', text)

    # --- Watermark: repeated-word pair e.g. "fukkardFUKKARD" ---
    # Remove fused lowercase+UPPERCASE repeat before splitting camelCase,
    # so "fukkardFUKKARDPM" → "PM" (not "fukkard FUKKARDPM" which breaks later steps).
    text = re.sub(r'([a-z]{3,})\1', '', text, flags=re.IGNORECASE)

    # --- CamelCase split (remaining lower→UPPER boundaries) ---
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # --- Number+word OCR fusions: "30Day"→"30 Day", "1365crore"→"1365 crore" ---
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)

    # --- ALL-CAPS word fusions: insert space before common fused-in English words ---
    # "INTERACTWITH" → "INTERACT WITH"
    _CAPS_WORDS = r'(?:WITH|VIA|AND\b|THE\b|FOR\b|FROM\b|REGARDING|RESPONSE|PLANS|STATES|WILL)'
    text = re.sub(rf'([A-Z])({_CAPS_WORDS})', r'\1 \2', text)
    # "EVENINGTO" → "EVENING TO"  (only when TO is 5+ chars in: avoids PHOTO, AUTO)
    text = re.sub(r'([A-Z]{4,})(TO)(?=\s|[A-Z]|$)', r'\1 \2', text)

    # --- Strip trailing watermark word after sentence end: ". . Fukkcard" ---
    text = re.sub(r'[\s.·]+[A-Z][a-z]{3,}\w*\s*$', '', text)

    # --- Collapse spaces ---
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    return text


def _is_meaningful_claim(text: str) -> bool:
    """Return True only if text has enough real words to be a searchable claim.

    Filters out pure UI artifacts and symbol-heavy strings left after OCR cleaning
    that would produce useless search queries.
    """
    words = re.findall(r'[a-zA-Z]{3,}', text)
    return len(words) >= 3


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
# Standard split: period/!/? followed by whitespace + capital
# Also handles OCR-fused sentences: "open.Italy" (no space between sentences)
_SENT_SPLIT = re.compile(r"(?<=[.!?])(?:\s+|(?=[A-Z]))(?=[A-Z\"'])")


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

    # Strip OCR UI artifacts before sentence splitting
    text = _clean_ocr_text(text)

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

        if not _is_meaningful_claim(sent):
            continue

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
