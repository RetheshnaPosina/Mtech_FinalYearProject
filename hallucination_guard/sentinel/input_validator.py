"""Tier 0: Reject garbage inputs before any model touches them."""
from __future__ import annotations

import re
from dataclasses import dataclass

# Detects repeated substring patterns (e.g., "abcabc..." repeated 4+ times)
_REPETITION_RE = re.compile(r"(.{3,})\1{4,}")
_MIN_WORDS: int = 3
_MAX_CHARS: int = 8000


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""


def validate(text: str) -> ValidationResult:
    """Validate *text* and return a :class:`ValidationResult`.

    Checks performed (in order):
    1. Non-empty after stripping whitespace.
    2. Total length <= _MAX_CHARS.
    3. Minimum word count >= _MIN_WORDS.
    4. Vocabulary ratio >= 0.15 (not excessively repetitive at word level).
    5. No character-level repetition pattern.
    6. ASCII spam ratio <= 0.80.
    """
    text = text.strip()
    if not text:
        return ValidationResult(valid=False, reason="empty_input")

    if len(text) > _MAX_CHARS:
        return ValidationResult(valid=False, reason="input_too_long")

    words = text.split()
    if len(words) < _MIN_WORDS:
        return ValidationResult(valid=False, reason="too_short")

    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    if unique_ratio < 0.15:
        return ValidationResult(valid=False, reason="low_vocabulary_ratio")

    if _REPETITION_RE.search(text):
        return ValidationResult(valid=False, reason="repetitive_content")

    non_ascii = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
    if non_ascii > 0.8:
        return ValidationResult(valid=False, reason="non_ascii_spam")

    return ValidationResult(valid=True)


# Public interface used by other modules
def validate_text(text: str) -> None:
    """Raise :exc:`ValueError` if *text* is invalid."""
    result = validate(text)
    if not result.valid:
        raise ValueError(f"Invalid input text: {result.reason}")


def validate_image_path(path: str) -> None:
    """Raise :exc:`ValueError` if the image path looks invalid."""
    import os

    if not path or not path.strip():
        raise ValueError("Image path must not be empty.")
    if not os.path.exists(path):
        raise ValueError(f"Image file not found: {path}")
