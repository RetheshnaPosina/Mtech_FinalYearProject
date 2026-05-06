"""Parse captions into structured {entity, attribute, value} facts."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# Color vocabulary used for scene-level color detection
_COLOR_WORDS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "brown", "grey", "gray", "violet", "gold", "silver",
]

_COUNT_PATTERNS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundreds|thousands|millions|\d+)\s+(\w+s?\b)",
    re.IGNORECASE,
)

_TEMPORAL_RE = re.compile(
    r"\b(in\s+\d{4}|on\s+\w+\s+\d{1,2}|\d{4})\b",
    re.IGNORECASE,
)


@dataclass
class TextFact:
    entity: str
    attribute: str
    value: str
    confidence: float = 0.8


def extract_text_facts(caption: str) -> List[TextFact]:
    """Parse a caption into structured {entity, attribute, value} TextFact records.

    Detects:
    - Colors  → {entity='scene', attribute='color', value=<color>}
    - Counts  → {entity=<object>, attribute='count', value=<number>}
    - Dates   → {entity='event', attribute='time', value=<date>}
    - Indoor/outdoor → {entity='scene', attribute='location', value=indoor|outdoor}
    - People counts  → {entity='people', attribute='count', value=<n>}

    Parameters
    ----------
    caption : Image caption or text to parse.

    Returns
    -------
    List of TextFact dataclass instances.
    """
    facts: List[TextFact] = []
    lower_cap = caption.lower()

    # Color detection
    for color in _COLOR_WORDS:
        if re.search(r"\b" + color + r"\b", lower_cap):
            facts.append(TextFact(entity="scene", attribute="color", value=color))

    # Count patterns (e.g. "three soldiers", "100 protesters")
    for m in _COUNT_PATTERNS.finditer(caption):
        count_word = m.group(1).rstrip("s")
        obj = m.group(2).rstrip("s")
        facts.append(TextFact(entity=obj, attribute="count", value=count_word))

    # Temporal references
    for m in _TEMPORAL_RE.finditer(caption):
        facts.append(TextFact(entity="event", attribute="time", value=m.group(0)))

    # Indoor / outdoor scene
    if any(w in lower_cap for w in ["indoor", "inside", "room", "hall", "office", "kitchen"]):
        facts.append(TextFact(entity="scene", attribute="location", value="indoor"))
    elif any(w in lower_cap for w in ["outdoor", "outside", "street", "park", "field", "sky"]):
        facts.append(TextFact(entity="scene", attribute="location", value="outdoor"))

    # People count pattern
    people_match = re.search(r"\b(\w+)\s+(people|persons|crowd|protesters|soldiers)\b", caption, re.IGNORECASE)
    if people_match:
        facts.append(TextFact(entity="people", attribute="count", value=people_match.group(1)))

    return facts
