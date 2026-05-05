"""Adversarial hypothesis generation for the Prosecutor agent.
Pure algorithmic — zero LLM, zero hallucination risk. AWP core.

Novelty improvement: STRATEGIES registry + exclude_strategies parameter
-----------------------------------------------------------------------
The original generate_adversarial() ran all 5 strategies unconditionally.
This made the AWP score impossible to ablate — there was no way to measure
each strategy's individual contribution to correct verdicts.

This version adds:
  STRATEGIES : dict[str, callable]
      Registry mapping strategy name → generator function.
      The ablation script can inspect and selectively exclude strategies.

  generate_adversarial(claim, exclude_strategies=None)
      When exclude_strategies={'numeric_alt'}, that strategy is skipped.
      The ablation benchmark (benchmarks/run_ablation.py) uses this to
      produce the per-strategy contribution table.

Strategies (unchanged from original)
--------------------------------------
  negation       : grammatical negation of the claim
  numeric_alt    : perturb numeric values (×1.1, ×0.9, ×2.0, ×0.5)
  temporal_alt   : substitute time references (formerly/previously/years ago)
  entity_swap    : swap named entities with disambiguation query
  citation_check : verify the claim's cited source actually exists
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Lazy import to avoid circular dependencies
# claim_extractor is only available via pyc; import deferred to function body


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdversarialHypothesis:
    text: str
    strategy: str
    search_query: str


# ---------------------------------------------------------------------------
# Negation helpers
# ---------------------------------------------------------------------------

_NEGATION_MAP: Dict[str, str] = {
    "is":     "is not",
    "are":    "are not",
    "was":    "was not",
    "were":   "were not",
    "has":    "has not",
    "have":   "have not",
    "did":    "did not",
    "does":   "does not",
    "will":   "will not",
    "can":    "cannot",
    "could":  "could not",
    "should": "should not",
}

_NEG_PATTERN = re.compile(r"\b(" + "|".join(_NEGATION_MAP.keys()) + r")\b")


def _apply_negation(text: str) -> str:
    """Negate the first matching auxiliary verb in the text."""
    def replacer(m: re.Match) -> str:
        return _NEGATION_MAP[m.group(0)]
    result, n = _NEG_PATTERN.subn(replacer, text, count=1)
    if n == 0:
        result = "NOT " + text
    return result


# ---------------------------------------------------------------------------
# Temporal substitution helpers
# ---------------------------------------------------------------------------

_KEYWORD_REPLACEMENTS = {
    "current":  "former",
    "now":      "formerly",
    "today":    "previously",
    "latest":   "outdated",
    "recent":   "outdated",
    "still":    "no longer",
    "anymore":  "formerly",
}


def _apply_temporal_shift(text: str):
    """Replace time references with staleness markers.

    Returns the modified string, or None if no temporal signal found.
    None signals _strategy_temporal_alt to return [] (emit nothing).

    Tier 1 — year substitution:  "2020" → "2019"
    Tier 2 — keyword substitution: "current" → "former"
    Tier 3 — no signal → None (do NOT emit a broken hypothesis)
    """
    # ── Tier 1: year substitution ────────────────────────────────────────
    _TEMPORAL_SUBS = {
        "2023": ["2022", "2024"],
        "2022": ["2021", "2023"],
        "2021": ["2020", "2022"],
        "2020": ["2019", "2021"],
        "2019": ["2018", "2020"],
        "2018": ["2017", "2019"],
    }
    for year, alts in _TEMPORAL_SUBS.items():
        if year in text:
            return text.replace(year, alts[0], 1)

    # ── Tier 2: keyword substitution ────────────────────────────────────
    _TEMPORAL_KEYWORDS_PATTERN = re.compile(
        r"\b(current|now|today|latest|recent|still|anymore"
        r"|as of \d{4}|january|february|march|april|may|june"
        r"|july|august|september|october|november|december)\b",
        re.IGNORECASE,
    )

    def _keyword_replacer(m: re.Match) -> str:
        word = m.group(0).lower()
        return _KEYWORD_REPLACEMENTS.get(word, "formerly")

    result, n = _TEMPORAL_KEYWORDS_PATTERN.subn(_keyword_replacer, text, count=1)
    if n > 0:
        return result[0].upper() + result[1:] if result else result

    # ── Tier 3: no temporal signal — emit nothing ────────────────────────
    return None


# ---------------------------------------------------------------------------
# Numeric perturbation
# ---------------------------------------------------------------------------

_NUM_PATTERN = re.compile(r"[\d,]+(?:\.\d+)?")


def _perturb_number(num_str: str, original: str) -> List[str]:
    """Return list of perturbed number strings for a numeric claim."""
    clean = re.sub(r"[,%]", "", num_str)
    try:
        val = float(clean)
    except ValueError:
        return []

    results = []
    for mult, fmt in [(1.1, ",.0f"), (0.9, ",.0f"), (2.0, ",.0f"), (0.5, ".1f")]:
        try:
            perturbed = format(val * mult, fmt)
            if "%" in num_str:
                perturbed += "%"
            results.append(original.replace(num_str, perturbed, 1))
        except (ValueError, OverflowError):
            pass
    return results[:2]  # top 2 perturbations per number


# ---------------------------------------------------------------------------
# Strategy generators
# ---------------------------------------------------------------------------

def _strategy_negation(claim_text: str, claim_obj) -> List[AdversarialHypothesis]:
    hyp = _apply_negation(claim_text)
    if hyp == claim_text:
        return []
    return [AdversarialHypothesis(
        text=hyp,
        strategy="negation",
        search_query="evidence against: " + claim_text[:90],
    )]


def _strategy_numeric_alt(claim_text: str, claim_obj) -> List[AdversarialHypothesis]:
    if not getattr(claim_obj, "has_number", False):
        return []
    hypotheses = []
    for m in _NUM_PATTERN.finditer(claim_text):
        num_str = m.group(0)
        for alt_text in _perturb_number(num_str, claim_text):
            hypotheses.append(AdversarialHypothesis(
                text=alt_text,
                strategy="numeric_alt",
                search_query="alternative number evidence: " + alt_text[:90],
            ))
        if len(hypotheses) >= 2:
            break
    return hypotheses


def _strategy_temporal_alt(claim_text: str, claim_obj) -> List[AdversarialHypothesis]:
    if not getattr(claim_obj, "has_temporal", False):
        return []
    hyp = _apply_temporal_shift(claim_text)
    if hyp is None:
        # No year or temporal keyword found — do not emit a broken hypothesis
        return []
    return [AdversarialHypothesis(
        text=hyp,
        strategy="temporal_alt",
        search_query="current status check: " + claim_text[:80],
    )]


def _is_valid_entity(ent: str) -> bool:
    """Return True only if ent looks like a meaningful named entity.

    Rejects tokens that come from noisy OCR or scraped HTML, e.g.:
        "Show More117T2 9697411 .+Pinned. people follow"
        "9697411"
        ".+Pinned"

    Rules (all must pass):
    - At least 2 characters long after stripping whitespace.
    - At least 40% of characters are ASCII letters (filters pure-numeric
      and symbol-heavy tokens).
    - Does not contain regex/glob metacharacters (. + * ? [ ] { }).
    - Does not start with a digit (rejects bare numbers like "9697411").
    """
    ent = ent.strip()
    if len(ent) < 2:
        return False
    if ent[0].isdigit():
        return False
    alpha_ratio = sum(c.isalpha() for c in ent) / len(ent)
    if alpha_ratio < 0.4:
        return False
    if re.search(r'[.+*?\[\]{}]', ent):
        return False
    return True


def _strategy_entity_swap(claim_text: str, claim_obj) -> List[AdversarialHypothesis]:
    entities = getattr(claim_obj, "entities", [])
    if not entities:
        return []
    hypotheses = []
    for ent in entities[:2]:
        # Skip garbage tokens produced by NER on noisy OCR or scraped input.
        if not _is_valid_entity(ent):
            continue
        # Generate a genuinely adversarial hypothesis by replacing the entity
        # with "someone/something other than <entity>".  The old format
        # f'"{ent}": ' + claim_text was semantically near-identical to the
        # original claim, so NLI scored it with HIGH entailment → inflated
        # best_alt_support → adv_score artificially low → all claims REFUTED.
        if ent in claim_text:
            hyp = claim_text.replace(ent, f"someone other than {ent}", 1)
        else:
            hyp = f"The attribution of this claim to {ent} is incorrect."
        hypotheses.append(AdversarialHypothesis(
            text=hyp,
            strategy="entity_swap",
            search_query=f'"{ent}" correct facts disambiguation',
        ))
    return hypotheses


def _extract_source(claim_text: str) -> str:
    """Extract the cited source name from a citation claim.

    Handles forms like:
        "According to WHO, ..."          → "WHO"
        "According to a 2023 MIT study"  → "MIT"
        "Published in Nature, ..."       → "Nature"
        "Per the CDC report, ..."        → "CDC"
    """
    text = claim_text.strip()

    triggers = [
        r"according to\s+",
        r"as (?:stated|reported|shown) (?:by|in)\s+",
        r"cite?d? (?:by|in)\s+",
        r"published (?:by|in)\s+",
        r"per\s+(?:the\s+)?",
        r"from\s+(?:the\s+)?",
    ]

    for trigger in triggers:
        m = re.search(trigger, text, re.IGNORECASE)
        if not m:
            continue
        after = text[m.end():].strip()
        after = re.sub(r'^(a|an|the|this)\s+', '', after, flags=re.IGNORECASE)

        # All-caps acronym: WHO, CDC, NASA
        acronym = re.match(r'^([A-Z]{2,8})\b', after)
        if acronym:
            return acronym.group(1)

        # Year + org: "2023 MIT study" → "MIT"
        year_org = re.match(r'^\d{4}\s+([A-Z][A-Za-z]+)', after)
        if year_org:
            return year_org.group(1)

        # Capitalised words: "New York Times", "Nature"
        words = after.split()
        source_words = []
        for w in words[:4]:
            clean = w.strip(".,;:()")
            if clean and (clean[0].isupper() or clean.isupper()):
                source_words.append(clean)
            else:
                break
        if source_words:
            return " ".join(source_words)

    # Last resort: first all-caps acronym anywhere in claim
    m = re.search(r'\b([A-Z]{2,8})\b', text)
    if m:
        return m.group(1)

    return "the cited source"


def _strategy_citation_check(claim_text: str, claim_obj) -> List[AdversarialHypothesis]:
    """Challenge the SOURCE's existence, not the content of the claim.

    Emits: "There is no credible evidence that WHO published or stated this."
    NLI scores this LOW when evidence confirms the source is real → SUPPORTED.
    NLI scores this HIGH when no evidence of the source exists → REFUTED.
    """
    if not getattr(claim_obj, "is_citation", False):
        return []

    source = _extract_source(claim_text)
    hyp = f"There is no credible evidence that {source} published or stated this."
    search_query = f'"{source}" source verification existence fact check'

    return [AdversarialHypothesis(
        text=hyp,
        strategy="citation_check",
        search_query=search_query,
    )]


# ---------------------------------------------------------------------------
# STRATEGIES registry  (novelty: enables ablation via exclude_strategies)
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, callable] = {
    "negation":       _strategy_negation,
    "numeric_alt":    _strategy_numeric_alt,
    "temporal_alt":   _strategy_temporal_alt,
    "entity_swap":    _strategy_entity_swap,
    "citation_check": _strategy_citation_check,
}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_adversarial(
    claim_obj,
    exclude_strategies: Optional[Set[str]] = None,
) -> List[AdversarialHypothesis]:
    """Generate adversarial hypotheses for a Claim object.

    Parameters
    ----------
    claim_obj          : Claim dataclass (from claim_extractor)
    exclude_strategies : set of strategy names to skip (used by ablation).
                         E.g. {'numeric_alt'} disables numeric perturbation.

    Returns
    -------
    List of AdversarialHypothesis, sorted by specificity (most targeted first).
    All 5 strategies run by default; ablation selectively disables them.
    """
    excluded = exclude_strategies or set()
    claim_text: str = claim_obj.text
    results: List[AdversarialHypothesis] = []

    for name, fn in STRATEGIES.items():
        if name in excluded:
            continue
        try:
            results.extend(fn(claim_text, claim_obj))
        except Exception:
            pass  # individual strategy failure never blocks others

    # Sort: more specific strategies (entity, citation) ranked ahead of generic
    _priority = {"citation_check": 0, "entity_swap": 1, "numeric_alt": 2,
                 "temporal_alt": 3, "negation": 4}
    results.sort(key=lambda h: _priority.get(h.strategy, 5))

    return results
