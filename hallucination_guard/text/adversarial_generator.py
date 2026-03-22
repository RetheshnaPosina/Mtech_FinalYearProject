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

# Maps year strings to plausible alternatives (±1/±2 years)
_TEMPORAL_SUBS: Dict[str, List[str]] = {
    "2023": ["2022", "2024"],
    "2022": ["2021", "2023"],
    "2021": ["2020", "2022"],
    "2020": ["2019", "2021"],
    "2019": ["2018", "2020"],
    "2018": ["2017", "2019"],
}

_TEMPORAL_KEYWORDS = re.compile(
    r"\b(current|now|today|latest|recent|still|anymore"
    r"|as of \d{4}|january|february|march|april|may|june"
    r"|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)

_STALE_SUBS = ["formerly", "previously", "years ago", "in the past"]


def _apply_temporal_shift(text: str) -> str:
    """Replace time references with staleness markers."""
    # First try year substitution
    for year, alts in _TEMPORAL_SUBS.items():
        if year in text:
            return text.replace(year, alts[0], 1)
    # Then try keyword substitution
    result, n = _TEMPORAL_KEYWORDS.subn(_STALE_SUBS[0], text, count=1)
    if n == 0:
        result = _STALE_SUBS[0] + " " + text
    return result


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
    return [AdversarialHypothesis(
        text=hyp,
        strategy="temporal_alt",
        search_query="current status check: " + claim_text[:80],
    )]


def _strategy_entity_swap(claim_text: str, claim_obj) -> List[AdversarialHypothesis]:
    entities = getattr(claim_obj, "entities", [])
    if not entities:
        return []
    hypotheses = []
    for ent in entities[:2]:
        hyp = f'"{ent}": ' + claim_text
        hypotheses.append(AdversarialHypothesis(
            text=hyp,
            strategy="entity_swap",
            search_query=f'"{ent}" correct facts disambiguation',
        ))
    return hypotheses


def _strategy_citation_check(claim_text: str, claim_obj) -> List[AdversarialHypothesis]:
    if not getattr(claim_obj, "is_citation", False):
        return []
    return [AdversarialHypothesis(
        text=claim_text,
        strategy="citation_check",
        search_query="verify source existence: " + claim_text[:80],
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
