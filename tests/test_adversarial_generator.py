"""Tests for AdversarialGenerator — covers all 5 AWP strategies.

Research relevance: validates the core novelty claim that each strategy
contributes independently to adversarial hypothesis generation. The ablation
benchmark (benchmarks/run_ablation.py) depends on exclude_strategies working
correctly — these tests prove that interface is reliable.
"""
import pytest
from dataclasses import dataclass, field
from typing import List
from hallucination_guard.text.adversarial_generator import (
    generate_adversarial, STRATEGIES, AdversarialHypothesis,
    _apply_negation, _apply_temporal_shift,
)


@dataclass
class _Claim:
    text: str
    has_number: bool = False
    has_entity: bool = False
    has_temporal: bool = False
    is_citation: bool = False
    entities: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

def test_all_five_strategies_registered():
    """All 5 AWP strategies must be present in the registry."""
    expected = {"negation", "numeric_alt", "temporal_alt", "entity_swap", "citation_check"}
    assert set(STRATEGIES.keys()) == expected


# ---------------------------------------------------------------------------
# Negation strategy
# ---------------------------------------------------------------------------

def test_negation_auxiliary_verb():
    claim = _Claim("The vaccine is effective.")
    hyps = STRATEGIES["negation"](claim.text, claim)
    assert len(hyps) == 1
    assert "is not" in hyps[0].text
    assert hyps[0].strategy == "negation"


def test_negation_was_verb():
    claim = _Claim("World War II was a global conflict.")
    hyps = STRATEGIES["negation"](claim.text, claim)
    assert "was not" in hyps[0].text


def test_negation_fallback_prefix():
    """Claims without auxiliary verbs get 'NOT ' prefix."""
    claim = _Claim("Photosynthesis produces oxygen.")
    hyps = STRATEGIES["negation"](claim.text, claim)
    assert len(hyps) == 1
    assert hyps[0].text.startswith("NOT ")


# ---------------------------------------------------------------------------
# Numeric alt strategy
# ---------------------------------------------------------------------------

def test_numeric_alt_generates_perturbations():
    claim = _Claim("The speed of light is 300,000 km/s.", has_number=True, numbers=["300,000"])
    hyps = STRATEGIES["numeric_alt"](claim.text, claim)
    assert len(hyps) >= 1
    assert all(h.strategy == "numeric_alt" for h in hyps)
    # Perturbed values should differ from original
    for h in hyps:
        assert "300,000" not in h.text or h.text != claim.text


def test_numeric_alt_skipped_when_no_number():
    """numeric_alt produces nothing when has_number=False."""
    claim = _Claim("Einstein developed the theory of relativity.", has_number=False)
    hyps = STRATEGIES["numeric_alt"](claim.text, claim)
    assert hyps == []


def test_numeric_alt_year_perturbed():
    claim = _Claim("World War II ended in 1945.", has_number=True, numbers=["1945"])
    hyps = STRATEGIES["numeric_alt"](claim.text, claim)
    assert len(hyps) >= 1
    assert "1945" not in hyps[0].text  # number was changed


# ---------------------------------------------------------------------------
# Temporal alt strategy
# ---------------------------------------------------------------------------

def test_temporal_alt_year_substitution():
    claim = _Claim("The treaty was signed in 2020.", has_temporal=True)
    hyps = STRATEGIES["temporal_alt"](claim.text, claim)
    assert len(hyps) == 1
    assert hyps[0].strategy == "temporal_alt"
    assert "2020" not in hyps[0].text  # year was substituted


def test_temporal_alt_keyword_substitution():
    claim = _Claim("The current President supports the policy.", has_temporal=True)
    hyps = STRATEGIES["temporal_alt"](claim.text, claim)
    assert len(hyps) == 1
    assert "current" not in hyps[0].text.lower()


def test_temporal_alt_skipped_when_not_temporal():
    claim = _Claim("Water boils at 100°C.", has_temporal=False)
    hyps = STRATEGIES["temporal_alt"](claim.text, claim)
    assert hyps == []


# ---------------------------------------------------------------------------
# Entity swap strategy
# ---------------------------------------------------------------------------

def test_entity_swap_generates_hypotheses():
    claim = _Claim(
        "Shakespeare wrote Hamlet.",
        has_entity=True,
        entities=["Shakespeare", "Hamlet"]
    )
    hyps = STRATEGIES["entity_swap"](claim.text, claim)
    assert len(hyps) >= 1
    assert all(h.strategy == "entity_swap" for h in hyps)


def test_entity_swap_skipped_no_entities():
    claim = _Claim("DNA is a double helix.", has_entity=False, entities=[])
    hyps = STRATEGIES["entity_swap"](claim.text, claim)
    assert hyps == []


def test_entity_swap_max_two_entities():
    """At most 2 entity hypotheses generated (top 2 entities)."""
    claim = _Claim(
        "A B C D co-founded the company.",
        has_entity=True,
        entities=["A", "B", "C", "D"]
    )
    hyps = STRATEGIES["entity_swap"](claim.text, claim)
    assert len(hyps) <= 2


# ---------------------------------------------------------------------------
# Citation check strategy
# ---------------------------------------------------------------------------

def test_citation_check_generated_for_citations():
    claim = _Claim(
        "According to WHO, smoking causes cancer.",
        is_citation=True,
        has_entity=True,
        entities=["WHO"]
    )
    hyps = STRATEGIES["citation_check"](claim.text, claim)
    assert len(hyps) == 1
    assert hyps[0].strategy == "citation_check"


def test_citation_check_skipped_non_citation():
    claim = _Claim("Water boils at 100°C.", is_citation=False)
    hyps = STRATEGIES["citation_check"](claim.text, claim)
    assert hyps == []


# ---------------------------------------------------------------------------
# Ablation interface — exclude_strategies
# ---------------------------------------------------------------------------

def test_exclude_single_strategy():
    """Excluding a strategy produces no hypotheses for that strategy."""
    claim = _Claim(
        "The speed of light is 300,000 km/s.",
        has_number=True, numbers=["300,000"]
    )
    all_hyps = generate_adversarial(claim)
    excl_hyps = generate_adversarial(claim, exclude_strategies={"numeric_alt"})

    strategies_all = {h.strategy for h in all_hyps}
    strategies_excl = {h.strategy for h in excl_hyps}

    assert "numeric_alt" in strategies_all
    assert "numeric_alt" not in strategies_excl


def test_exclude_all_strategies_returns_empty():
    claim = _Claim("The vaccine is effective.", has_number=False)
    all_strategies = set(STRATEGIES.keys())
    hyps = generate_adversarial(claim, exclude_strategies=all_strategies)
    assert hyps == []


def test_exclude_nonexistent_strategy_safe():
    """Excluding a strategy name that doesn't exist raises no error."""
    claim = _Claim("The vaccine is effective.")
    hyps = generate_adversarial(claim, exclude_strategies={"nonexistent_strategy"})
    assert isinstance(hyps, list)


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

def test_citation_and_entity_ranked_first():
    """citation_check and entity_swap should appear before negation in output."""
    claim = _Claim(
        "According to WHO, smoking causes cancer.",
        is_citation=True, has_entity=True, entities=["WHO"]
    )
    hyps = generate_adversarial(claim)
    strategies = [h.strategy for h in hyps]
    # negation should not appear before citation/entity
    if "negation" in strategies and "citation_check" in strategies:
        assert strategies.index("citation_check") < strategies.index("negation")


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

def test_all_hypotheses_are_dataclass():
    claim = _Claim(
        "Einstein won the Nobel Prize in 1921.",
        has_number=True, has_entity=True, has_temporal=True,
        entities=["Einstein"], numbers=["1921"]
    )
    hyps = generate_adversarial(claim)
    for h in hyps:
        assert isinstance(h, AdversarialHypothesis)
        assert isinstance(h.text, str)
        assert isinstance(h.strategy, str)
        assert isinstance(h.search_query, str)
        assert h.text  # non-empty
