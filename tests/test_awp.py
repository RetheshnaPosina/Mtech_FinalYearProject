"""Unit tests for AWP scorer."""
import pytest
from hallucination_guard.text.awp_scorer import compute_awp_score
from dataclasses import dataclass


@dataclass
class _Row:
    is_adversarial: bool
    entailment: float
    contradiction: float
    hypothesis: str = ""


def test_awp_supported_claim():
    """High original support, weak adversarial → score near 1 → SUPPORTED."""
    rows = [
        _Row(False, 0.85, 0.10),
        _Row(False, 0.80, 0.12),
        _Row(True,  0.15, 0.10, "alt hyp"),
    ]
    result = compute_awp_score(rows)
    assert result["adversarial_score"] > 0.72
    assert result["original_support"] > 0.5
    assert result["best_alt_support"] == pytest.approx(0.15)


def test_awp_refuted_claim():
    """Weak original support, strong adversarial → score near 0 → REFUTED."""
    rows = [
        _Row(False, 0.20, 0.70),
        _Row(False, 0.25, 0.65),
        _Row(True,  0.80, 0.10, "strong alt"),
    ]
    result = compute_awp_score(rows)
    assert result["adversarial_score"] < 0.35
    assert result["best_alt_text"] == "strong alt"


def test_awp_no_adversarial_rows():
    """No adversarial rows → best_alt_support = 0, score = 1.0 if original > 0."""
    rows = [_Row(False, 0.75, 0.15)]
    result = compute_awp_score(rows)
    assert result["best_alt_support"] == 0.0
    assert result["adversarial_score"] == pytest.approx(1.0)


def test_awp_empty_rows():
    """Empty input → safe fallback defaults returned."""
    result = compute_awp_score([])
    assert result["adversarial_score"] == 0.5
    assert result["original_support"] == 0.0


def test_awp_score_bounded():
    """adversarial_score always in [0, 1]."""
    rows = [
        _Row(False, 0.99, 0.01),
        _Row(True,  0.99, 0.01, "hyp"),
    ]
    result = compute_awp_score(rows)
    assert 0.0 <= result["adversarial_score"] <= 1.0
