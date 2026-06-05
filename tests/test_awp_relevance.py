"""Regression tests for the 2026-05-30 'evident truths flagged' fix.

Root cause: off-topic evidence (neutral-dominated w.r.t. the claim — e.g.
fact-checks of unrelated quotes by the same entity) was (a) averaged into
original_support, crushing it toward 0, and (b) handed to the adversarial
hypotheses, inflating best_alt_support. The combination drove AWP toward 0,
so the Prosecutor returned REFUTED for true claims even with no contradicting
evidence.

Fixes verified here:
  #1  off-topic evidence is excluded from BOTH sides before scoring
  #2  absence of supporting evidence yields relevant_support_count == 0
      (NOT_ENOUGH_INFO territory) rather than a low score read as refutation
"""
from dataclasses import dataclass

from hallucination_guard.text.awp_scorer import compute_awp_score
from hallucination_guard.text.entailment_matrix import relevant_evidence_texts


@dataclass
class _Row:
    is_adversarial: bool
    entailment: float
    contradiction: float
    evidence_text: str = ""
    hypothesis: str = ""


def test_offtopic_evidence_excluded_from_scoring():
    """Neutral-dominated evidence must not contribute to original_support.

    Reproduces the Trump-quote-factcheck case: one genuinely supporting item
    plus two off-topic items (entailment≈0, contradiction≈0 → neutral≈1) that
    also happen to entail the adversarial hypothesis. After the fix, only the
    on-topic item counts and the adversarial inflation from the off-topic
    items is dropped.
    """
    rows = [
        # On-topic supporting evidence for the claim.
        _Row(False, 0.78, 0.05, evidence_text="ev_bio"),
        # Off-topic: a fact-check of an unrelated quote (neutral-dominated).
        _Row(False, 0.003, 0.01, evidence_text="ev_quote1"),
        _Row(False, 0.001, 0.02, evidence_text="ev_quote2"),
        # Adversarial hypothesis spuriously entails the off-topic quote.
        _Row(True, 0.99, 0.01, evidence_text="ev_quote2", hypothesis="someone other than X"),
        # Adversarial vs the on-topic evidence scores low (correct).
        _Row(True, 0.10, 0.02, evidence_text="ev_bio", hypothesis="someone other than X"),
    ]

    result = compute_awp_score(rows)

    # Off-topic evidence dropped: only the on-topic item drives original_support.
    assert result["relevant_support_count"] == 1
    assert result["original_support"] > 0.5
    # The spurious 0.99 adversarial (tied to off-topic evidence) is excluded;
    # only the 0.10 adversarial (tied to on-topic evidence) survives.
    assert result["best_alt_support"] < 0.35
    # Net: the true claim is supported, not refuted.
    assert result["adversarial_score"] > 0.72


def test_no_relevant_evidence_is_not_refutation():
    """All evidence off-topic → relevant_support_count 0 and neutral score 0.5.

    This is the signal the Prosecutor uses to return NOT_ENOUGH_INFO instead
    of REFUTED when nothing actually addresses the claim.
    """
    rows = [
        _Row(False, 0.002, 0.01, evidence_text="off1"),
        _Row(False, 0.004, 0.02, evidence_text="off2"),
        _Row(True, 0.99, 0.01, evidence_text="off2", hypothesis="adv"),
    ]

    result = compute_awp_score(rows)

    assert result["relevant_support_count"] == 0
    assert result["adversarial_score"] == 0.5
    assert result["best_alt_support"] == 0.0


def test_genuine_contradiction_still_refutes():
    """A claim with real contradicting evidence must still be refutable.

    Ensures the relevance gate does not neuter detection of false claims:
    contradicting evidence is on-topic (high contradiction → passes the gate).
    """
    rows = [
        _Row(False, 0.10, 0.85, evidence_text="contra1"),
        _Row(False, 0.08, 0.80, evidence_text="contra2"),
        _Row(True, 0.88, 0.05, evidence_text="contra1", hypothesis="adv"),
    ]

    result = compute_awp_score(rows)

    assert result["relevant_support_count"] == 2
    assert result["avg_contradiction"] > 0.5
    assert result["adversarial_score"] < 0.35


def test_relevant_evidence_texts_filters_neutral():
    """The helper keeps on-topic items and drops neutral-dominated ones."""
    rows = [
        _Row(False, 0.70, 0.05, evidence_text="keep_entail"),
        _Row(False, 0.05, 0.70, evidence_text="keep_contra"),
        _Row(False, 0.02, 0.03, evidence_text="drop_neutral"),
    ]
    relevant = relevant_evidence_texts(rows)
    assert "keep_entail" in relevant
    assert "keep_contra" in relevant
    assert "drop_neutral" not in relevant
