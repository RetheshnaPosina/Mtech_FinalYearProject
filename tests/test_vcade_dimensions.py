"""Deep VCADE dimension tests — validates each difficulty axis independently.

Research relevance: VCADE's 5-dimension difficulty model is a novel contribution.
These tests prove each dimension responds correctly to its inputs, supporting the
thesis claim that VCADE produces better-calibrated trust than raw NLI scores.
They also validate the monotonicity property: harder claims get more downward
calibration adjustment.
"""
import time
import pytest
from hallucination_guard.text.vcade_calibrator import compute_vcade
from hallucination_guard.trust_score import EvidenceItem, SuspicionFlag


def _ev(relevance: float, url: str = "") -> EvidenceItem:
    return EvidenceItem(text="ev", source="s", relevance=relevance,
                        timestamp_retrieved=time.time(), url=url)


# ---------------------------------------------------------------------------
# d_retrieval dimension
# ---------------------------------------------------------------------------

def test_d_retrieval_no_evidence():
    result = compute_vcade(0.5, "SUPPORTED", [], 0.0, 0, False)
    assert result.d_retrieval == 1.0


def test_d_retrieval_perfect_evidence():
    evidence = [_ev(1.0), _ev(1.0), _ev(1.0)]
    result = compute_vcade(0.9, "SUPPORTED", evidence, 0.0, 0, False)
    assert result.d_retrieval == pytest.approx(0.0, abs=0.01)


def test_d_retrieval_partial_evidence():
    evidence = [_ev(0.6)]
    result = compute_vcade(0.8, "SUPPORTED", evidence, 0.0, 0, False)
    assert result.d_retrieval == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# d_consensus dimension
# ---------------------------------------------------------------------------

def test_d_consensus_single_evidence_returns_0_5():
    """Single evidence item → stdev undefined → default 0.5."""
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 0, False)
    assert result.d_consensus == pytest.approx(0.5)


def test_d_consensus_uniform_evidence_near_zero():
    """All sources agree → low consensus difficulty."""
    evidence = [_ev(0.85), _ev(0.85), _ev(0.85), _ev(0.85)]
    result = compute_vcade(0.8, "SUPPORTED", evidence, 0.0, 0, False)
    assert result.d_consensus < 0.1


def test_d_consensus_mixed_evidence_high():
    """Sources strongly disagree → high consensus difficulty."""
    evidence = [_ev(0.1), _ev(0.9), _ev(0.1), _ev(0.9)]
    result = compute_vcade(0.5, "NOT_ENOUGH_INFO", evidence, 0.0, 0, False)
    assert result.d_consensus > 0.5


# ---------------------------------------------------------------------------
# d_adversarial dimension
# ---------------------------------------------------------------------------

def test_d_adversarial_equals_best_alt_support():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.65, 0, False)
    assert result.d_adversarial == pytest.approx(0.65)


def test_d_adversarial_zero_when_no_alt():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 0, False)
    assert result.d_adversarial == 0.0


# ---------------------------------------------------------------------------
# d_entity dimension
# ---------------------------------------------------------------------------

def test_d_entity_zero_entities():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 0, False)
    assert result.d_entity == 0.0


def test_d_entity_caps_at_one():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 100, False)
    assert result.d_entity == 1.0


def test_d_entity_four_entities():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 4, False)
    assert result.d_entity == pytest.approx(1.0, abs=0.01)


def test_d_entity_one_entity():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 1, False)
    assert result.d_entity == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# d_temporal dimension
# ---------------------------------------------------------------------------

def test_d_temporal_high_when_temporal():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 0, True)
    assert result.d_temporal == pytest.approx(0.65)


def test_d_temporal_low_when_not_temporal():
    result = compute_vcade(0.8, "SUPPORTED", [_ev(0.9)], 0.0, 0, False)
    assert result.d_temporal == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Difficulty score aggregation
# ---------------------------------------------------------------------------

def test_difficulty_bounded_zero_to_one():
    for entity_count in [0, 5, 20]:
        for has_temporal in [True, False]:
            result = compute_vcade(0.5, "NOT_ENOUGH_INFO", [], 1.0, entity_count, has_temporal)
            assert 0.0 <= result.difficulty <= 1.0, \
                f"difficulty out of bounds: {result.difficulty}"


def test_high_difficulty_all_dimensions_maxed():
    """All difficulty dimensions at max → difficulty close to 1.0."""
    result = compute_vcade(
        raw_trust=0.5, verdict_label="NOT_ENOUGH_INFO",
        evidence=[],          # d_retrieval = 1.0
        best_alt_support=1.0, # d_adversarial = 1.0
        entity_count=20,      # d_entity = 1.0
        has_temporal=True,    # d_temporal = 0.65
    )
    assert result.difficulty > 0.7


def test_low_difficulty_all_dimensions_zero():
    """All difficulty dimensions at min → difficulty close to 0."""
    evidence = [_ev(1.0)] * 5  # d_retrieval ≈ 0, d_consensus ≈ 0
    result = compute_vcade(
        raw_trust=0.9, verdict_label="SUPPORTED",
        evidence=evidence,
        best_alt_support=0.0,  # d_adversarial = 0
        entity_count=0,        # d_entity = 0
        has_temporal=False,    # d_temporal = 0.1
    )
    assert result.difficulty < 0.15


# ---------------------------------------------------------------------------
# Calibration monotonicity
# ---------------------------------------------------------------------------

def test_supported_harder_claim_gets_lower_calibrated_trust():
    """For SUPPORTED: harder claim (more adversarial) → more downward adjustment."""
    easy = compute_vcade(0.85, "SUPPORTED", [_ev(0.9)], 0.05, 0, False)
    hard = compute_vcade(0.85, "SUPPORTED", [_ev(0.9)], 0.80, 0, True)
    assert easy.calibrated_trust >= hard.calibrated_trust


def test_refuted_easy_fact_is_more_suspicious():
    """For REFUTED: easy claim (low difficulty) → lower calibrated trust → HIGH_SUSPICION."""
    easy_refuted = compute_vcade(0.8, "REFUTED", [_ev(0.95)], 0.01, 0, False)
    hard_refuted = compute_vcade(0.8, "REFUTED", [], 0.9, 10, True)
    assert easy_refuted.difficulty < hard_refuted.difficulty
    assert easy_refuted.calibrated_trust <= hard_refuted.calibrated_trust


# ---------------------------------------------------------------------------
# Suspicion flag logic
# ---------------------------------------------------------------------------

def test_no_flag_for_easy_supported():
    result = compute_vcade(0.9, "SUPPORTED", [_ev(0.95)], 0.05, 0, False)
    assert result.suspicion_flag == SuspicionFlag.NONE


def test_high_suspicion_easy_fact_refuted():
    result = compute_vcade(0.9, "REFUTED", [_ev(0.98)], 0.02, 0, False)
    assert result.difficulty < 0.3
    assert result.suspicion_flag == SuspicionFlag.HIGH_SUSPICION


def test_temporal_staleness_flag():
    result = compute_vcade(0.7, "REFUTED", [_ev(0.5)], 0.3, 0, True)
    assert result.suspicion_flag == SuspicionFlag.TEMPORAL_STALENESS


def test_entity_confusion_flag():
    result = compute_vcade(0.5, "REFUTED", [_ev(0.4)], 0.3, 6, False)
    assert result.suspicion_flag == SuspicionFlag.ENTITY_CONFUSION
