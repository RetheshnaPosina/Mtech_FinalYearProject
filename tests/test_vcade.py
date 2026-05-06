"""Unit tests for VCADE calibrator."""
import pytest
from hallucination_guard.text.vcade_calibrator import compute_vcade
from hallucination_guard.trust_score import EvidenceItem, SuspicionFlag
import time


def _ev(relevance: float) -> EvidenceItem:
    return EvidenceItem(
        text="evidence", source="wiki",
        relevance=relevance, timestamp_retrieved=time.time()
    )


def test_vcade_supported_high_evidence():
    """Well-supported claim with strong evidence → high calibrated trust."""
    evidence = [_ev(0.9), _ev(0.85), _ev(0.88)]
    result = compute_vcade(
        raw_trust=0.9, verdict_label="SUPPORTED",
        evidence=evidence, best_alt_support=0.1,
        entity_count=1, has_temporal=False,
    )
    assert result.calibrated_trust > 0.7
    assert result.suspicion_flag == SuspicionFlag.NONE
    assert 0.0 <= result.calibrated_trust <= 1.0


def test_vcade_refuted_easy_fact():
    """Easy refuted fact (low difficulty) → HIGH_SUSPICION flag."""
    evidence = [_ev(0.9), _ev(0.85)]
    result = compute_vcade(
        raw_trust=0.8, verdict_label="REFUTED",
        evidence=evidence, best_alt_support=0.05,
        entity_count=0, has_temporal=False,
    )
    assert result.suspicion_flag == SuspicionFlag.HIGH_SUSPICION
    assert result.difficulty < 0.3


def test_vcade_temporal_refuted():
    """Temporal claim that is refuted → TEMPORAL_STALENESS flag."""
    evidence = [_ev(0.5)]
    result = compute_vcade(
        raw_trust=0.7, verdict_label="REFUTED",
        evidence=evidence, best_alt_support=0.3,
        entity_count=0, has_temporal=True,
    )
    assert result.suspicion_flag == SuspicionFlag.TEMPORAL_STALENESS


def test_vcade_entity_confusion():
    """Many entities, not supported → ENTITY_CONFUSION flag."""
    evidence = [_ev(0.4)]
    result = compute_vcade(
        raw_trust=0.5, verdict_label="NOT_ENOUGH_INFO",
        evidence=evidence, best_alt_support=0.3,
        entity_count=6, has_temporal=False,
    )
    assert result.suspicion_flag == SuspicionFlag.ENTITY_CONFUSION


def test_vcade_calibrated_trust_bounded():
    """calibrated_trust always in [0, 1] regardless of inputs."""
    for verdict in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
        result = compute_vcade(
            raw_trust=1.0, verdict_label=verdict,
            evidence=[], best_alt_support=1.0,
            entity_count=10, has_temporal=True,
        )
        assert 0.0 <= result.calibrated_trust <= 1.0, \
            f"Out of bounds for verdict={verdict}: {result.calibrated_trust}"


def test_vcade_no_evidence():
    """No evidence → d_retrieval = 1.0 (maximum difficulty)."""
    result = compute_vcade(
        raw_trust=0.8, verdict_label="SUPPORTED",
        evidence=[], best_alt_support=0.0,
        entity_count=0, has_temporal=False,
    )
    assert result.d_retrieval == 1.0
