"""Tests that audit logger writes calibrated_trust in [0,1], not percentage."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hallucination_guard.trust_score import (
    TrustScore, ClaimResult, Verdict, Policy, SuspicionFlag,
)
from hallucination_guard.audit.logger import log_result


def _make_result() -> TrustScore:
    claim = ClaimResult(
        claim="Test claim",
        verdict=Verdict.SUPPORTED,
        raw_trust=0.85,
        calibrated_trust=0.82,      # should appear as 0.82 NOT 82.0
        difficulty_score=0.25,
        adversarial_score=0.9,
        prosecutor_confidence=0.3,
        defender_confidence=0.85,
        judge_reasoning="local_vote",
        best_alt_hypothesis="",
        best_alt_support=0.1,
        suspicion_flag=SuspicionFlag.NONE,
        correction_suggestion="",
    )
    result = TrustScore(text="Test claim")
    result.claims = [claim]
    result.overall_trust = 0.82
    result.policy = Policy.PUBLISH
    result.tier_used = 1
    result.latency_ms = 123.4
    return result


def test_calibrated_trust_not_percentage(tmp_path):
    """calibrated_trust in audit log must be in [0,1], not multiplied by 100."""
    log_file = tmp_path / "audit.jsonl"

    with patch("hallucination_guard.audit.logger.settings") as mock_settings:
        mock_settings.audit_dir = tmp_path
        log_result(_make_result(), request_id="test-001")

    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(entries) == 1
    ct = entries[0]["claims"][0]["calibrated_trust"]
    assert ct <= 1.0, f"calibrated_trust should be in [0,1], got {ct}"
    assert ct == pytest.approx(0.82, abs=0.01)


def test_suspicion_flag_is_string(tmp_path):
    """suspicion_flag in audit log must be a string, not an enum object."""
    with patch("hallucination_guard.audit.logger.settings") as mock_settings:
        mock_settings.audit_dir = tmp_path
        log_result(_make_result(), request_id="test-002")

    log_file = tmp_path / "audit.jsonl"
    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    flag = entries[0]["claims"][0]["suspicion_flag"]
    assert isinstance(flag, str), f"suspicion_flag should be str, got {type(flag)}"


def test_request_id_recorded(tmp_path):
    """request_id must be written to audit log."""
    with patch("hallucination_guard.audit.logger.settings") as mock_settings:
        mock_settings.audit_dir = tmp_path
        log_result(_make_result(), request_id="abc12345")

    log_file = tmp_path / "audit.jsonl"
    entries = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert entries[0]["request_id"] == "abc12345"
