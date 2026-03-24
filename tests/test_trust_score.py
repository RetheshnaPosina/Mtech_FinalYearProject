"""Tests for TrustScore.compute_overall — multimodal trust fusion.

Research relevance: The weighted fusion formula (text 0.5, image 0.3,
cross-modal 0.2) is a design choice that affects all multimodal results.
These tests verify the formula behaves correctly in all three input modes:
text-only, image-only, and full multimodal.
"""
import pytest
from hallucination_guard.trust_score import (
    TrustScore, ClaimResult, Verdict, Policy, SuspicionFlag,
)


def _claim(calibrated_trust: float, verdict: Verdict = Verdict.SUPPORTED) -> ClaimResult:
    return ClaimResult(
        claim="test", verdict=verdict,
        raw_trust=calibrated_trust, calibrated_trust=calibrated_trust,
        difficulty_score=0.3, adversarial_score=0.8,
        prosecutor_confidence=0.3, defender_confidence=0.8,
        judge_reasoning="test", best_alt_hypothesis="",
        best_alt_support=0.1, suspicion_flag=SuspicionFlag.NONE,
        correction_suggestion="",
    )


# ---------------------------------------------------------------------------
# Text-only mode
# ---------------------------------------------------------------------------

def test_text_only_overall_is_mean_calibrated_trust():
    ts = TrustScore(text="test")
    ts.claims = [_claim(0.8), _claim(0.6)]
    ts.compute_overall()
    assert ts.overall_trust == pytest.approx(0.7, abs=0.01)


def test_text_only_single_claim():
    ts = TrustScore(text="test")
    ts.claims = [_claim(0.75)]
    ts.compute_overall()
    assert ts.overall_trust == pytest.approx(0.75)


def test_text_only_all_refuted():
    ts = TrustScore(text="test")
    ts.claims = [_claim(0.1, Verdict.REFUTED), _claim(0.2, Verdict.REFUTED)]
    ts.compute_overall()
    assert ts.overall_trust < 0.4


# ---------------------------------------------------------------------------
# Image-only mode
# ---------------------------------------------------------------------------

def test_image_only_no_ocr_claims():
    """Image-only with no OCR claims: trust from forensics signal."""
    ts = TrustScore(image_path="/img.jpg")
    ts.image_fusion_score = 0.2   # low fusion = authentic
    ts.deepfake_probability = 0.1
    ts.context_trust = 0.9
    ts.is_out_of_context = False
    ts.compute_overall()
    # authenticity = 1 - 0.2 = 0.8
    # forensics_trust = 0.5*0.8 + 0.3*0.9 + 0.2*(1-0.1) = 0.4 + 0.27 + 0.18 = 0.85
    assert ts.overall_trust > 0.6


def test_image_only_high_fusion_score_lowers_trust():
    """High fusion score (likely manipulated) → lower overall trust."""
    ts_clean = TrustScore(image_path="/clean.jpg")
    ts_clean.image_fusion_score = 0.1
    ts_clean.deepfake_probability = 0.05
    ts_clean.context_trust = 1.0
    ts_clean.compute_overall()

    ts_fake = TrustScore(image_path="/fake.jpg")
    ts_fake.image_fusion_score = 0.8
    ts_fake.deepfake_probability = 0.7
    ts_fake.context_trust = 0.5
    ts_fake.compute_overall()

    assert ts_clean.overall_trust > ts_fake.overall_trust


def test_out_of_context_caps_trust_at_0_25():
    """Out-of-context image must be capped at 0.25 (forces REJECT)."""
    ts = TrustScore(image_path="/img.jpg")
    ts.image_fusion_score = 0.1   # would normally give high trust
    ts.deepfake_probability = 0.0
    ts.context_trust = 0.9
    ts.is_out_of_context = True
    ts.compute_overall()
    assert ts.overall_trust <= 0.25
    assert "OUT_OF_CONTEXT" in ts.active_suspicion_flags


def test_out_of_context_flag_not_duplicated():
    """OUT_OF_CONTEXT flag added only once even if compute_overall called twice."""
    ts = TrustScore(image_path="/img.jpg")
    ts.image_fusion_score = 0.1
    ts.is_out_of_context = True
    ts.compute_overall()
    ts.compute_overall()
    assert ts.active_suspicion_flags.count("OUT_OF_CONTEXT") == 1


def test_image_only_with_ocr_claims():
    """Image with verified OCR claims: trust blends claim truth (0.6) + forensics (0.4)."""
    ts = TrustScore(image_path="/img.jpg")
    ts.image_fusion_score = 0.2
    ts.deepfake_probability = 0.1
    ts.context_trust = 0.9
    ts.claims = [_claim(0.9), _claim(0.85)]  # OCR claims fact-checked
    ts.is_out_of_context = False
    ts.compute_overall()
    # Blended: 0.6 * mean_claim_trust + 0.4 * forensics_trust
    assert ts.overall_trust > 0.7


# ---------------------------------------------------------------------------
# Full multimodal mode
# ---------------------------------------------------------------------------

def test_multimodal_uses_weighted_formula():
    """text_weight=0.5, image_weight=0.3, modal_weight=0.2."""
    ts = TrustScore(text="test", image_path="/img.jpg")
    ts.claims = [_claim(0.8)]
    ts.image_fusion_score = 0.9   # image path has data
    ts.cross_modal_trust = 0.7
    ts.compute_overall()
    # text_trust = 0.8
    # expected = 0.5*0.8 + 0.3*0.9 + 0.2*0.7 = 0.4 + 0.27 + 0.14 = 0.81
    assert ts.overall_trust == pytest.approx(0.81, abs=0.02)


def test_no_input_overall_stays_zero():
    """compute_overall with no text or image → overall_trust unchanged (0.0)."""
    ts = TrustScore()
    ts.compute_overall()
    assert ts.overall_trust == 0.0


# ---------------------------------------------------------------------------
# Overall trust bounds
# ---------------------------------------------------------------------------

def test_overall_trust_never_exceeds_one():
    ts = TrustScore(text="test", image_path="/img.jpg")
    ts.claims = [_claim(1.0)]
    ts.image_fusion_score = 1.0
    ts.cross_modal_trust = 1.0
    ts.compute_overall()
    assert ts.overall_trust <= 1.0


def test_overall_trust_never_below_zero():
    ts = TrustScore(text="test", image_path="/img.jpg")
    ts.claims = [_claim(0.0, Verdict.REFUTED)]
    ts.image_fusion_score = 0.0
    ts.cross_modal_trust = 0.0
    ts.compute_overall()
    assert ts.overall_trust >= 0.0
