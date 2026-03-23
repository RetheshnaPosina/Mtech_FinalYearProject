"""Pydantic request/response schemas for AMADA API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TextVerifyRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=8000)
    use_api_judge: bool = False


class ImageVerifyRequest(BaseModel):
    image_path: str
    caption: Optional[str] = None
    use_api_judge: bool = False


class FullVerifyRequest(BaseModel):
    text: Optional[str] = None
    image_path: Optional[str] = None
    caption: Optional[str] = None
    use_api_judge: bool = Field(default=False)


class ClaimResultOut(BaseModel):
    claim: str
    verdict: str
    raw_trust: float
    calibrated_trust: float
    difficulty_score: float
    adversarial_score: float
    prosecutor_confidence: float
    defender_confidence: float
    judge_reasoning: str
    best_alt_hypothesis: str
    suspicion_flag: str
    correction_suggestion: str
    evidence_snippets: List[str]
    nli_entailment: float = 0.0
    nli_contradiction: float = 0.0
    debate_rounds: int
    api_judge_used: bool


class VerifyResponse(BaseModel):
    overall_trust: float
    policy: str
    tier_used: int
    latency_ms: float
    api_calls_made: int
    claims: List[ClaimResultOut] = []
    fact_score: float = 0.0
    awp_fact_score: float = 0.0
    adversarial_detection_rate: float = 0.0
    active_suspicion_flags: List[str] = []
    # Image forensics fields
    cnn_probability: float = 0.0
    ela_energy: float = 0.0
    fft_score: float = 0.0
    image_fusion_score: float = 0.0
    has_periodic_artifacts: bool = False
    image_verdict: str = ""
    image_description: str = ""
    clip_similarity: float = 0.0
    contradictions_found: bool = False
    contradiction_severity: float = 0.0
    cross_modal_trust: float = 1.0
    matched_elements: List[str] = []
    ocr_text: str = ""
    objects_detected: List[str] = []
    extracted_claims: List[str] = []
    numeric_values: List[str] = []
    detailed_caption: str = ""
    is_out_of_context: bool = False
    context_trust: float = 1.0
    mismatched_entities: List[str] = []
    per_claim_clip: List[dict] = []
    faces_found: bool = False
    deepfake_probability: float = 0.0
    watermark_type: str = ""


class MetricsResponse(BaseModel):
    total_requests: int
    latency: dict
    tier_distribution: dict
    awp_detection_rate: float
    api_call_rate: float
