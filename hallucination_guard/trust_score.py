"""Core dataclasses and enums shared across all AMADA agents."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class Policy(str, Enum):
    PUBLISH = "PUBLISH"
    FLAG = "FLAG"
    REJECT = "REJECT"


class SuspicionFlag(str, Enum):
    NONE = "NONE"
    HIGH_SUSPICION = "HIGH_SUSPICION_EASY_FACT_FAILED"
    TEMPORAL_STALENESS = "TEMPORAL_STALENESS"
    NUMERIC_DISCREPANCY = "NUMERIC_DISCREPANCY"
    CITATION_UNVERIFIABLE = "CITATION_UNVERIFIABLE"
    ENTITY_CONFUSION = "ENTITY_CONFUSION"


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    text: str
    source: str
    relevance: float
    timestamp_retrieved: float
    url: str = ""


# ---------------------------------------------------------------------------
# Agent communication
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """Typed structured message between AMADA agents. Never contains raw social media content."""
    agent: str
    claim: str
    verdict: Verdict
    confidence: float
    evidence_used: list[EvidenceItem] = field(default_factory=list)
    adversarial_hypotheses: list[str] = field(default_factory=list)
    best_alt_support: float = 0.0
    reasoning: str = ""
    suspicion_flags: list[SuspicionFlag] = field(default_factory=list)
    # Argument-graph memory: the single most compelling point this agent made.
    # Populated after run(); passed as counter_argument to the opposing agent in Round 2.
    strongest_point: str = ""


# ---------------------------------------------------------------------------
# Claim-level result
# ---------------------------------------------------------------------------

@dataclass
class ClaimResult:
    claim: str
    verdict: Verdict
    raw_trust: float
    calibrated_trust: float
    difficulty_score: float
    adversarial_score: float
    prosecutor_confidence: float
    defender_confidence: float
    judge_reasoning: str
    best_alt_hypothesis: str
    best_alt_support: float
    suspicion_flag: SuspicionFlag
    correction_suggestion: str
    evidence_snippets: list[str] = field(default_factory=list)
    nli_entailment: float = 0.0
    nli_contradiction: float = 0.0
    debate_rounds: int = 1
    api_judge_used: bool = False


# ---------------------------------------------------------------------------
# Full request-level result
# ---------------------------------------------------------------------------

@dataclass
class TrustScore:
    text: Optional[str] = None
    image_path: Optional[str] = None
    caption: Optional[str] = None
    claims: list[ClaimResult] = field(default_factory=list)
    fact_score: float = 0.0
    awp_fact_score: float = 0.0
    adversarial_detection_rate: float = 0.0
    ece_without_vcade: float = 0.0
    ece_with_vcade: float = 0.0
    # Image forensics
    cnn_probability: float = 0.0
    ela_energy: float = 0.0
    fft_score: float = 0.0
    image_fusion_score: float = 0.0
    has_periodic_artifacts: bool = False
    watermark_present: bool = False
    image_verdict: str = ""
    clip_similarity: float = 0.0
    contradictions_found: bool = False
    contradiction_severity: float = 0.0
    cross_modal_trust: float = 1.0
    image_description: str = ""
    matched_elements: list[str] = field(default_factory=list)
    ocr_text: str = ""
    objects_detected: list[str] = field(default_factory=list)
    extracted_claims: list[str] = field(default_factory=list)
    numeric_values: list[str] = field(default_factory=list)
    detailed_caption: str = ""
    is_out_of_context: bool = False
    context_trust: float = 1.0
    mismatched_entities: list[str] = field(default_factory=list)
    per_claim_clip: list[dict] = field(default_factory=list)
    faces_found: bool = False
    deepfake_probability: float = 0.0
    watermark_type: str = ""
    # Request metadata
    tier_used: int = 0
    latency_ms: float = 0.0
    api_calls_made: int = 0
    overall_trust: float = 0.0
    policy: Policy = Policy.FLAG
    active_suspicion_flags: list[str] = field(default_factory=list)
    text_weight: float = 0.5
    image_weight: float = 0.3
    modal_weight: float = 0.2

    def compute_overall(self) -> None:
        has_text = bool(self.text)
        has_image = bool(self.image_path and self.image_fusion_score > 0)

        if not has_text and not has_image:
            return

        if has_text and has_image:
            text_trust = sum(c.calibrated_trust for c in self.claims) / len(self.claims)
            self.overall_trust = (
                self.text_weight * text_trust
                + self.image_weight * self.image_fusion_score
                + self.modal_weight * self.cross_modal_trust
            )
        elif has_text:
            self.overall_trust = sum(c.calibrated_trust for c in self.claims) / len(self.claims)
        else:
            # Image-only path.
            authenticity = 1.0 - self.image_fusion_score  # fusion_score = deepfake likelihood
            forensics_trust = (
                0.5 * authenticity
                + 0.3 * self.context_trust
                + 0.2 * (1.0 - self.deepfake_probability)
            )
            if self.claims:
                # OCR claims were fact-checked — blend claim truth with forensics.
                # Claim truthfulness carries more weight (0.6) than image authenticity (0.4)
                # because a real screenshot of misinformation must be caught.
                claim_trust = sum(c.calibrated_trust for c in self.claims) / len(self.claims)
                self.overall_trust = 0.6 * claim_trust + 0.4 * forensics_trust
            else:
                self.overall_trust = forensics_trust

            # Out-of-context penalty: caption clearly doesn't match image content.
            # Cap at 0.25 so an authentic image with a completely false caption
            # always scores REJECT, not just FLAG.
            if self.is_out_of_context:
                self.overall_trust = min(self.overall_trust, 0.25)
                if "OUT_OF_CONTEXT" not in self.active_suspicion_flags:
                    self.active_suspicion_flags.append("OUT_OF_CONTEXT")


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

import time as _time


class Timer:
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = _time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = (_time.perf_counter() - self._start) * 1000  # ms
