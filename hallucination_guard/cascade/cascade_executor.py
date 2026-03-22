"""Tiered cascade executor: Tier 0→1→2→3 with early exit."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from hallucination_guard.trust_score import Policy, TrustScore, Verdict
from hallucination_guard.config import settings

logger = logging.getLogger(__name__)

_ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

# Lazy singleton for ForensicsAgent (avoid re-loading models every call)
_forensics = None


def _validate_image_path(image_path: str) -> str:
    """Fix #4: Validate image_path before passing to ForensicsAgent."""
    try:
        resolved = Path(image_path).resolve()
    except Exception as e:
        raise ValueError(f"Invalid image path: {image_path}") from e
    if ".." in resolved.parts:
        raise ValueError("Image path traversal detected")
    if not resolved.exists():
        raise ValueError(f"Image file not found: {resolved}")
    if resolved.suffix.lower() not in _ALLOWED_IMAGE_EXTS:
        raise ValueError(f"Unsupported image format: {resolved.suffix}")
    return str(resolved)


async def execute(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    caption: Optional[str] = None,
    use_api_judge: bool = False,
) -> TrustScore:
    from hallucination_guard.sentinel.input_validator import validate
    from hallucination_guard.sentinel.risk_estimator import estimate_risk
    from hallucination_guard.text.policy_engine import decide_policy
    from hallucination_guard.agents.debate_orchestrator import verify_text
    from hallucination_guard.agents.forensics_agent import ForensicsAgent
    from hallucination_guard.consistency.pipeline import run_cmcd
    from hallucination_guard.cascade.budget_manager import BudgetManager
    from hallucination_guard.cascade.router import route

    global _forensics

    budget = BudgetManager()
    result = TrustScore(text=text, image_path=image_path, caption=caption)

    # --- Tier 0: Input validation ---
    if text:
        v = validate(text)
        if not v.valid:
            result.policy = Policy.REJECT
            result.active_suspicion_flags.append(f"rejected: {v.reason}")
            result.tier_used = 0
            result.latency_ms = budget.elapsed_ms
            return result

    # --- Risk estimation and routing ---
    risk = estimate_risk(text or "")
    tier = route(risk, has_image=bool(image_path))
    result.tier_used = tier

    # --- Tier 1+: Text verification via AWP debate ---
    claims = []
    if text:
        claims = await verify_text(text, use_api_judge=use_api_judge)
        result.claims = claims

        if claims:
            supported = sum(1 for c in claims if c.verdict == Verdict.SUPPORTED)
            avg_trust = sum(c.calibrated_trust for c in claims) / len(claims)
            result.overall_trust = avg_trust
            result.awp_fact_score = supported / len(claims)
            result.fact_score = result.awp_fact_score
            result.adversarial_detection_rate = sum(1 for c in claims if c.api_judge_used) / len(claims)
            result.active_suspicion_flags = [
                c.suspicion_flag.value
                for c in claims
                if c.suspicion_flag.value != "NONE"
            ]

        result.policy = decide_policy(result.overall_trust)

    # --- Tier 2+: Image forensics ---
    visual_facts = None
    if image_path and tier >= 2:
        try:
            safe_path = _validate_image_path(image_path)
        except ValueError as e:
            logger.error("Image validation failed: %s", e)
            result.latency_ms = budget.elapsed_ms
            return result

        if _forensics is None:
            _forensics = ForensicsAgent()

        forensics_task = asyncio.create_task(_forensics.run_with_image(safe_path, caption))

        try:
            forensics_result = await forensics_task
            result.api_calls_made += forensics_result.get("api_calls_made", 0)
            result.cnn_probability = forensics_result.get("cnn_probability", 0.0)
            result.ela_energy = forensics_result.get("ela_energy", 0.0)
            result.fft_score = forensics_result.get("fft_score", 0.0)
            result.image_fusion_score = forensics_result.get("fusion_score", 0.0)
            result.has_periodic_artifacts = forensics_result.get("has_periodic_artifacts", False)
            result.image_description = forensics_result.get("caption", "")
            result.image_verdict = forensics_result.get("verdict", {}).get("value", "")
            result.ocr_text = forensics_result.get("ocr_text", "")
            result.objects_detected = forensics_result.get("objects_detected", [])
            result.extracted_claims = forensics_result.get("extracted_claims", [])
            result.numeric_values = forensics_result.get("numeric_values", [])
            result.detailed_caption = forensics_result.get("detailed_caption", "")
            result.is_out_of_context = forensics_result.get("is_out_of_context", False)
            result.context_trust = forensics_result.get("context_trust", 1.0)
            result.mismatched_entities = forensics_result.get("mismatched_entities", [])
            result.per_claim_clip = forensics_result.get("per_claim_clip", [])
            result.faces_found = forensics_result.get("faces_found", False)
            result.deepfake_probability = forensics_result.get("deepfake_probability", 0.0)
            result.watermark_type = forensics_result.get("watermark_type", "")
            visual_facts = forensics_result

            adv_caught = sum(1 for c in claims if c.api_judge_used)
            forensics_msg = forensics_result.get("caption", "")

        except Exception as e:
            logger.error("Forensics failed: %s", e)

        # --- Tier 3: Cross-modal consistency (CMCD) ---
        if tier >= 3 and visual_facts and text:
            try:
                ocr_claims_text = result.ocr_text
                ocr_passage = ". ".join(result.extracted_claims)
                if ocr_passage:
                    ocr_verified = await verify_text(ocr_passage, use_api_judge=use_api_judge)
                    for c in ocr_verified:
                        if c not in result.claims:
                            result.claims.append(c)

                effective_caption = caption or result.image_description
                cmcd = await run_cmcd(text, safe_path, effective_caption, visual_facts)
                result.clip_similarity = cmcd.get("clip_similarity", 0.0)
                result.contradictions_found = cmcd.get("contradiction_count", 0) > 0
                result.contradiction_severity = cmcd.get("avg_severity", 0.0)
                result.cross_modal_trust = cmcd.get("cross_modal_trust", 1.0)
                result.matched_elements = cmcd.get("matched_elements", [])

            except Exception as e:
                logger.error("CMCD failed: %s", e)

    # Compute overall trust across modalities
    result.compute_overall()

    # Collect any remaining suspicion flags
    flags: list[str] = []
    for c in result.claims:
        if c.suspicion_flag.value != "NONE":
            flags.append(c.suspicion_flag.value)
    if flags and not result.active_suspicion_flags:
        result.active_suspicion_flags = flags

    result.latency_ms = budget.elapsed_ms
    return result
