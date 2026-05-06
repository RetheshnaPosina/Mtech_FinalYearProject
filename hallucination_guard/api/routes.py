"""FastAPI route handlers for all AMADA endpoints."""
from __future__ import annotations

import logging
import time
import uuid
import collections
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

from hallucination_guard.api.schemas import (
    TextVerifyRequest,
    ImageVerifyRequest,
    FullVerifyRequest,
    VerifyResponse,
    MetricsResponse,
    ClaimResultOut,
)
from hallucination_guard.cascade.cascade_executor import execute
from hallucination_guard.audit.logger import log_result
from hallucination_guard.audit.metrics import full_metrics_report
from hallucination_guard.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

_rate_counts: Dict[str, List[float]] = collections.defaultdict(list)
_rate_lock = Lock()
_RATE_WINDOW_S = 60.0

_ALLOWED_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jfif", ".jpe",          # JPEG family (incl. JFIF)
    ".png",                                     # PNG
    ".webp",                                    # WebP
    ".bmp",                                     # Bitmap
    ".gif",                                     # GIF
    ".tiff", ".tif",                            # TIFF
    ".heic", ".heif",                           # HEIC/HEIF (Apple)
    ".avif",                                    # AVIF
    ".ico",                                     # ICO
    ".raw", ".cr2", ".nef", ".dng",             # Camera RAW formats
}


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    with _rate_lock:
        times = [t for t in _rate_counts[client_ip] if now - t < _RATE_WINDOW_S]
        _rate_counts[client_ip] = times
        if len(times) >= settings.api_rate_limit_rpm:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        _rate_counts[client_ip].append(now)


def _validate_image_path(image_path: str) -> str:
    """Resolve and validate image_path against traversal and unsupported formats."""
    try:
        resolved = Path(image_path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image path")
    if ".." in resolved.parts:
        raise HTTPException(status_code=400, detail="Invalid image path")
    if not resolved.exists():
        raise HTTPException(status_code=400, detail="Image file not found")
    if resolved.suffix.lower() not in _ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    return str(resolved)


def _to_response(result) -> VerifyResponse:
    claims_out = [
        ClaimResultOut(
            claim=c.claim,
            verdict=c.verdict.value,
            raw_trust=round(c.raw_trust, 4),
            calibrated_trust=round(c.calibrated_trust, 4),
            difficulty_score=c.difficulty_score,
            adversarial_score=c.adversarial_score,
            prosecutor_confidence=c.prosecutor_confidence,
            defender_confidence=c.defender_confidence,
            judge_reasoning=c.judge_reasoning,
            best_alt_hypothesis=c.best_alt_hypothesis,
            suspicion_flag=c.suspicion_flag.value if hasattr(c.suspicion_flag, "value") else str(c.suspicion_flag),
            correction_suggestion=c.correction_suggestion,
            evidence_snippets=c.evidence_snippets,
            nli_entailment=round(c.nli_entailment, 4),
            nli_contradiction=round(c.nli_contradiction, 4),
            debate_rounds=c.debate_rounds,
            api_judge_used=c.api_judge_used,
        )
        for c in result.claims
    ]
    return VerifyResponse(
        overall_trust=round(result.overall_trust, 4),
        policy=result.policy.value,
        tier_used=result.tier_used,
        latency_ms=result.latency_ms,
        api_calls_made=result.api_calls_made,
        claims=claims_out,
        fact_score=result.fact_score,
        awp_fact_score=result.awp_fact_score,
        adversarial_detection_rate=result.adversarial_detection_rate,
        active_suspicion_flags=result.active_suspicion_flags,
        cnn_probability=result.cnn_probability,
        ela_energy=result.ela_energy,
        fft_score=result.fft_score,
        image_fusion_score=result.image_fusion_score,
        has_periodic_artifacts=result.has_periodic_artifacts,
        image_verdict=result.image_verdict,
        image_description=result.image_description,
        clip_similarity=result.clip_similarity,
        contradictions_found=result.contradictions_found,
        contradiction_severity=result.contradiction_severity,
        cross_modal_trust=result.cross_modal_trust,
        matched_elements=result.matched_elements,
        ocr_text=result.ocr_text,
        objects_detected=result.objects_detected,
        extracted_claims=result.extracted_claims,
        numeric_values=result.numeric_values,
        detailed_caption=result.detailed_caption,
        is_out_of_context=result.is_out_of_context,
        context_trust=result.context_trust,
        mismatched_entities=result.mismatched_entities,
        per_claim_clip=result.per_claim_clip,
        faces_found=result.faces_found,
        deepfake_probability=result.deepfake_probability,
        watermark_type=result.watermark_type,
    )


@router.post("/verify/text", summary="Verify text claims via AWP debate")
async def verify_text(req: TextVerifyRequest, request: Request) -> VerifyResponse:
    _check_rate_limit(request.client.host)
    rid = str(uuid.uuid4())[:8]
    try:
        result = await execute(text=req.text, use_api_judge=req.use_api_judge)
        log_result(result, request_id=rid)
        return _to_response(result)
    except Exception as e:
        logger.error("verify_text error [%s]: %s", rid, e)
        raise HTTPException(status_code=500, detail="Verification failed")


@router.post("/verify/image", summary="Verify image authenticity via ELA+FFT+CNN")
async def verify_image(req: ImageVerifyRequest, request: Request) -> VerifyResponse:
    _check_rate_limit(request.client.host)
    safe_path = _validate_image_path(req.image_path)
    rid = str(uuid.uuid4())[:8]
    try:
        result = await execute(image_path=safe_path, caption=req.caption, use_api_judge=req.use_api_judge)
        log_result(result, request_id=rid)
        return _to_response(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("verify_image error [%s]: %s", rid, e)
        raise HTTPException(status_code=500, detail="Verification failed")


@router.post("/verify/image/upload", summary="Verify image via file upload")
async def verify_image_upload(
    request: Request,
    file: UploadFile = File(...),
    caption: str = Form(default=""),
    use_api_judge: bool = Form(default=False),
) -> VerifyResponse:
    _check_rate_limit(request.client.host)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    rid = str(uuid.uuid4())[:8]
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        result = await execute(image_path=tmp_path, caption=caption or None, use_api_judge=use_api_judge)
        log_result(result, request_id=rid)
        return _to_response(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("verify_image_upload error [%s]: %s", rid, e)
        raise HTTPException(status_code=500, detail="Verification failed")
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


@router.post("/verify/full/upload", summary="Full multimodal verification via file upload")
async def verify_full_upload(
    request: Request,
    file: UploadFile = File(...),
    text: str = Form(default=""),
    caption: str = Form(default=""),
    use_api_judge: bool = Form(default=False),
) -> VerifyResponse:
    _check_rate_limit(request.client.host)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    rid = str(uuid.uuid4())[:8]
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        result = await execute(
            text=text or None,
            image_path=tmp_path,
            caption=caption or None,
            use_api_judge=use_api_judge,
        )
        log_result(result, request_id=rid)
        return _to_response(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("verify_full_upload error [%s]: %s", rid, e)
        raise HTTPException(status_code=500, detail="Verification failed")
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


@router.post("/verify/full", summary="Full multimodal verification (AWP+CMCD+VCADE)")
async def verify_full(req: FullVerifyRequest, request: Request) -> VerifyResponse:
    if not req.text and not req.image_path:
        raise HTTPException(status_code=422, detail="Provide text or image_path")
    _check_rate_limit(request.client.host)
    safe_image = None
    if req.image_path:
        safe_image = _validate_image_path(req.image_path)
    rid = str(uuid.uuid4())[:8]
    try:
        result = await execute(
            text=req.text,
            image_path=safe_image,
            caption=req.caption,
            use_api_judge=req.use_api_judge,
        )
        log_result(result, request_id=rid)
        return _to_response(result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("verify_full error [%s]: %s", rid, e)
        raise HTTPException(status_code=500, detail="Verification failed")


@router.get("/metrics", summary="System metrics: latency, tiers, AWP rates")
def get_metrics() -> MetricsResponse:
    report = full_metrics_report()
    return MetricsResponse(**report)


@router.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok", "version": "6.0.0", "system": "AMADA"}
