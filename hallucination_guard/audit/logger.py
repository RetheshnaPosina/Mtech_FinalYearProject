"""JSON-lines audit logger — every verification run logged for reproducibility."""
from __future__ import annotations

import json
import time
from pathlib import Path

from hallucination_guard.trust_score import TrustScore
from hallucination_guard.config import settings


def _ensure_dir() -> Path:
    d = settings.audit_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_result(result: TrustScore, request_id: str = "") -> None:
    audit_dir = _ensure_dir()
    log_file = audit_dir / "audit.jsonl"

    entry = {
        "request_id": request_id,
        "timestamp": time.time(),
        "tier_used": result.tier_used,
        "latency_ms": round(result.latency_ms, 4),
        "overall_trust": result.overall_trust,
        "policy": result.policy.value,
        "api_calls_made": result.api_calls_made,
        "claims_count": len(result.claims),
        "awp_fact_score": result.awp_fact_score,
        "adversarial_detection_rate": result.adversarial_detection_rate,
        "image_fusion_score": result.image_fusion_score,
        "cross_modal_trust": result.cross_modal_trust,
        "active_suspicion_flags": result.active_suspicion_flags,
        "claims": [
            {
                "claim": c.claim,
                "verdict": c.verdict.value,
                "calibrated_trust": round(c.calibrated_trust, 4),
                "difficulty_score": round(c.difficulty_score, 4),
                "api_judge_used": c.api_judge_used,
                "suspicion_flag": c.suspicion_flag.value if hasattr(c.suspicion_flag, "value") else str(c.suspicion_flag),
            }
            for c in result.claims
        ],
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
