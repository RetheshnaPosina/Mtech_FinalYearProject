"""P50/P95/P99 latency, ECE computation, novel metrics tracking."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import List

from hallucination_guard.config import settings


def load_audit_entries() -> List[dict]:
    log_file = settings.audit_dir / "audit.jsonl"
    entries = []
    if not log_file.exists():
        return entries
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def compute_latency_percentiles(entries: List[dict]) -> dict:
    latencies = sorted(e.get("latency_ms", 0) for e in entries)
    n = len(latencies)
    if n == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": latencies[int(min(n * 0.5, n - 1))],
        "p95": latencies[int(min(n * 0.95, n - 1))],
        "p99": latencies[int(min(n * 0.99, n - 1))],
        "mean": statistics.mean(latencies),
    }


def compute_ece(
    calibrated_trusts: List[float],
    ground_truths: List[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — lower is better."""
    n = len(calibrated_trusts)
    if n == 0:
        return 0.0
    bin_size = 1.0 / n_bins
    ece = 0.0
    for b in range(n_bins):
        indices = [i for i, p in enumerate(calibrated_trusts) if b * bin_size <= p < (b + 1) * bin_size]
        if not indices:
            continue
        bin_confidence = statistics.mean(calibrated_trusts[i] for i in indices)
        bin_accuracy = statistics.mean(ground_truths[i] for i in indices)
        ece += len(indices) / n * abs(bin_confidence - bin_accuracy)
    return ece


def compute_tier_distribution(entries: List[dict]) -> dict:
    dist: dict = {}
    for e in entries:
        tier = e.get("tier_used", 0)
        dist[tier] = dist.get(tier, 0) + 1
    total = sum(dist.values())
    if total == 0:
        return {}
    return {t: round(c / total * 100, 2) for t, c in dist.items()}


def compute_awp_detection_rate(entries: List[dict]) -> float:
    """AWP-FActScore: fraction of claims that survived adversarial challenge."""
    all_claims: list = []
    for e in entries:
        all_claims.extend(e.get("claims", []))
    if not all_claims:
        return 0.0
    supported = sum(1 for c in all_claims if c.get("verdict") == "SUPPORTED")
    return supported / len(all_claims)


def full_metrics_report() -> dict:
    entries = load_audit_entries()
    total = len(entries)
    latency = compute_latency_percentiles(entries)
    tier_dist = compute_tier_distribution(entries)
    awp_rate = compute_awp_detection_rate(entries)
    total_api_calls = sum(e.get("api_calls_made", 0) for e in entries)
    total_claims = sum(e.get("claims_count", 0) for e in entries)
    api_call_rate = total_api_calls / max(total_claims, 1)
    return {
        "total_requests": total,
        "latency": latency,
        "tier_distribution": tier_dist,
        "awp_detection_rate": awp_rate,
        "api_call_rate": api_call_rate,
    }
