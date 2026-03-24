"""VCADE — Versatile Calibration and Difficulty Estimation.

Novel contribution: calibrates raw NLI trust scores using five difficulty
dimensions and (optionally) an isotonic regression calibrator fitted on
labelled audit logs, replacing hand-tuned linear weights with data-driven
calibration.

Difficulty dimensions
---------------------
d_retrieval   : 1 − avg_relevance  (high = poor evidence retrieval)
d_consensus   : min(1, stdev(relevances) × 2.5)  (high = sources disagree)
d_adversarial : best_alt_support  (high = strong alternative hypothesis exists)
d_entity      : min(1, entity_count × 0.25)  (high = many entities → confusion risk)
d_temporal    : 0.65 if has_temporal else 0.1  (high = time-sensitive claim)

Calibration
-----------
difficulty = Σ(vcade_w_* × d_*)   (from settings, default hand-tuned values)
calibrated_trust:
    SUPPORTED    → raw × (1 − 0.2 × difficulty)
    NEI          → raw × (1 − 0.3 × difficulty)
    REFUTED      → raw × difficulty

Isotonic enhancement (calibrate_from_logs):
    Fits an IsotonicRegression on (raw_scores, ground_truth) pairs extracted
    from audit logs and returns a Calibrator that replaces the formula above.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Optional

from hallucination_guard.trust_score import EvidenceItem, SuspicionFlag
from hallucination_guard.config import settings


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VCADEResult:
    d_retrieval: float
    d_consensus: float
    d_adversarial: float
    d_entity: float
    d_temporal: float
    difficulty: float
    raw_trust: float
    calibrated_trust: float
    suspicion_flag: SuspicionFlag
    verdict_label: str


# ---------------------------------------------------------------------------
# Core VCADE computation  (matches original bytecode-derived logic)
# ---------------------------------------------------------------------------

def compute_vcade(
    raw_trust: float,
    verdict_label: str,
    evidence: List[EvidenceItem],
    best_alt_support: float,
    entity_count: int,
    has_temporal: bool,
    _calibrator=None,           # optional fitted IsotonicRegression
) -> VCADEResult:
    """Compute VCADE calibrated trust and difficulty score.

    Parameters
    ----------
    raw_trust        : judge confidence score before calibration [0, 1]
    verdict_label    : 'SUPPORTED' | 'REFUTED' | 'NOT_ENOUGH_INFO'
    evidence         : top-k EvidenceItems from shared pool
    best_alt_support : AWP best alternative support score [0, 1]
    entity_count     : number of ambiguous entities detected by Investigator
    has_temporal     : whether claim has a temporal dimension
    _calibrator      : optional fitted sklearn IsotonicRegression (from
                       calibrate_from_logs). When provided, replaces the
                       formula-based calibration step.
    """

    # ---- d_retrieval: poor evidence retrieval = high difficulty ----
    if not evidence:
        d_retrieval = 1.0
    else:
        avg_rel = sum(e.relevance for e in evidence) / len(evidence)
        d_retrieval = max(0.0, 1.0 - min(avg_rel, 1.0))

    # ---- d_consensus: high variance = sources disagree ----
    if len(evidence) < 2:
        d_consensus = 0.5
    else:
        rels = [e.relevance for e in evidence]
        try:
            d_consensus = min(1.0, statistics.stdev(rels) * 2.5)
        except statistics.StatisticsError:
            d_consensus = 0.5

    # ---- d_adversarial: strong alternative = high difficulty ----
    d_adversarial = float(best_alt_support)

    # ---- d_entity: many entities → confusion risk ----
    d_entity = min(1.0, entity_count * 0.25)

    # ---- d_temporal: time-sensitive claims are harder ----
    d_temporal = 0.65 if has_temporal else 0.1

    # ---- Weighted difficulty score (weights from settings / config) ----
    cfg = settings
    difficulty = (
        cfg.vcade_w_retrieval   * d_retrieval
        + cfg.vcade_w_consensus   * d_consensus
        + cfg.vcade_w_adversarial * d_adversarial
        + cfg.vcade_w_entity      * d_entity
        + cfg.vcade_w_temporal    * d_temporal
    )
    difficulty = min(1.0, max(0.0, difficulty))

    # ---- Calibration ----
    if _calibrator is not None:
        # Data-driven path: isotonic regression fitted on audit logs
        calibrated = float(_calibrator.predict([[raw_trust, difficulty]])[0])
    elif verdict_label == "SUPPORTED":
        # SUPPORTED: harder claim (more adversarial) → more downward adjustment
        calibrated = raw_trust * (1.0 - 0.2 * difficulty)
    elif verdict_label == "NOT_ENOUGH_INFO":
        # NEI: larger adjustment for harder claims — high difficulty = low confidence
        calibrated = raw_trust * (1.0 - 0.3 * difficulty)
    else:
        # REFUTED: scale by difficulty (easy fact failed = very suspicious)
        calibrated = raw_trust * difficulty

    calibrated = min(1.0, max(0.0, calibrated))

    # ---- Suspicion flag ----
    flag = SuspicionFlag.NONE
    if difficulty < 0.3 and verdict_label != "SUPPORTED":
        flag = SuspicionFlag.HIGH_SUSPICION
    elif has_temporal and verdict_label == "REFUTED":
        flag = SuspicionFlag.TEMPORAL_STALENESS
    elif entity_count > 4 and verdict_label != "SUPPORTED":
        flag = SuspicionFlag.ENTITY_CONFUSION

    return VCADEResult(
        d_retrieval=d_retrieval,
        d_consensus=d_consensus,
        d_adversarial=d_adversarial,
        d_entity=d_entity,
        d_temporal=d_temporal,
        difficulty=difficulty,
        raw_trust=raw_trust,
        calibrated_trust=calibrated,
        suspicion_flag=flag,
        verdict_label=verdict_label,
    )


# ---------------------------------------------------------------------------
# Isotonic calibrator training  (novel improvement over hand-tuned weights)
# ---------------------------------------------------------------------------

def calibrate_from_logs(audit_log_path: str = "./audit_logs/audit.jsonl"):
    """Fit an IsotonicRegression calibrator from labelled audit log entries.

    Entries with ground-truth labels (if present) are used as training data.
    Returns a fitted calibrator that can be passed as ``_calibrator`` to
    :func:`compute_vcade`, replacing the hand-tuned formula.

    Falls back gracefully when sklearn is unavailable or no labelled data
    exists — returns None in that case (formula-based calibration is used).

    Usage
    -----
    >>> cal = calibrate_from_logs()
    >>> result = compute_vcade(..., _calibrator=cal)
    """
    try:
        from sklearn.isotonic import IsotonicRegression
        import json
    except ImportError:
        return None

    X, y = [], []
    try:
        with open(audit_log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for claim in entry.get("claims", []):
                    # Only use entries that have been manually labelled
                    if "ground_truth" not in claim:
                        continue
                    raw = claim.get("calibrated_trust", entry.get("overall_trust", 0.5))
                    diff = claim.get("difficulty", 0.5)
                    label = 1.0 if claim["ground_truth"] == claim.get("verdict") else 0.0
                    X.append([raw, diff])
                    y.append(label)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if len(X) < 10:
        # Not enough labelled data — fall back to formula
        return None

    # Isotonic regression on the 1-D raw_trust projection
    # (difficulty used as a feature via a simple linear combination)
    raw_scores = [x[0] * (1 - 0.3 * x[1]) for x in X]   # feature: formula output
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_scores, y)

    # Wrap so compute_vcade can call .predict([[raw_trust, difficulty]])
    class _Wrapper:
        def predict(self, Xy):
            scores = [x[0] * (1 - 0.3 * x[1]) for x in Xy]
            return ir.predict(scores)

    return _Wrapper()
