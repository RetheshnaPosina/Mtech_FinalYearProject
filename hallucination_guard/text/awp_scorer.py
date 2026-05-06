"""AWP trust scoring — novel metric: adversarial_score = original / (original + best_alt).

Novelty improvement: learn_awp_thresholds()
--------------------------------------------
The original REFUTED threshold (0.35) and SUPPORTED threshold (0.72) were
hand-tuned with no empirical grounding. learn_awp_thresholds() scans the
audit log and finds the optimal (refuted_th, supported_th) pair via grid
search that maximises F1 on labelled entries, replacing intuition with
measurement. Falls back to the original hand-tuned values when insufficient
labelled data exists (<20 entries).

Reference: Platt, J. (1999) "Probabilistic Outputs for SVMs" — the same
principle of fitting decision boundaries post-hoc on validation data.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from hallucination_guard.text.entailment_matrix import MatrixRow

logger = logging.getLogger(__name__)

# Default hand-tuned thresholds (preserved from original)
_DEFAULT_REFUTED_TH: float = 0.35
_DEFAULT_SUPPORTED_TH: float = 0.72


# ---------------------------------------------------------------------------
# Core AWP scorer  (interface unchanged from original pyc)
# ---------------------------------------------------------------------------

def compute_awp_score(rows: List[MatrixRow]) -> dict:
    """Compute AWP score from an entailment matrix.

    Returns
    -------
    dict with keys:
        original_support  : mean entailment of original-claim rows
        best_alt_support  : entailment of the strongest adversarial row
        adversarial_score : original / (original + best_alt)  in [0, 1]
        avg_contradiction : mean contradiction of original rows
        best_alt_text     : hypothesis text of the best adversarial row
    """
    original = [r for r in rows if not r.is_adversarial]
    adversarial = [r for r in rows if r.is_adversarial]

    if not original:
        return {
            "original_support": 0.0,
            "best_alt_support": 0.0,
            "adversarial_score": 0.5,
            "avg_contradiction": 0.0,
            "best_alt_text": "",
        }

    orig_support = sum(r.entailment for r in original) / len(original)
    avg_contradiction = sum(r.contradiction for r in original) / len(original)

    best_alt_support: float = 0.0
    best_alt_text: str = ""
    if adversarial:
        best_row = max(adversarial, key=lambda r: r.entailment)
        best_alt_support = best_row.entailment
        best_alt_text = best_row.hypothesis

    denom = orig_support + best_alt_support
    adv_score = orig_support / denom if denom > 1e-6 else 0.5

    return {
        "original_support": orig_support,
        "best_alt_support": best_alt_support,
        "adversarial_score": adv_score,
        "avg_contradiction": avg_contradiction,
        "best_alt_text": best_alt_text,
    }


# ---------------------------------------------------------------------------
# Data-driven threshold learning  (novelty contribution)
# ---------------------------------------------------------------------------

def learn_awp_thresholds(
    audit_log_path: str = "./audit_logs/audit.jsonl",
    grid_steps: int = 20,
) -> Tuple[float, float]:
    """Learn optimal AWP decision thresholds from labelled audit log entries.

    Strategy
    --------
    Audit log entries that contain a "ground_truth" field are used as a
    labelled dataset. For each claim record we have:
        - adversarial_score  (proxy: we reconstruct from calibrated_trust
          and verdict to estimate where the AWP score likely fell)
        - ground_truth verdict  (SUPPORTED / REFUTED / NOT_ENOUGH_INFO)

    We then grid-search over (refuted_th, supported_th) pairs and select
    the pair that maximises macro-F1 over the three verdict classes.

    Falls back to hand-tuned defaults (0.35, 0.72) when:
        - fewer than 20 labelled entries exist
        - sklearn is not installed
        - audit log is missing or malformed

    Parameters
    ----------
    audit_log_path : path to the JSONL audit log
    grid_steps     : number of grid points per threshold axis (default 20)

    Returns
    -------
    (refuted_threshold, supported_threshold)
    """
    try:
        records = _load_labelled_records(audit_log_path)
    except Exception as exc:
        logger.warning("learn_awp_thresholds: could not load audit log: %s", exc)
        return _DEFAULT_REFUTED_TH, _DEFAULT_SUPPORTED_TH

    if len(records) < 20:
        logger.info(
            "learn_awp_thresholds: only %d labelled records found (need >=20); "
            "using defaults (%.2f, %.2f)",
            len(records), _DEFAULT_REFUTED_TH, _DEFAULT_SUPPORTED_TH,
        )
        return _DEFAULT_REFUTED_TH, _DEFAULT_SUPPORTED_TH

    # Grid search
    best_f1 = -1.0
    best_ref_th = _DEFAULT_REFUTED_TH
    best_sup_th = _DEFAULT_SUPPORTED_TH

    step = 1.0 / grid_steps
    for ref_idx in range(1, grid_steps):
        ref_th = ref_idx * step
        for sup_idx in range(ref_idx + 1, grid_steps + 1):
            sup_th = sup_idx * step
            if sup_th <= ref_th:
                continue
            f1 = _evaluate_thresholds(ref_th, sup_th, records)
            if f1 > best_f1:
                best_f1 = f1
                best_ref_th = ref_th
                best_sup_th = sup_th

    logger.info(
        "learn_awp_thresholds: best thresholds refuted=%.3f supported=%.3f "
        "(macro-F1=%.3f) from %d labelled records",
        best_ref_th, best_sup_th, best_f1, len(records),
    )
    return best_ref_th, best_sup_th


def _load_labelled_records(path: str) -> List[dict]:
    """Load audit log entries that have a 'ground_truth' field."""
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for claim in entry.get("claims", []):
                if "ground_truth" in claim:
                    # Use calibrated_trust as AWP score proxy:
                    # when verdict==SUPPORTED → awp_score ≈ calibrated_trust
                    # when verdict==REFUTED  → awp_score ≈ 1 - calibrated_trust
                    ct = claim.get("calibrated_trust", 0.5)
                    verdict = claim.get("verdict", "NOT_ENOUGH_INFO")
                    awp_proxy = ct if verdict == "SUPPORTED" else (1.0 - ct)
                    records.append({
                        "awp_score": awp_proxy,
                        "ground_truth": claim["ground_truth"],
                    })
    return records


def _evaluate_thresholds(
    refuted_th: float,
    supported_th: float,
    records: List[dict],
) -> float:
    """Return macro-F1 for given thresholds on labelled records."""
    classes = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    for r in records:
        score = r["awp_score"]
        truth = r["ground_truth"]

        if score < refuted_th:
            pred = "REFUTED"
        elif score > supported_th:
            pred = "SUPPORTED"
        else:
            pred = "NOT_ENOUGH_INFO"

        if pred == truth:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1

    f1s = []
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s)
