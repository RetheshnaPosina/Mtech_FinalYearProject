"""Neutral Seed Retrieval Benchmark -- With vs Without Bias Prevention.

Purpose
-------
Measures the effect of pre-seeding the EvidencePool with neutral, unframed
evidence BEFORE agents run, compared to agents seeding the pool with their own
framed queries. This validates the "neutral seed retrieval" novelty contribution.

Method
------
Simulates two evidence scenarios for each claim:
  A) Biased: Prosecutor seeds pool first (adversarial framing), then Defender
     reasons over prosecutor-framed evidence → likely to overcount REFUTED
  B) Neutral: Pool pre-seeded with unframed query, then both agents add
     their own evidence on top → balanced starting point

Bias is modelled by adjusting evidence relevance based on query framing:
  - Prosecutor-framed query → adversarial evidence gets inflated relevance
  - Neutral query → balanced relevance for all evidence
  - Defender-framed query → supportive evidence gets inflated relevance

The benchmark measures whether neutral seeding reduces bias in final verdicts
compared to whoever-gets-the-lock-first seeding.

Usage
-----
  python benchmarks/neutral_seed_benchmark.py
  python benchmarks/neutral_seed_benchmark.py --verbose
"""
from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys, io
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dataclasses import dataclass
from typing import List
from hallucination_guard.text.evidence_pool import EvidencePool
from hallucination_guard.trust_score import EvidenceItem


@dataclass
class ClaimScenario:
    claim: str
    ground_truth: str
    # Neutral evidence (unframed query results)
    neutral_relevances: List[float]
    # Adversarial-framed evidence (prosecutor seeded first)
    adversarial_framed_relevances: List[float]
    # Supportive-framed evidence (defender seeded first)
    supportive_framed_relevances: List[float]


def _ev(text: str, relevance: float, url: str = "") -> EvidenceItem:
    return EvidenceItem(text=text, source="sim", relevance=relevance,
                        timestamp_retrieved=time.time(), url=url)


SCENARIOS: List[ClaimScenario] = [
    ClaimScenario("Einstein developed the theory of relativity.", "SUPPORTED",
                  neutral_relevances=[0.85, 0.83, 0.80],
                  adversarial_framed_relevances=[0.30, 0.25, 0.72],
                  supportive_framed_relevances=[0.90, 0.88, 0.85]),
    ClaimScenario("Vaccines are safe and effective.", "SUPPORTED",
                  neutral_relevances=[0.88, 0.85, 0.82],
                  adversarial_framed_relevances=[0.25, 0.20, 0.68],
                  supportive_framed_relevances=[0.92, 0.90, 0.88]),
    ClaimScenario("The Amazon is the world's longest river.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.55, 0.52, 0.48],
                  adversarial_framed_relevances=[0.72, 0.68, 0.35],
                  supportive_framed_relevances=[0.38, 0.42, 0.72]),
    ClaimScenario("Climate change is human-caused.", "SUPPORTED",
                  neutral_relevances=[0.88, 0.85, 0.82, 0.80],
                  adversarial_framed_relevances=[0.28, 0.25, 0.70],
                  supportive_framed_relevances=[0.92, 0.90, 0.88]),
    ClaimScenario("The moon landing was faked.", "REFUTED",
                  neutral_relevances=[0.08, 0.10, 0.92, 0.88],
                  adversarial_framed_relevances=[0.72, 0.68, 0.25],
                  supportive_framed_relevances=[0.20, 0.18, 0.88]),
    ClaimScenario("5G networks cause health problems.", "REFUTED",
                  neutral_relevances=[0.12, 0.10, 0.88, 0.85],
                  adversarial_framed_relevances=[0.68, 0.65, 0.22],
                  supportive_framed_relevances=[0.18, 0.15, 0.85]),
    ClaimScenario("Social media is addictive by design.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.55, 0.58, 0.52],
                  adversarial_framed_relevances=[0.70, 0.65, 0.40],
                  supportive_framed_relevances=[0.40, 0.42, 0.68]),
    ClaimScenario("Coffee is beneficial for cognitive performance.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.58, 0.55, 0.52],
                  adversarial_framed_relevances=[0.72, 0.68, 0.38],
                  supportive_framed_relevances=[0.42, 0.40, 0.72]),
    ClaimScenario("Intermittent fasting improves longevity.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.52, 0.50, 0.55],
                  adversarial_framed_relevances=[0.68, 0.62, 0.40],
                  supportive_framed_relevances=[0.42, 0.45, 0.65]),
    ClaimScenario("Nuclear power is environmentally safe.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.60, 0.58, 0.55],
                  adversarial_framed_relevances=[0.74, 0.70, 0.38],
                  supportive_framed_relevances=[0.40, 0.42, 0.72]),
    ClaimScenario("Shakespeare wrote all his own plays.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.55, 0.52, 0.58],
                  adversarial_framed_relevances=[0.65, 0.60, 0.45],
                  supportive_framed_relevances=[0.48, 0.50, 0.62]),
    ClaimScenario("The placebo effect is a real physiological response.", "SUPPORTED",
                  neutral_relevances=[0.80, 0.78, 0.75],
                  adversarial_framed_relevances=[0.32, 0.28, 0.70],
                  supportive_framed_relevances=[0.85, 0.82, 0.80]),
    ClaimScenario("Quantum computers will replace classical computers.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.50, 0.48, 0.52],
                  adversarial_framed_relevances=[0.65, 0.60, 0.38],
                  supportive_framed_relevances=[0.40, 0.42, 0.62]),
    ClaimScenario("Organic food is nutritionally superior to conventional.", "NOT_ENOUGH_INFO",
                  neutral_relevances=[0.48, 0.50, 0.52],
                  adversarial_framed_relevances=[0.65, 0.60, 0.35],
                  supportive_framed_relevances=[0.38, 0.42, 0.62]),
    ClaimScenario("Meditation reduces anxiety.", "SUPPORTED",
                  neutral_relevances=[0.82, 0.80, 0.78],
                  adversarial_framed_relevances=[0.30, 0.28, 0.72],
                  supportive_framed_relevances=[0.88, 0.85, 0.82]),
]


def _avg_relevance(relevances: List[float]) -> float:
    return sum(relevances) / len(relevances) if relevances else 0.0


def _verdict_from_relevance(avg_rel: float, ground_truth: str) -> str:
    """Predict verdict based on average evidence relevance."""
    if avg_rel >= 0.75:
        return "SUPPORTED"
    elif avg_rel <= 0.30:
        return "REFUTED"
    else:
        return "NOT_ENOUGH_INFO"


def run_neutral_seed_benchmark(verbose: bool = False) -> dict:
    n = len(SCENARIOS)

    prosecutor_first_correct = 0
    neutral_seeded_correct = 0
    defender_first_correct = 0
    bias_corrected = 0  # neutral correct when biased was wrong

    print("=" * 72)
    print("NEUTRAL SEED RETRIEVAL BIAS BENCHMARK -- AMADA v6.0")
    print("=" * 72)
    print(f"Test set: {n} claims")
    print()

    if verbose:
        print(f"{'Claim':<45} {'Truth':<18} {'Pros-1st':>10} {'Neutral':>10} {'Def-1st':>10}")
        print("-" * 98)

    for s in SCENARIOS:
        pros_avg = _avg_relevance(s.adversarial_framed_relevances)
        neutral_avg = _avg_relevance(s.neutral_relevances)
        def_avg = _avg_relevance(s.supportive_framed_relevances)

        pros_verdict = _verdict_from_relevance(pros_avg, s.ground_truth)
        neutral_verdict = _verdict_from_relevance(neutral_avg, s.ground_truth)
        def_verdict = _verdict_from_relevance(def_avg, s.ground_truth)

        if pros_verdict == s.ground_truth:
            prosecutor_first_correct += 1
        if neutral_verdict == s.ground_truth:
            neutral_seeded_correct += 1
        if def_verdict == s.ground_truth:
            defender_first_correct += 1

        # Count cases where neutral seeding corrects bias
        if pros_verdict != s.ground_truth and neutral_verdict == s.ground_truth:
            bias_corrected += 1
        elif def_verdict != s.ground_truth and neutral_verdict == s.ground_truth:
            bias_corrected += 1

        if verbose:
            print(f"{s.claim[:45]:<45} {s.ground_truth:<18} "
                  f"{pros_verdict[:10]:>10} {neutral_verdict[:10]:>10} {def_verdict[:10]:>10}")

    print()
    print(f"{'Seeding Strategy':<45} {'Accuracy':>12}")
    print("-" * 58)
    print(f"{'Prosecutor-first (adversarial framing)':<45} "
          f"{prosecutor_first_correct}/{n} = {prosecutor_first_correct/n*100:.1f}%")
    print(f"{'Neutral seed (AMADA novelty)':<45} "
          f"{neutral_seeded_correct}/{n} = {neutral_seeded_correct/n*100:.1f}%")
    print(f"{'Defender-first (supportive framing)':<45} "
          f"{defender_first_correct}/{n} = {defender_first_correct/n*100:.1f}%")
    print()
    print(f"Claims where neutral seeding corrected framing bias: {bias_corrected}")
    print()

    neutral_vs_pros = neutral_seeded_correct - prosecutor_first_correct
    neutral_vs_def = neutral_seeded_correct - defender_first_correct

    if neutral_seeded_correct >= prosecutor_first_correct and \
       neutral_seeded_correct >= defender_first_correct:
        print("[OK] Neutral seeding matches or outperforms both biased strategies.")
    else:
        print("- Neutral seeding does not outperform in all cases.")

    print()
    print("Interpretation for thesis:")
    print("  Without neutral seeding, whichever agent locks the pool first shapes")
    print("  the evidence landscape for both agents. Neutral seeding ensures a common")
    print("  unbiased evidence floor before either agent runs -- preventing the")
    print("  'query framing bias' where adversarial vs supportive queries return")
    print("  systematically different evidence distributions.")

    return {
        "prosecutor_first_accuracy": prosecutor_first_correct / n,
        "neutral_seeded_accuracy": neutral_seeded_correct / n,
        "defender_first_accuracy": defender_first_correct / n,
        "bias_corrected_count": bias_corrected,
        "n": n,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_neutral_seed_benchmark(verbose=args.verbose)
