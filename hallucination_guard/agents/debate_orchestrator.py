"""AMADA Debate Orchestrator — parallel agents, argument-graph memory, VCADE calibration.

Argument-Graph Memory (novelty contribution):
    Round 1: Prosecutor, Defender, and Investigator run in parallel.
    After Round 1 the orchestrator extracts each agent's strongest_point —
    the single most compelling argument that agent produced.
    Round 2 (triggered when confidence gap > judge_api_disagreement_threshold):
        - Prosecutor receives Defender's strongest_point as counter_argument
        - Defender receives Prosecutor's strongest_point as counter_argument
    Each Round-2 agent explicitly searches for evidence that counters the
    opposing side's best argument, turning the second round into a true
    adversarial rebuttal rather than a repeated independent inference.

Neutral Seed Retrieval (novelty contribution — fixes EvidencePool bias):
    Before agents run, the orchestrator fetches evidence using a plain,
    unframed query (just the raw claim text) and pre-populates the pool.
    This ensures both Prosecutor and Defender reason over a common floor
    of unbiased evidence rather than evidence shaped by whoever locks the
    pool first.  Agents still add their own targeted counter-evidence on top.
    See: EvidencePool bias analysis (2026-03-21).
"""
from __future__ import annotations

import asyncio
from typing import List, Optional

from hallucination_guard.trust_score import ClaimResult, SuspicionFlag, Verdict
from hallucination_guard.text.evidence_pool import EvidencePool
from hallucination_guard.text.claim_extractor import extract_claims
from hallucination_guard.text.vcade_calibrator import compute_vcade
from hallucination_guard.text.policy_engine import decide_policy
from hallucination_guard.agents.prosecutor_agent import ProsecutorAgent
from hallucination_guard.agents.defender_agent import DefenderAgent
from hallucination_guard.agents.investigator_agent import InvestigatorAgent
from hallucination_guard.agents.judge_agent import judge
from hallucination_guard.config import settings

_prosecutor = ProsecutorAgent()
_defender = DefenderAgent()
_investigator = InvestigatorAgent()


async def debate_claim(
    claim_text: str,
    visual_facts: dict | None = None,
    use_api_judge: bool = True,
) -> ClaimResult:
    """Full adversarial debate on one claim with argument-graph memory across rounds."""

    pool = EvidencePool()
    claim_key = claim_text[:64]

    # ------------------------------------------------------------------
    # Neutral seed retrieval (novelty: prevents query-framing bias)
    # Fetch unframed evidence BEFORE agents run so neither agent's query
    # shapes the shared pool.  retrieve_evidence is imported lazily to
    # avoid circular imports with the agent modules.
    # ------------------------------------------------------------------
    try:
        from hallucination_guard.text.evidence_retriever import retrieve_evidence
        neutral_items = await retrieve_evidence(claim_text, top_k=settings.evidence_top_k)
        pool.pre_populate(claim_key, neutral_items)
    except Exception:
        pass  # neutral seed is best-effort; debate proceeds without it

    # ------------------------------------------------------------------
    # Round 1: Prosecutor + Defender + Investigator run in parallel
    # ------------------------------------------------------------------
    prosecutor_msg, defender_msg, investigator_msg = await asyncio.gather(
        _prosecutor.run(claim_text, pool, visual_facts),
        _defender.run(claim_text, pool, visual_facts),
        _investigator.run(claim_text, pool, visual_facts),
    )

    all_messages = [prosecutor_msg, defender_msg, investigator_msg]
    debate_rounds = 1

    # ------------------------------------------------------------------
    # Convergence check — do agents agree?
    # ------------------------------------------------------------------
    gap = abs(prosecutor_msg.confidence - defender_msg.confidence)

    if gap > settings.judge_api_disagreement_threshold:
        # ------------------------------------------------------------------
        # Round 2: Argument-graph memory — each agent counters the other's
        # strongest point from Round 1 rather than repeating the same search.
        # ------------------------------------------------------------------
        prosecutor_counter = defender_msg.strongest_point    # Defender's best → Prosecutor counters
        defender_counter = prosecutor_msg.strongest_point    # Prosecutor's best → Defender rebuts

        prosecutor_msg2, defender_msg2 = await asyncio.gather(
            _prosecutor.run(claim_text, pool, visual_facts, counter_argument=prosecutor_counter),
            _defender.run(claim_text, pool, visual_facts, counter_argument=defender_counter),
        )
        all_messages = [
            prosecutor_msg, defender_msg,
            prosecutor_msg2, defender_msg2,
            investigator_msg,
        ]
        debate_rounds = 2

    # ------------------------------------------------------------------
    # Judge verdict
    # ------------------------------------------------------------------
    verdict, confidence, reasoning, api_used = judge(
        claim=claim_text,
        agent_messages=all_messages,
        use_api=use_api_judge or (gap > settings.judge_api_disagreement_threshold),
    )

    # ------------------------------------------------------------------
    # Retrieve best evidence and VCADE calibration
    # Use credibility-weighted ranking so high-authority sources are
    # preferred over low-credibility but superficially relevant results.
    # ------------------------------------------------------------------
    top_evidence = pool.credibility_weighted_top_k(claim_key, k=settings.evidence_top_k)

    # AWP scores come from the prosecutor messages (use Round 2 if available)
    latest_prosecutor = (
        all_messages[2] if debate_rounds == 2 else prosecutor_msg
    )

    vcade_result = compute_vcade(
        raw_trust=confidence,
        verdict_label=verdict.value,
        evidence=top_evidence,
        best_alt_support=latest_prosecutor.best_alt_support,
        entity_count=sum(
            1 for msg in all_messages
            if hasattr(msg, "suspicion_flags")
            and SuspicionFlag.ENTITY_CONFUSION in msg.suspicion_flags
        ),
        has_temporal=any(
            SuspicionFlag.TEMPORAL_STALENESS in (msg.suspicion_flags or [])
            for msg in all_messages
        ),
    )

    # ------------------------------------------------------------------
    # Build ClaimResult
    # ------------------------------------------------------------------
    suspicion_flag = vcade_result.suspicion_flag

    # Best adversarial hypothesis for correction suggestion
    best_alt_hypothesis = latest_prosecutor.adversarial_hypotheses[0] \
        if latest_prosecutor.adversarial_hypotheses else ""
    correction_suggestion = (
        f"Consider: {best_alt_hypothesis}" if best_alt_hypothesis else ""
    )

    nli_entailment = (
        sum(e.relevance for e in top_evidence) / len(top_evidence)
        if top_evidence else 0.0
    )
    nli_contradiction = max(
        0.0, 1.0 - nli_entailment - 0.2
    ) if top_evidence else 0.0

    return ClaimResult(
        claim=claim_text,
        verdict=verdict,
        raw_trust=confidence,
        calibrated_trust=vcade_result.calibrated_trust,
        difficulty_score=vcade_result.difficulty,
        adversarial_score=latest_prosecutor.best_alt_support,
        prosecutor_confidence=latest_prosecutor.confidence,
        defender_confidence=defender_msg.confidence
        if debate_rounds == 1 else all_messages[3].confidence,
        judge_reasoning=reasoning,
        best_alt_hypothesis=best_alt_hypothesis,
        best_alt_support=latest_prosecutor.best_alt_support,
        suspicion_flag=suspicion_flag,
        correction_suggestion=correction_suggestion,
        evidence_snippets=[e.text[:120] for e in top_evidence],
        nli_entailment=nli_entailment,
        nli_contradiction=nli_contradiction,
        debate_rounds=debate_rounds,
        api_judge_used=api_used,
    )


async def verify_text(
    text: str,
    visual_facts: dict | None = None,
    use_api_judge: bool = True,
) -> List[ClaimResult]:
    """Extract all claims and verify them concurrently."""
    claims = extract_claims(text, max_claims=settings.max_claims_per_request)
    results = await asyncio.gather(
        *[debate_claim(c.text, visual_facts, use_api_judge) for c in claims]
    )
    return list(results)
