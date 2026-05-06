"""DEFENDER Agent — searches for confirming evidence to SUPPORT each claim.

Argument-graph memory (Round 2):
    When counter_argument is provided (the Prosecutor's strongest adversarial
    hypothesis from Round 1), the Defender adds a targeted search to find
    evidence that specifically rebuts that adversarial claim, rather than
    repeating a generic support sweep.
"""
from __future__ import annotations

from hallucination_guard.agents.base_agent import BaseAgent
from hallucination_guard.trust_score import AgentMessage, Verdict
from hallucination_guard.text.evidence_pool import EvidencePool
from hallucination_guard.text.evidence_retriever import retrieve_evidence
from hallucination_guard.text.entailment_matrix import build_matrix
from hallucination_guard.config import settings


class DefenderAgent(BaseAgent):
    name = "defender"

    async def run(
        self,
        claim: str,
        evidence_pool: EvidencePool,
        visual_facts: dict | None = None,
        counter_argument: str | None = None,
    ) -> AgentMessage:
        """Search for supporting evidence and score via NLI entailment.

        Round 2 enhancement: if counter_argument is set (the Prosecutor's best
        adversarial hypothesis from Round 1), the Defender adds a targeted
        rebuttal search: evidence that specifically disproves that adversarial claim.
        """
        claim_key = claim[:64]

        # Build supporting queries
        queries = [claim]
        if counter_argument:
            # Explicitly look for evidence that refutes the adversarial alternative
            queries.append(f"why '{counter_argument}' is incorrect")
            queries.append(f"evidence supporting original claim against: {counter_argument}")

        async with evidence_pool.fetch_lock(claim_key):
            if not evidence_pool.is_populated(claim_key):
                for q in queries:
                    items = await retrieve_evidence(q, top_k=settings.evidence_top_k)
                    for item in items:
                        evidence_pool.add(claim_key, item)
            else:
                # Pool seeded; add targeted rebuttal evidence if Round 2
                if counter_argument:
                    for q in queries[1:]:
                        extra = await retrieve_evidence(
                            q, top_k=max(2, settings.evidence_top_k // 2)
                        )
                        for item in extra:
                            evidence_pool.add(claim_key, item)

        pool_evidence = evidence_pool.get(claim_key)
        # Fix: exclude adversarial-tagged items (added by Prosecutor) from the
        # Defender's NLI matrix — they contaminate avg_ent and push it below 0.45.
        # Adversarial items are still in the pool for VCADE credibility ranking.
        supporting_evidence = [e for e in pool_evidence if not e.is_adversarial]
        if not supporting_evidence:
            supporting_evidence = pool_evidence  # fallback: use all if pool not tagged
        matrix = await build_matrix(claim, supporting_evidence, [])

        original_rows = [r for r in matrix if not r.is_adversarial]
        avg_ent = (
            sum(r.entailment for r in original_rows) / len(original_rows)
            if original_rows else 0.0
        )
        avg_contra = (
            sum(r.contradiction for r in original_rows) / len(original_rows)
            if original_rows else 0.0
        )

        if avg_ent > 0.45:
            # Lowered from 0.6 → 0.45: NLI entailment over a mixed evidence pool
            # rarely exceeds 0.6 even for well-supported claims because adversarial
            # queries pollute the shared pool with off-topic evidence.
            verdict = Verdict.SUPPORTED
            confidence = avg_ent
        elif avg_contra > 0.5:
            verdict = Verdict.REFUTED
            confidence = avg_contra
        else:
            verdict = Verdict.NOT_ENOUGH_INFO
            # Use actual avg_ent as confidence signal instead of hardcoded 0.5.
            # This prevents the Defender always contributing 0.5 to the judge vote,
            # which caused REFUTED to dominate even for well-evidenced true claims.
            confidence = max(0.4, avg_ent)

        # strongest_point: the evidence sentence with the highest entailment score
        strongest_row = max(original_rows, key=lambda r: r.entailment) if original_rows else None
        strongest_point = strongest_row.evidence_text if strongest_row else ""

        reasoning = (
            f"avg_entailment={avg_ent:.3f}  avg_contradiction={avg_contra:.3f}  "
            f"evidence_count={len(pool_evidence)}"
        )
        if counter_argument:
            reasoning = (
                f"[Round 2 — rebutting: '{counter_argument[:80]}'] " + reasoning
            )

        return self._make_message(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidence=pool_evidence[: settings.evidence_top_k],
            reasoning=reasoning,
            strongest_point=strongest_point,
        )
