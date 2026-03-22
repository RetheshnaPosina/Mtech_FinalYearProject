"""PROSECUTOR Agent — systematically tries to DISPROVE every claim (AWP).

Argument-graph memory (Round 2):
    When counter_argument is provided, the Prosecutor adds a targeted search
    query derived from the Defender's strongest supporting point.  This forces
    the Prosecutor to seek evidence that directly refutes that specific argument
    rather than re-running a generic adversarial sweep.
"""
from __future__ import annotations

from hallucination_guard.agents.base_agent import BaseAgent
from hallucination_guard.trust_score import AgentMessage, Verdict
from hallucination_guard.text.evidence_pool import EvidencePool
from hallucination_guard.text.claim_extractor import Claim, extract_claims
from hallucination_guard.text.adversarial_generator import generate_adversarial
from hallucination_guard.text.evidence_retriever import retrieve_evidence
from hallucination_guard.text.entailment_matrix import build_matrix
from hallucination_guard.text.awp_scorer import compute_awp_score
from hallucination_guard.config import settings


class ProsecutorAgent(BaseAgent):
    name = "prosecutor"

    async def run(
        self,
        claim: str,
        evidence_pool: EvidencePool,
        visual_facts: dict | None = None,
        counter_argument: str | None = None,
    ) -> AgentMessage:
        """Attempt to disprove the claim using adversarial weakness probing (AWP).

        Round 2 enhancement: if counter_argument is set (the Defender's strongest
        supporting point from Round 1), an additional targeted search is appended
        to look for evidence that contradicts that specific support.
        """
        claims = extract_claims(claim)
        claim_obj: Claim = claims[0] if claims else Claim(
            text=claim, has_number=False, has_entity=False,
            has_temporal=False, is_citation=False, entities=[], numbers=[],
        )

        # Generate adversarial hypotheses (negation, numeric, temporal, entity, citation)
        adv_hypotheses = generate_adversarial(claim_obj)

        # Build search queries from adversarial hypotheses
        queries = [adv.search_query for adv in adv_hypotheses[: settings.adversarial_top_k]]

        # Round 2 argument-graph: search for direct counter-evidence to Defender's best point
        if counter_argument:
            queries.append(f"evidence against: {counter_argument}")

        # Populate shared evidence pool (lock prevents duplicate fetches)
        claim_key = claim[:64]
        async with evidence_pool.fetch_lock(claim_key):
            if not evidence_pool.is_populated():
                for q in queries:
                    items = await retrieve_evidence(q, top_k=settings.evidence_top_k)
                    for item in items:
                        evidence_pool.add(claim_key, item)
            else:
                # Pool already seeded; fetch counter-evidence if new query introduced
                if counter_argument:
                    extra = await retrieve_evidence(
                        f"evidence against: {counter_argument}",
                        top_k=max(2, settings.evidence_top_k // 2),
                    )
                    for item in extra:
                        evidence_pool.add(claim_key, item)

        pool_evidence = evidence_pool.get(claim_key)

        # Build entailment matrix: original claim vs adversarial hypotheses
        matrix = build_matrix(claim, pool_evidence, adv_hypotheses)

        # AWP score: original_support / (original_support + best_alt_support)
        awp = compute_awp_score(matrix)
        adv_score: float = awp["adversarial_score"]
        best_alt_support: float = awp["best_alt_support"]
        best_alt_text: str = awp.get("best_alt_text", "")

        # Verdict: low AWP → evidence of refutation
        if adv_score < 0.35:
            verdict = Verdict.REFUTED
            confidence = 1.0 - adv_score
        elif adv_score > 0.72:
            verdict = Verdict.SUPPORTED
            confidence = adv_score
        else:
            verdict = Verdict.NOT_ENOUGH_INFO
            confidence = 0.5

        reasoning = (
            f"AWP score={adv_score:.3f}  "
            f"original_support={awp['original_support']:.3f}  "
            f"best_alt_support={best_alt_support:.3f}  "
            f"avg_contradiction={awp['avg_contradiction']:.3f}"
        )
        if counter_argument:
            reasoning = f"[Round 2 — countering: '{counter_argument[:80]}'] " + reasoning

        # strongest_point: the adversarial hypothesis that most threatened the claim
        strongest_point = (
            best_alt_text if best_alt_text
            else (adv_hypotheses[0].text if adv_hypotheses else "")
        )

        return self._make_message(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidence=pool_evidence[: settings.evidence_top_k],
            reasoning=reasoning,
            adversarial_hypotheses=[a.text for a in adv_hypotheses],
            best_alt_support=best_alt_support,
            strongest_point=strongest_point,
        )
