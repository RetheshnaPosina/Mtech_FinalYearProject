"""INVESTIGATOR Agent — source existence, temporal freshness, entity disambiguation, consensus.
Feeds difficulty dimensions to VCADE. Receives visual_facts from ForensicsAgent.
"""
from __future__ import annotations

import re

from hallucination_guard.agents.base_agent import BaseAgent
from hallucination_guard.trust_score import AgentMessage, EvidenceItem, SuspicionFlag, Verdict
from hallucination_guard.text.evidence_pool import EvidencePool
from hallucination_guard.text.evidence_retriever import retrieve_evidence
from hallucination_guard.text.claim_extractor import extract_claims

# Temporal keyword pattern for staleness detection
_TEMPORAL_RE = re.compile(
    r"\b(current|now|today|latest|recent|still|anymore"
    r"|as of \d{4}|january|february|march|april|may|june"
    r"|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)

# Citation pattern for verifiability detection
_CITATION_RE = re.compile(
    r"(study|report|paper|journal|according to|published in)",
    re.IGNORECASE,
)


class InvestigatorAgent(BaseAgent):
    name = "investigator"

    async def run(
        self,
        claim: str,
        evidence_pool: EvidencePool,
        visual_facts: dict | None = None,
        counter_argument: str | None = None,
    ) -> AgentMessage:
        """Investigate source credibility, temporal freshness, and entity disambiguation.

        Feeds difficulty signals to VCADE:
        - entity_count   → d_entity dimension
        - has_temporal   → d_temporal dimension
        - avg_relevance  → d_retrieval dimension (via evidence pool)

        Receives visual_facts from ForensicsAgent for CMCD cross-modal context.

        Parameters
        ----------
        claim          : Claim text to investigate.
        evidence_pool  : Shared evidence pool (read-only for investigator).
        visual_facts   : Optional dict from ForensicsAgent (CMCD fix).
        counter_argument : Not used by Investigator (single-round agent).
        """
        # Extract claim features
        claims = extract_claims(claim)
        c = claims[0] if claims else None

        entity_count = len(c.entities) if c else 0
        has_temporal = bool(_TEMPORAL_RE.search(claim))
        is_citation = bool(_CITATION_RE.search(claim))

        # Build suspicion flags
        suspicion_flags: list[SuspicionFlag] = []
        if has_temporal:
            suspicion_flags.append(SuspicionFlag.TEMPORAL_STALENESS)
        if is_citation:
            suspicion_flags.append(SuspicionFlag.CITATION_UNVERIFIABLE)
        if entity_count > 3:
            suspicion_flags.append(SuspicionFlag.ENTITY_CONFUSION)

        # Build targeted investigation queries
        queries: list[str] = []
        if has_temporal:
            queries.append("current 2024 2025 status: " + claim[:80])
        if is_citation:
            queries.append("verify publication source: " + claim[:80])
        if c and c.entities:
            for ent in c.entities[:3]:
                queries.append(f'"{ent}" correct facts')
        if visual_facts and visual_facts.get("caption"):
            queries.append(
                "image context: " + str(visual_facts.get("caption", ""))[:60]
                + " vs claim: " + claim[:60]
            )
        if not queries:
            queries.append("multiple sources: " + claim[:80])

        # Retrieve evidence (up to 4 queries)
        all_evidence: list[EvidenceItem] = []
        for q in queries[:4]:
            items = await retrieve_evidence(q, top_k=3)
            all_evidence.extend(items)
            for item in items:
                evidence_pool.add(claim[:64], item)

        # Compute average relevance for VCADE d_retrieval signal
        avg_rel = sum(e.relevance for e in all_evidence) / len(all_evidence) if all_evidence else 0.0

        # Verdict: high-confidence when good evidence found, no temporal/citation issues
        if avg_rel >= 0.6 and not suspicion_flags:
            verdict = Verdict.SUPPORTED
            confidence = avg_rel
        elif suspicion_flags or avg_rel < 0.2:
            verdict = Verdict.NOT_ENOUGH_INFO
            confidence = 0.45
        else:
            verdict = Verdict.NOT_ENOUGH_INFO
            confidence = 0.5

        reasoning = (
            f"entity_count={entity_count}, temporal={has_temporal}, "
            f"citation={is_citation}, sources={len(all_evidence)}, "
            f"avg_relevance={avg_rel:.3f}"
        )
        if visual_facts:
            reasoning += f", visual_facts={'yes' if visual_facts else 'no'}"

        msg = self._make_message(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidence=all_evidence[:5],
            reasoning=reasoning,
            suspicion_flags=suspicion_flags,
            strongest_point=reasoning,
        )
        return msg
