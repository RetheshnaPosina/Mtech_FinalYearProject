"""Abstract base class for all AMADA agents."""
from __future__ import annotations
from abc import ABC, abstractmethod

from hallucination_guard.trust_score import AgentMessage, EvidenceItem, Verdict, SuspicionFlag
from hallucination_guard.text.evidence_pool import EvidencePool


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    async def run(
        self,
        claim: str,
        evidence_pool: EvidencePool,
        visual_facts: dict | None = None,
        counter_argument: str | None = None,
    ) -> AgentMessage:
        """Run the agent on a single claim and return a structured AgentMessage.

        Args:
            claim:            The claim text to evaluate.
            evidence_pool:    Shared pool of evidence (prevents search-luck bias).
            visual_facts:     Optional dict of visual facts from ForensicsAgent (CMCD).
            counter_argument: Round 2 only — the opposing agent's strongest point from
                              Round 1. The agent should search for evidence that
                              directly counters this specific argument.
        """

    def _make_message(
        self,
        claim: str,
        verdict: Verdict,
        confidence: float,
        evidence: list[EvidenceItem],
        reasoning: str,
        adversarial_hypotheses: list[str] | None = None,
        best_alt_support: float = 0.0,
        suspicion_flags: list[SuspicionFlag] | None = None,
        strongest_point: str = "",
    ) -> AgentMessage:
        return AgentMessage(
            agent=self.name,
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidence_used=evidence,
            adversarial_hypotheses=adversarial_hypotheses or [],
            best_alt_support=best_alt_support,
            reasoning=reasoning,
            suspicion_flags=suspicion_flags or [],
            strongest_point=strongest_point,
        )
