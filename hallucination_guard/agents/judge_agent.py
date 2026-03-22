"""JUDGE Agent — SECURITY: never receives raw content. Only structured AgentMessages.
Primary: Claude API (ambiguous cases). Fallback: DeBERTa weighted voting.
"""
from __future__ import annotations

import json
import logging
from typing import List, Tuple

from hallucination_guard.trust_score import AgentMessage, Verdict
from hallucination_guard.config import settings
from hallucination_guard.security.key_manager import key_manager

logger = logging.getLogger(__name__)

# System prompt: deliberately does NOT include raw claim content — only structured summaries
_SYSTEM_PROMPT = (
    'You are a neutral fact-verification Judge in an adversarial multi-agent debate system.\n'
    'You receive ONLY structured arguments from specialist agents — never raw social media content.\n'
    'Your role: weigh evidence quality and reasoning, determine final verdict.\n\n'
    'Respond with ONLY valid JSON (no markdown):\n'
    '{"verdict":"SUPPORTED|REFUTED|NOT_ENOUGH_INFO","confidence":0.0,"reasoning":"2-3 sentences"}'
)

# DeBERTa-weighted voting weights for local vote
_WEIGHTS = {
    "prosecutor": 0.4,
    "defender": 0.35,
    "investigator": 0.25,
}


def _local_verdict(messages: List[AgentMessage]) -> Tuple[Verdict, float, str]:
    """Local DeBERTa-weighted voting across agent messages.

    Each agent's verdict contributes to per-label score weighted by
    agent role weight × agent confidence. Fallback when API unavailable.

    Returns
    -------
    (verdict, confidence, reasoning_json)
    """
    scores: dict = {}
    for msg in messages:
        w = _WEIGHTS.get(msg.agent, 0.1)
        label = msg.verdict.value
        scores[label] = scores.get(label, 0.0) + w * msg.confidence

    if not scores:
        return Verdict.NOT_ENOUGH_INFO, 0.3, "no_evidence"

    norm = sum(scores.values()) + 1e-6
    scores = {k: round(v / norm, 3) for k, v in scores.items()}
    best = max(scores, key=scores.__getitem__)

    reasoning = "local_vote: " + json.dumps(scores)
    return Verdict(best), scores[best], reasoning


def judge(
    claim: str,
    agent_messages: List[AgentMessage],
    use_api: bool = True,
) -> Tuple[Verdict, float, str, bool]:
    """Returns (verdict, confidence, reasoning, api_used). Secure: only structured args.

    Decision logic:
    1. Check confidence gap — if agents converge (gap < threshold), use local vote.
    2. If use_api=True and gap > threshold, try Claude API first.
    3. If Claude fails, try Gemini.
    4. If both APIs fail, fall back to local DeBERTa weighted vote.

    Parameters
    ----------
    claim         : The claim text (used for API judge context only — never raw content).
    agent_messages: Structured AgentMessage objects from Prosecutor, Defender, Investigator.
    use_api       : Whether to attempt API judge for ambiguous cases.

    Returns
    -------
    (Verdict, confidence float, reasoning str, api_used bool)
    """
    # Extract per-agent confidence
    prosecutor = next((m.confidence for m in agent_messages if m.agent == "prosecutor"), 0.5)
    defender = next((m.confidence for m in agent_messages if m.agent == "defender"), 0.5)
    total_evidence = sum(len(m.evidence_used) for m in agent_messages)

    gap = abs(prosecutor - defender)

    # Convergence check: agents agree → skip API
    if not use_api or gap <= settings.judge_api_disagreement_threshold:
        v, conf, reasoning = _local_verdict(agent_messages)
        return v, conf, "converged: " + reasoning, False

    # Try Claude API
    if key_manager.has_anthropic():
        try:
            v, conf, reasoning = _api_judge(claim, agent_messages)
            return v, conf, "[claude] " + reasoning, True
        except Exception as e:
            logger.warning("Claude judge failed, trying Gemini: %s", e)

    # Try Gemini API
    if key_manager.has_gemini():
        try:
            v, conf, reasoning = _gemini_judge(claim, agent_messages)
            return v, conf, "[gemini] " + reasoning, True
        except Exception as e:
            logger.warning("Gemini judge failed, falling back to local: %s", e)

    # Local fallback
    v, conf, reasoning = _local_verdict(agent_messages)
    return v, conf, "fallback: " + reasoning, False


def _api_judge(
    claim: str,
    agent_messages: List[AgentMessage],
) -> Tuple[Verdict, float, str]:
    """Call Claude API with structured agent summaries (never raw content).

    Fix #5: validates JSON response keys, verdict enum, confidence in [0,1].
    Fix #6: timeout enforced via asyncio.wait_for.
    """
    import anthropic
    import asyncio

    summaries = [
        {
            "agent": m.agent,
            "verdict": m.verdict.value,
            "confidence": round(m.confidence, 3),
            "reasoning": m.reasoning[:200],
            "evidence_count": len(m.evidence_used),
            "best_alt_support": round(m.best_alt_support, 3),
        }
        for m in agent_messages
    ]

    api_key = key_manager.get_anthropic_key()
    client = anthropic.Anthropic(api_key=api_key)

    loop = asyncio.get_event_loop()

    response = loop.run_in_executor(
        None,
        lambda: client.messages.create(
            model=settings.judge_model,
            max_tokens=300,
            system=_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": json.dumps({"claim_id": claim[:150], "agent_reports": summaries}),
            }],
        ),
    )
    response = loop.run_until_complete(asyncio.wait_for(response, timeout=settings.judge_api_timeout_s))

    raw = response.content[0].text.strip()

    # Fix #5: strip markdown code fences if present
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if "```" in raw:
        raw = raw.split("```")[0]

    data = json.loads(raw)

    # Fix #5: validate required keys
    required = {"verdict", "confidence", "reasoning"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError("Judge response missing required keys: " + str(missing))

    # Fix #5: validate verdict enum
    if data["verdict"] not in {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}:
        raise ValueError("Invalid verdict value: " + str(data["verdict"]))

    # Fix #5: validate confidence range
    conf = float(data["confidence"])
    if not (0.0 <= conf <= 1.0):
        raise ValueError("confidence out of [0,1]: " + str(conf))

    reasoning = data.get("reasoning", "")
    return Verdict(data["verdict"]), conf, reasoning


def _gemini_judge(
    claim: str,
    agent_messages: List[AgentMessage],
) -> Tuple[Verdict, float, str]:
    """Call Gemini API with structured agent summaries.

    Fix #5 and #6 apply here too: validates JSON, enforces timeout.
    """
    import google.generativeai as generativeai
    import asyncio

    summaries = [
        {
            "agent": m.agent,
            "verdict": m.verdict.value,
            "confidence": round(m.confidence, 3),
            "reasoning": m.reasoning[:200],
            "evidence_count": len(m.evidence_used),
        }
        for m in agent_messages
    ]

    prompt = _SYSTEM_PROMPT + "\n\n" + json.dumps({
        "claim_id": claim[:150],
        "agent_reports": summaries,
    })

    api_key = key_manager.get_gemini_key()
    loop = asyncio.get_event_loop()

    def _call():
        generativeai.configure(api_key=api_key)
        model = generativeai.GenerativeModel(settings.gemini_judge_model)
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 300, "temperature": 0.1},
        )
        return response.text.strip()

    raw = loop.run_until_complete(
        asyncio.wait_for(
            loop.run_in_executor(None, _call),
            timeout=settings.judge_api_timeout_s,
        )
    )

    # Strip markdown code fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if "```" in raw:
        raw = raw.split("```")[0]

    data = json.loads(raw)

    # Fix #5: validate
    required = {"verdict", "confidence", "reasoning"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError("Gemini judge response missing keys: " + str(missing))

    if data["verdict"] not in {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}:
        raise ValueError("Invalid verdict: " + str(data["verdict"]))

    conf = float(data["confidence"])
    if not (0.0 <= conf <= 1.0):
        raise ValueError("confidence out of [0,1]: " + str(conf))

    reasoning = data.get("reasoning", "")
    return Verdict(data["verdict"]), conf, reasoning
