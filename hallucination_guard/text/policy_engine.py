"""Policy engine: calibrated trust -> PUBLISH / FLAG / REJECT."""
from __future__ import annotations

from hallucination_guard.trust_score import Policy
from hallucination_guard.config import settings


def decide_policy(calibrated_trust: float) -> Policy:
    """Map calibrated trust score to publish policy.

    Thresholds (from settings):
      calibrated_trust >= publish_threshold (0.70) -> PUBLISH
      calibrated_trust >= flag_threshold   (0.40) -> FLAG
      else                                         -> REJECT

    Parameters
    ----------
    calibrated_trust : VCADE-calibrated trust score in [0, 1].

    Returns
    -------
    Policy enum value.
    """
    if calibrated_trust >= settings.publish_threshold:
        return Policy.PUBLISH
    elif calibrated_trust >= settings.flag_threshold:
        return Policy.FLAG
    else:
        return Policy.REJECT
