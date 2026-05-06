"""Unit tests for policy engine and trust score."""
import pytest
from hallucination_guard.text.policy_engine import decide_policy
from hallucination_guard.trust_score import Policy


def test_policy_publish():
    assert decide_policy(0.75) == Policy.PUBLISH
    assert decide_policy(0.70) == Policy.PUBLISH
    assert decide_policy(1.00) == Policy.PUBLISH


def test_policy_flag():
    assert decide_policy(0.60) == Policy.FLAG
    assert decide_policy(0.40) == Policy.FLAG


def test_policy_reject():
    assert decide_policy(0.39) == Policy.REJECT
    assert decide_policy(0.00) == Policy.REJECT


def test_policy_boundary():
    """Boundary values sit in correct buckets."""
    assert decide_policy(0.699) == Policy.FLAG
    assert decide_policy(0.399) == Policy.REJECT
