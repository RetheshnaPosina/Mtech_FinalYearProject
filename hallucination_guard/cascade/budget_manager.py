"""Latency budget tracking per request."""
from __future__ import annotations

import time

from hallucination_guard.config import settings


class BudgetManager:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._start

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000

    def budget_for_tier(self, tier: int) -> float:
        budgets = {
            0: settings.tier0_budget_s,
            1: settings.tier1_budget_s,
            2: settings.tier2_budget_s,
            3: settings.tier3_budget_s,
        }
        return budgets.get(tier, settings.tier3_budget_s)

    def within_budget(self, tier: int) -> bool:
        return self.elapsed_s < self.budget_for_tier(tier)
