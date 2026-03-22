"""Shared evidence pool — Prosecutor and Defender draw from the same evidence.

Novelty improvements over v1:
1. Neutral seed retrieval pass (pre_populate):
   The debate_orchestrator calls pool.pre_populate() with a neutral query
   before either agent runs. This seeds the pool with unbiased evidence,
   preventing the first agent's query framing from shaping what the other
   agent reasons over.
   See: EvidencePool bias analysis (2026-03-21).

2. Source credibility scoring:
   source_credibility_score(url) assigns domain-authority weight [0.3, 1.0]
   using a tiered whitelist (academic/gov/major-news > Wikipedia > generic).
   credibility_weighted_top_k() re-ranks by relevance × credibility so
   low-credibility but superficially relevant pages don't dominate.

Fix #21 (preserved): per-claim fetch locks prevent concurrent duplicate fetches.
"""
from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Set

from hallucination_guard.trust_score import EvidenceItem


# ---------------------------------------------------------------------------
# Source credibility domain registry
# ---------------------------------------------------------------------------

_TIER1_DOMAINS: frozenset[str] = frozenset({
    # Academic / peer-reviewed
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "arxiv.org",
    "nature.com", "science.org", "sciencemag.org", "thelancet.com",
    "nejm.org", "bmj.com", "jamanetwork.com", "plos.org",
    "springer.com", "wiley.com", "elsevier.com", "ieee.org", "acm.org",
    # Government / intergovernmental
    "cdc.gov", "who.int", "nih.gov", "fda.gov", "nist.gov",
    "europa.eu", "un.org", "worldbank.org",
    # Major fact-checkers and newswires
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "theguardian.com", "washingtonpost.com",
    "economist.com", "ft.com", "bloomberg.com",
    "snopes.com", "factcheck.org", "politifact.com",
})

_TIER2_DOMAINS: frozenset[str] = frozenset({
    "wikipedia.org", "britannica.com", "merriam-webster.com",
    "statista.com", "ourworldindata.org", "pewresearch.org",
    "cnbc.com", "wsj.com", "forbes.com", "time.com",
    "nationalgeographic.com", "smithsonianmag.com",
    "nasa.gov", "noaa.gov", "usgs.gov",
    "stanford.edu", "harvard.edu", "mit.edu", "ox.ac.uk", "cam.ac.uk",
})


def source_credibility_score(url: str) -> float:
    """Return source credibility weight in [0.3, 1.0] based on domain authority.

    Tiers
    -----
    Tier 1 (academic, government, major fact-checkers, top press)  → 1.0
    Tier 2 (Wikipedia, reference sites, established outlets)       → 0.75
    .edu / .gov / .ac.* catchall not in explicit lists             → 0.80
    Tier 3 (general web, unrecognised domain)                      → 0.45
    No URL provided                                                 → 0.35
    """
    if not url:
        return 0.35
    domain = re.sub(r"^https?://", "", url.lower())
    domain = re.sub(r"^www\.", "", domain).split("/")[0]

    for t1 in _TIER1_DOMAINS:
        if domain == t1 or domain.endswith("." + t1):
            return 1.0
    for t2 in _TIER2_DOMAINS:
        if domain == t2 or domain.endswith("." + t2):
            return 0.75
    if any(domain.endswith(s) for s in (".edu", ".gov", ".ac.uk", ".ac.jp", ".gov.uk")):
        return 0.80
    return 0.45


# ---------------------------------------------------------------------------
# EvidencePool
# ---------------------------------------------------------------------------

class EvidencePool:
    """Thread-safe, claim-keyed evidence pool shared by all AMADA agents.

    Novelty additions
    -----------------
    pre_populate(claim_key, items)
        Seeds the pool with neutrally-retrieved evidence before any agent
        runs. Prevents the first agent's query framing from defining the
        full evidence landscape.

    credibility_weighted_top_k(claim_key, k)
        Re-ranks evidence by  0.7 × relevance + 0.3 × credibility  so that
        high-authority sources are preferred over low-credibility but
        superficially relevant results.
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[EvidenceItem]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._fetch_locks: Dict[str, asyncio.Lock] = {}
        self._populated: Set[str] = set()
        self._neutral_seeded: Set[str] = set()

    # ------------------------------------------------------------------
    # Locking helpers (Fix #21)
    # ------------------------------------------------------------------

    def fetch_lock(self, claim_key: str) -> asyncio.Lock:
        """Return a per-claim lock to prevent concurrent duplicate fetches."""
        if claim_key not in self._fetch_locks:
            self._fetch_locks[claim_key] = asyncio.Lock()
        return self._fetch_locks[claim_key]

    def is_populated(self, claim_key: str | None = None) -> bool:
        """Return True if evidence for this claim has already been fetched."""
        if claim_key is not None:
            return claim_key in self._populated
        return bool(self._populated)

    # ------------------------------------------------------------------
    # Neutral seed pass  (novelty: prevents query-framing bias)
    # ------------------------------------------------------------------

    def pre_populate(self, claim_key: str, items: List[EvidenceItem]) -> None:
        """Pre-seed pool with neutrally-retrieved evidence (no adversarial framing).

        Called by the debate_orchestrator BEFORE launching Prosecutor and
        Defender. Items are added but the claim key is NOT marked populated —
        agents still add their own targeted counter-evidence on top.  The
        neutral evidence simply provides an unbiased floor of signal.
        Idempotent: safe to call multiple times for the same claim.
        """
        if claim_key in self._neutral_seeded:
            return
        existing_texts: Set[str] = {e.text for e in self._store.get(claim_key, [])}
        for item in items:
            if item.text not in existing_texts:
                self._store.setdefault(claim_key, []).append(item)
                existing_texts.add(item.text)
        self._neutral_seeded.add(claim_key)

    # ------------------------------------------------------------------
    # Core add / get / top_k
    # ------------------------------------------------------------------

    def add(self, claim_key: str, item: EvidenceItem) -> None:
        """Add a single EvidenceItem, deduplicating by text."""
        existing_texts: Set[str] = {e.text for e in self._store.get(claim_key, [])}
        if item.text not in existing_texts:
            self._store.setdefault(claim_key, []).append(item)
        self._populated.add(claim_key)

    def get(self, claim_key: str) -> List[EvidenceItem]:
        """Return all stored evidence for this claim."""
        return list(self._store.get(claim_key, []))

    def top_k(self, claim_key: str, k: int = 5) -> List[EvidenceItem]:
        """Return top-k evidence sorted by relevance (descending)."""
        items = self._store.get(claim_key, [])
        return sorted(items, key=lambda e: e.relevance, reverse=True)[:k]

    def credibility_weighted_top_k(self, claim_key: str, k: int = 5) -> List[EvidenceItem]:
        """Return top-k evidence ranked by 0.7×relevance + 0.3×credibility.

        Novelty: high-authority sources receive a boost, reducing the
        influence of SEO-optimised low-credibility pages on verdict outcomes.
        """
        items = self._store.get(claim_key, [])
        return sorted(
            items,
            key=lambda e: 0.7 * e.relevance + 0.3 * source_credibility_score(e.url),
            reverse=True,
        )[:k]

    def size(self, claim_key: str | None = None) -> int:
        """Return total item count (or per-claim count if key given)."""
        if claim_key is not None:
            return len(self._store.get(claim_key, []))
        return sum(len(v) for v in self._store.values())
