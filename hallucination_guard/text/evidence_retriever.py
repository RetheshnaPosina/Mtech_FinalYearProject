"""Evidence retrieval: Fact Check API -> Tavily -> Google -> Wikipedia -> mock store.

Source priority (highest signal first):
  1. Google Fact Check Tools API — directly indexes Snopes, PolitiFact, FactCheck.org
  2. Tavily Search API           — real-time web search, clean text, covers breaking news
  3. Google Custom Search API    — broad web search (requires key + engine ID)
  4. Wikipedia REST API          — background context, no key needed
  5. Mock store                  — offline fallback for common demo claims

Results cached with 1-hour TTL to prevent redundant network calls.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import re
import urllib.parse
from typing import List

from cachetools import TTLCache

from hallucination_guard.trust_score import EvidenceItem
from hallucination_guard.config import settings
from hallucination_guard.security.key_manager import key_manager

logger = logging.getLogger(__name__)

# Fix #7: shared TTL cache (1 hour) with bounded size (512 entries) prevents unbounded growth
_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)

_HEADERS = {
    "User-Agent": "HallucinationGuard/6.0 (research; contact: amada@research.org)"
}

# ---------------------------------------------------------------------------
# Curated mock evidence store for common demo queries (fallback when offline)
# ---------------------------------------------------------------------------
_MOCK_STORE: dict = {
    "donald trump": [
        {
            "text": "Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021. Born into a wealthy New York City family, Trump graduated from the University of Pennsylvania in 1968 with a bachelor's degree in economics.",
            "url": "https://en.wikipedia.org/wiki/Donald_Trump",
            "relevance": 0.93,
        }
    ],
    "joe biden": [
        {
            "text": "Joseph Robinette Biden Jr. (born November 20, 1942) is an American politician who was the 46th president of the United States from 2021 to 2025. A member of the Democratic Party, he represented Delaware in the United States Senate from 1973 to 2009 and also served as the 47th vice president under President Barack Obama from 2009 to 2017.",
            "url": "https://en.wikipedia.org/wiki/Joe_Biden",
            "relevance": 0.92,
        }
    ],
    "barack obama": [
        {
            "text": "Barack Hussein Obama II (born August 4, 1961) is an American politician who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, he was the first African American president. Obama previously served as a U.S. senator representing Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.",
            "url": "https://en.wikipedia.org/wiki/Barack_Obama",
            "relevance": 0.93,
        }
    ],
    "narendra modi": [
        {
            "text": "Narendra Damodardas Modi (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation.",
            "url": "https://en.wikipedia.org/wiki/Narendra_Modi",
            "relevance": 0.92,
        }
    ],
    "elon musk": [
        {
            "text": "Elon Reeve Musk (born June 28, 1971) is a businessman known for his key roles in the space company SpaceX and the automotive company Tesla, Inc. He is also the owner and CTO of X (formerly Twitter) and founder of xAI. He is one of the world's wealthiest people.",
            "url": "https://en.wikipedia.org/wiki/Elon_Musk",
            "relevance": 0.92,
        }
    ],
    "climate change": [
        {
            "text": "Climate change refers to long-term shifts in temperatures and weather patterns. Such shifts can be natural, due to changes in the sun's activity or large volcanic eruptions. But since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas.",
            "url": "https://www.un.org/en/climatechange/what-is-climate-change",
            "relevance": 0.91,
        }
    ],
    "covid": [
        {
            "text": "COVID-19 is a disease caused by a coronavirus called SARS-CoV-2. WHO first learned of this new virus on 31 December 2019, following a report of a cluster of cases of so-called 'viral pneumonia' in Wuhan, People's Republic of China. It was declared a pandemic on 11 March 2020.",
            "url": "https://www.who.int/health-topics/coronavirus",
            "relevance": 0.90,
        }
    ],
    "artificial intelligence": [
        {
            "text": "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
            "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "relevance": 0.89,
        }
    ],
}


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


def _mock_lookup(query: str) -> List[EvidenceItem]:
    """Return mock evidence for known queries (offline fallback)."""
    q_lower = query.lower()
    items: List[EvidenceItem] = []
    for keyword, entries in _MOCK_STORE.items():
        if keyword in q_lower:
            for entry in entries:
                items.append(EvidenceItem(
                    text=entry["text"],
                    source="mock_store",
                    relevance=entry["relevance"],
                    timestamp_retrieved=time.time(),
                    url=entry["url"],
                ))
    return items


async def _wikipedia_search_async(query: str, max_results: int = 5) -> List[EvidenceItem]:
    """Async Wikipedia search using httpx connection pool (Fix #7)."""
    try:
        import httpx
        import json

        encoded_query = urllib.parse.quote(query)
        search_url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded_query}"
            f"&format=json&srlimit={max_results}"
        )

        async with httpx.AsyncClient(headers=_HEADERS, timeout=5.0) as client:
            resp1 = await client.get(search_url)
            resp1.raise_for_status()
            data = resp1.json()

            titles = [item["title"] for item in data.get("query", {}).get("search", [])]
            if not titles:
                return []

            titles_param = urllib.parse.quote("|".join(titles[:2]))
            extract_url = (
                f"https://en.wikipedia.org/w/api.php"
                f"?action=query&prop=extracts&exintro&explaintext&titles={titles_param}"
                f"&format=json&exsentences=3&exlimit=2"
            )
            resp2 = await client.get(extract_url)
            resp2.raise_for_status()
            edata = resp2.json()

            items: List[EvidenceItem] = []
            pages = edata.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                extract = page.get("extract", "").strip()
                title = page.get("title", "")
                if not extract:
                    continue
                # Take first 3 sentences
                sentences = re.split(r"(?<=[.!?])\s+", extract)
                snippet = " ".join(sentences[:3])[:400]
                items.append(EvidenceItem(
                    text=snippet,
                    source=f"wikipedia:{title}",
                    relevance=0.75,
                    timestamp_retrieved=time.time(),
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                ))
            return items

    except Exception as e:
        logger.debug("Wikipedia async search failed: %s", e)
        return []


def _wikipedia_search(query: str, max_results: int = 5) -> List[EvidenceItem]:
    """Synchronous Wikipedia search using urllib (fallback when httpx unavailable)."""
    try:
        import urllib.request
        import json

        encoded_query = urllib.parse.quote(query)
        search_url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={encoded_query}"
            f"&format=json&srlimit={max_results}"
        )

        req = urllib.request.Request(search_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        titles = [item["title"] for item in data.get("query", {}).get("search", [])]
        if not titles:
            return []

        titles_param = urllib.parse.quote("|".join(titles[:2]))
        extract_url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro&explaintext&titles={titles_param}"
            f"&format=json&exsentences=3&exlimit=2"
        )
        req2 = urllib.request.Request(extract_url, headers=_HEADERS)
        with urllib.request.urlopen(req2, timeout=5) as resp2:
            edata = json.loads(resp2.read().decode())

        items: List[EvidenceItem] = []
        pages = edata.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            extract = page.get("extract", "").strip()
            title = page.get("title", "")
            if not extract:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", extract)
            snippet = " ".join(sentences[:3])[:400]
            items.append(EvidenceItem(
                text=snippet,
                source=f"wikipedia:{title}",
                relevance=0.75,
                timestamp_retrieved=time.time(),
                url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            ))
        return items

    except Exception as e:
        logger.debug("Wikipedia sync search failed: %s", e)
        return []


async def _tavily_search(query: str, max_results: int = 5) -> List[EvidenceItem]:
    """Tavily real-time web search — returns clean text, covers breaking news."""
    api_key = key_manager.get_tavily_key()
    if not api_key:
        return []
    try:
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=api_key)
        response = await client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )
        items: List[EvidenceItem] = []
        for r in response.get("results", []):
            content = r.get("content", "").strip()
            if not content:
                continue
            items.append(EvidenceItem(
                text=content[:500],
                source=f"tavily:{r.get('url', '')}",
                relevance=float(r.get("score", 0.8)),
                timestamp_retrieved=time.time(),
                url=r.get("url", ""),
            ))
        return items
    except Exception as e:
        logger.debug("Tavily search failed: %s", e)
        return []


async def _fact_check_search(query: str, max_results: int = 5) -> List[EvidenceItem]:
    """Google Fact Check Tools API — searches Snopes, PolitiFact, FactCheck.org etc."""
    api_key = key_manager.get_fact_check_key()
    if not api_key:
        return []
    try:
        import httpx
        params = {
            "query": query,
            "key": api_key,
            "pageSize": max_results,
            "languageCode": "en",
        }
        async with httpx.AsyncClient(headers=_HEADERS, timeout=5.0) as client:
            resp = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        items: List[EvidenceItem] = []
        for claim in data.get("claims", []):
            claim_text = claim.get("text", "").strip()
            for review in claim.get("claimReview", []):
                publisher = review.get("publisher", {}).get("name", "unknown")
                rating = review.get("textualRating", "")
                snippet = f'Claim: "{claim_text}" — Rated "{rating}" by {publisher}.'
                items.append(EvidenceItem(
                    text=snippet,
                    source=f"factcheck:{publisher}",
                    relevance=0.95,
                    timestamp_retrieved=time.time(),
                    url=review.get("url", ""),
                ))
        return items
    except Exception as e:
        logger.debug("Fact Check API search failed: %s", e)
        return []


def _google_search(query: str, max_results: int = 5) -> List[EvidenceItem]:
    """Google Custom Search API (requires GOOGLE_SEARCH_API_KEY)."""
    api_key = key_manager.get_google_search_key()
    engine_id = settings.google_search_engine_id
    if not api_key or not engine_id:
        return []
    try:
        from googleapiclient.discovery import build as _build
        service = _build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(
            q=query,
            cx=engine_id,
            num=min(max_results, 10),
        ).execute()
        items: List[EvidenceItem] = []
        for item in res.get("items", []):
            snippet = item.get("snippet", "").strip()
            if not snippet:
                continue
            items.append(EvidenceItem(
                text=snippet,
                source=f"google:{item.get('displayLink', '')}",
                relevance=0.70,
                timestamp_retrieved=time.time(),
                url=item.get("link", ""),
            ))
        return items
    except Exception:
        return []


async def retrieve_evidence(query: str, top_k: int = 3) -> List[EvidenceItem]:
    """Retrieve evidence from Google -> Wikipedia -> mock store, with TTL cache.

    Fix #7: uses httpx async connection pool for Wikipedia to avoid spawning
    a new TCP connection per request.

    Parameters
    ----------
    query  : Search query string.
    top_k  : Maximum evidence items to return.

    Returns
    -------
    List of EvidenceItem (deduplicated, sorted by relevance descending).
    """
    ck = _cache_key(query)
    if ck in _cache:
        return list(_cache[ck])[:top_k]

    loop = asyncio.get_event_loop()
    items: List[EvidenceItem] = []

    # 1. Google Fact Check Tools API (highest signal — directly indexes fact-checkers)
    if key_manager.has_fact_check():
        try:
            fc_items = await _fact_check_search(query, max_results=top_k)
            items.extend(fc_items)
        except Exception as e:
            logger.debug("Fact Check API failed: %s", e)

    # 2. Tavily real-time web search (breaking news, social media claims)
    if len(items) < top_k and key_manager.has_tavily():
        try:
            tavily_items = await _tavily_search(query, max_results=top_k)
            items.extend(tavily_items)
        except Exception as e:
            logger.debug("Tavily search failed: %s", e)

    # 3. Google Custom Search API
    if len(items) < top_k and key_manager.has_google_search():
        try:
            google_items = await loop.run_in_executor(None, _google_search, query, top_k)
            items.extend(google_items)
        except Exception as e:
            logger.debug("Google search failed: %s", e)

    # 4. Wikipedia (background context, no key needed)
    if len(items) < top_k:
        try:
            wiki_items = await _wikipedia_search_async(query, max_results=top_k)
            items.extend(wiki_items)
        except Exception:
            try:
                wiki_items = await loop.run_in_executor(None, _wikipedia_search, query, top_k)
                items.extend(wiki_items)
            except Exception:
                pass

    # 5. Mock store fallback (offline)
    if not items:
        items = _mock_lookup(query)

    # Deduplicate by text, sort by relevance descending
    seen: set = set()
    unique: List[EvidenceItem] = []
    for ev in sorted(items, key=lambda e: e.relevance, reverse=True):
        if ev.text not in seen:
            seen.add(ev.text)
            unique.append(ev)

    result = unique[:top_k]
    _cache[ck] = result
    return result
