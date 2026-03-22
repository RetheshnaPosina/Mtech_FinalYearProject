"""MiniLM-L6 sentence similarity — evidence relevance scoring."""
from __future__ import annotations

import numpy as np
from typing import List


class SentenceModel:
    """Wraps sentence-transformers for dense embedding and cosine similarity."""

    def __init__(self, model_id: str, cache_dir: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, cache_folder=cache_dir)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of strings into L2-normalised embeddings.

        Returns an array of shape (len(texts), embedding_dim).
        """
        return self.model.encode(texts, normalize_embeddings=True)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two sentences in [0, 1] (normalised embeddings)."""
        embs = self.encode([text_a, text_b])
        return float(np.dot(embs[0], embs[1]))

    def top_k_similar(
        self,
        query: str,
        candidates: List[str],
        k: int = 3,
    ) -> List[tuple[str, float]]:
        """Return the k most similar candidates to query, as (text, score) pairs."""
        all_texts = [query] + candidates
        embs = self.encode(all_texts)
        query_emb = embs[0]
        scored = [
            (candidates[i], float(np.dot(query_emb, embs[i + 1])))
            for i in range(len(candidates))
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
