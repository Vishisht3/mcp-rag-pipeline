"""
retrieval/reranker.py
Cross-encoder re-ranker using sentence-transformers.

A cross-encoder scores (query, passage) pairs jointly, which is far more
accurate than the inner-product approximation used in bi-encoder retrieval.
The trade-off is speed — O(k) inference calls — so we only re-rank a small
candidate set (top_k × candidate_multiplier from Phase 2 config).
"""
from __future__ import annotations

import math
from typing import List

from store.vector_store import RetrievedChunk


class CrossEncoderReranker:
    """
    Wraps a sentence-transformers CrossEncoder.
    Lazy-loaded on first use to avoid import overhead when re-ranking
    is disabled via config.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "Run: pip install sentence-transformers\n"
                    "Or disable re-ranking in config: reranker.enabled = false"
                )
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Score each (query, chunk.text) pair and return the top_k highest-
        scoring chunks. Preserves pre-rerank retrieval_score in metadata.
        """
        if not candidates:
            return []

        self._load()

        pairs = [(query, chunk.text) for chunk in candidates]
        raw_scores: List[float] = self._model.predict(pairs).tolist()

        def sigmoid(x: float) -> float:
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            else:
                z = math.exp(x)
                return z / (1.0 + z)

        for chunk, raw in zip(candidates, raw_scores):
            chunk.metadata["retrieval_score"] = float(chunk.score)
            chunk.metadata["raw_rerank_logit"] = float(raw)
            rerank_prob = sigmoid(float(raw))
            chunk.metadata["rerank_score"] = rerank_prob
            chunk.score = rerank_prob

        reranked = sorted(
            candidates,
            key=lambda c: (-round(c.score, 12), c.citation_id, c.source, c.chunk_index),
        )
        return reranked[:top_k]
