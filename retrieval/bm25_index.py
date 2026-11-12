"""
retrieval/bm25_index.py
Sparse keyword search using BM25 (Okapi BM25 via rank-bm25).

The index is built in-memory from the chunks already stored in the vector
store.  It is rebuilt on each process start (cheap — milliseconds for
thousands of chunks) so there is no separate persistence concern.

Interface mirrors BaseVectorStore.query() so the hybrid layer can treat
both retrievers identically.
"""
from __future__ import annotations

import string
from typing import List

from ingestion.chunker import Chunk
from store.vector_store import BaseVectorStore, RetrievedChunk, _dict_to_chunk

BM25_SATURATION_K = 1.5


class BM25Index:
    """
    In-memory BM25 index over a fixed corpus of chunks.

    Build once after ingestion; rebuild whenever new docs are added.
    """

    def __init__(self, saturation_k: float = BM25_SATURATION_K):
        self._chunks: List[Chunk] = []
        self._bm25 = None
        self.saturation_k = saturation_k

    def build(self, chunks: List[Chunk]) -> None:
        """
        (Re)build the BM25 index over the given chunks.
        Call this after every ingestion run.
        """
        if not chunks:
            self._chunks = []
            self._bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Run: pip install rank-bm25")

        self._chunks = chunks
        tokenised_corpus = [self._tokenise(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenised_corpus)
        # Ensure terms present in >= 50% of a small corpus (where Okapi IDF is <= 0)
        # still receive a positive floor weight rather than 0.0.
        floor_idf = max(float(getattr(self._bm25, "epsilon", 0.25)) * max(float(getattr(self._bm25, "average_idf", 0.0)), 0.5), 0.05)
        for term, val in list(self._bm25.idf.items()):
            if val <= 0.0:
                self._bm25.idf[term] = floor_idf

    def build_from_vector_store(self, vector_store: BaseVectorStore) -> None:
        """
        Pull all chunks out of a vector store and build the index.
        Works across both ChromaDB and Weaviate via get_all_chunks().
        """
        if hasattr(vector_store, "get_all_chunks"):
            chunks = vector_store.get_all_chunks()
        elif hasattr(vector_store, "_collection"):
            raw = vector_store._collection.get(include=["documents", "metadatas"])
            chunks = [
                _dict_to_chunk({**(meta or {}), "text": doc})
                for doc, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
            ]
        else:
            chunks = []
        self.build(chunks)

    def query(
        self,
        query_text: str,
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        """
        Return the top_k chunks ranked by BM25 score.
        Scores are calibrated to [0, 1) via saturation normalisation s / (s + k)
        so weak lexical matches are not artificially inflated to 1.0.
        """
        if self._bm25 is None or not self._chunks or top_k <= 0:
            return []

        tokenised_query = self._tokenise(query_text)
        if not tokenised_query:
            return []

        raw_scores = self._bm25.get_scores(tokenised_query)
        k = max(1e-6, float(self.saturation_k))
        norm_scores = [
            float(s) / (float(s) + k) if float(s) > 0.0 else 0.0
            for s in raw_scores
        ]

        ranked = sorted(
            enumerate(norm_scores), key=lambda x: (-x[1], x[0])
        )

        results: List[RetrievedChunk] = []
        for idx, score in ranked[:top_k]:
            if score < score_threshold or score <= 0.0:
                break
            chunk = self._chunks[idx]
            results.append(RetrievedChunk(chunk.to_dict(), score=score))

        return results

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """
        Lowercase, strip punctuation, split on whitespace.
        Deliberately simple — fast and sufficient for BM25 token matching.
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()
