"""
agent/tools.py
Tool definitions and adapters for local document search.
"""
from __future__ import annotations

import string
from typing import Any, Dict, List

from retrieval.retriever import RetrievalResult
from store.vector_store import RetrievedChunk

LOCAL_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_local_documents",
        "description": (
            "Search the user's own ingested documents (PDFs, Word docs, LaTeX, notes, etc). "
            "Use this first for anything that plausibly lives in the user's private corpus. "
            "Returns ranked chunks with citation IDs and similarity scores; a low confidence "
            "score means the corpus likely doesn't cover this question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to match against local document chunks.",
                }
            },
            "required": ["query"],
        },
    },
}


def _compute_calibrated_confidence(query: str, chunks: List[RetrievedChunk]) -> float:
    """
    Compute a calibrated retrieval sufficiency confidence score in [0.0, 1.0]
    rather than relying solely on an uncalibrated cross-encoder sigmoid output.
    """
    if not chunks:
        return 0.0

    top = chunks[0]
    top_score = float(top.score)
    meta = getattr(top, "metadata", None) or {}
    pre_rerank_score = meta.get("retrieval_score")

    trans = str.maketrans("", "", string.punctuation)
    q_tokens = {w for w in query.lower().translate(trans).split() if len(w) >= 3}
    if q_tokens:
        combined_text = " ".join(c.text.lower() for c in chunks[:3]).translate(trans)
        doc_tokens = set(combined_text.split())
        lexical_overlap = len(q_tokens & doc_tokens) / len(q_tokens)
    else:
        lexical_overlap = 1.0

    if pre_rerank_score is not None:
        # Blend cross-encoder probability, bi-encoder/hybrid similarity, and lexical coverage
        return round(
            0.50 * top_score + 0.35 * float(pre_rerank_score) + 0.15 * lexical_overlap,
            4,
        )

    # When reranker was not active, combine similarity score with lexical coverage check
    if lexical_overlap == 0.0 and q_tokens:
        return round(0.70 * top_score, 4)
    return round(0.85 * top_score + 0.15 * lexical_overlap, 4)


def run_local_search(
    retriever: Any,
    query: str,
    score_threshold: float = 0.75,
) -> Dict[str, Any]:
    """
    Executes local document search via retriever, returns structured results with
    top_score, calibrated confidence_score, below_confidence_threshold metadata,
    and raw RetrievalResult.
    """
    res: RetrievalResult = retriever.retrieve(query)
    chunks_data = [
        {
            "citation_id": chunk.citation_id,
            "source": chunk.source,
            "score": chunk.score,
            "text": chunk.text,
        }
        for chunk in res.chunks
    ]

    top_score = float(res.chunks[0].score) if res.chunks else 0.0
    confidence_score = _compute_calibrated_confidence(query, res.chunks)
    below_threshold = confidence_score < score_threshold

    return {
        "chunks": chunks_data,
        "top_score": top_score,
        "confidence_score": confidence_score,
        "below_confidence_threshold": below_threshold,
        "_raw_result": res,
    }
