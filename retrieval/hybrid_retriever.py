"""
retrieval/hybrid_retriever.py
Phase 2 retriever. Extends the Phase 1 Retriever with:

  1. Hybrid fusion  — combines BM25 keyword scores with dense vector scores
  2. Cross-encoder re-ranking  — rescores the fused candidate set
  3. Citation enforcement      — validates / retries answers post-generation
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from config.loader import PipelineConfig, create_llm_client, get_default_model
from ingestion.embedder import BaseEmbedder
from retrieval.bm25_index import BM25Index
from retrieval.citation_enforcer import CitationEnforcer
from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import (
    RAGAnswer,
    RetrievalResult,
    Retriever,
    disambiguate_citation_ids,
)
from store.vector_store import BaseVectorStore, RetrievedChunk


def _chunk_identity_key(c: RetrievedChunk) -> Tuple[str, int]:
    """Return a collision-resistant identity key using full doc_id + chunk_index."""
    return (c.doc_id or c.citation_id, int(c.chunk_index))


class HybridRetriever(Retriever):
    """
    Drop-in replacement for Phase 1 Retriever.
    Adds: hybrid fusion · re-ranking · citation enforcement.
    """

    def __init__(
        self,
        embedder:      BaseEmbedder,
        vector_store:  BaseVectorStore,
        bm25_index:    BM25Index,
        cfg:           PipelineConfig,
    ):
        super().__init__(embedder, vector_store, cfg)
        self.bm25_index = bm25_index

        rc = cfg.retrieval
        self.reranker = (
            CrossEncoderReranker(rc.reranker.model)
            if rc.reranker.enabled
            else None
        )
        self.enforcer = CitationEnforcer(cfg.citation_enforcement)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        rc = self.cfg.retrieval
        effective_top_k = top_k if top_k is not None else rc.top_k

        n_candidates = effective_top_k * rc.hybrid.candidate_multiplier

        query_embedding = self.embedder.embed_query(query)
        vector_chunks   = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=n_candidates,
            score_threshold=rc.score_threshold,
        )

        bm25_chunks = self.bm25_index.query(query, top_k=n_candidates)

        fused = self._fuse(
            vector_chunks,
            bm25_chunks,
            alpha=rc.hybrid.vector_weight,
            bm25_weight=rc.hybrid.bm25_weight,
        )

        if self.reranker is not None:
            rerank_k = top_k if top_k is not None else rc.reranker.top_k_after_rerank
            final_chunks = self.reranker.rerank(
                query=query,
                candidates=fused,
                top_k=rerank_k,
            )
        else:
            final_chunks = fused[:effective_top_k]

        disambiguate_citation_ids(final_chunks)
        context_block, citations = self._format_context(
            final_chunks, include_metadata=rc.include_metadata
        )

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            context_block=context_block,
            citations=citations,
        )

    def build_retry_prompt(
        self,
        result: RetrievalResult,
        previous_answer: str,
    ) -> Tuple[str, str]:
        prompts = self.cfg.prompts
        retry_template = prompts.rag_retry or prompts.rag_user

        user_prompt = retry_template.format(
            context=result.context_block,
            question=result.query,
            previous_answer=previous_answer,
        )
        return prompts.rag_system, user_prompt

    @staticmethod
    def _fuse(
        vector_chunks: List[RetrievedChunk],
        bm25_chunks:   List[RetrievedChunk],
        alpha: float = 0.6,
        bm25_weight: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        if bm25_weight is None:
            w_vec = float(alpha)
            w_bm25 = 1.0 - w_vec
        else:
            total = float(alpha) + float(bm25_weight)
            if total <= 0.0:
                raise ValueError("vector_weight + bm25_weight must be > 0.0")
            w_vec = float(alpha) / total
            w_bm25 = float(bm25_weight) / total

        vec_map: Dict[Tuple[str, int], Tuple[RetrievedChunk, float]] = {
            _chunk_identity_key(c): (c, c.score) for c in vector_chunks
        }
        bm25_map: Dict[Tuple[str, int], Tuple[RetrievedChunk, float]] = {
            _chunk_identity_key(c): (c, c.score) for c in bm25_chunks
        }

        # Preserve deterministic insertion order across both candidate lists
        all_keys = list(dict.fromkeys(list(vec_map.keys()) + list(bm25_map.keys())))

        fused: List[Tuple[float, RetrievedChunk]] = []
        for key in all_keys:
            if key in vec_map:
                chunk, vec_score = vec_map[key]
            else:
                chunk, _ = bm25_map[key]
                vec_score = 0.0

            bm25_score = bm25_map[key][1] if key in bm25_map else 0.0
            score = w_vec * vec_score + w_bm25 * bm25_score
            chunk.score = score
            fused.append((score, chunk))

        disambiguate_citation_ids([c for _, c in fused])
        fused.sort(
            key=lambda x: (-round(x[0], 12), x[1].citation_id, x[1].source, x[1].chunk_index)
        )
        return [chunk for _, chunk in fused]


class HybridRAGPipeline:
    """
    Phase 2 full pipeline:
        HybridRetriever → OpenRouter LLM generation → CitationEnforcer (with retry)
    """

    def __init__(self, retriever: HybridRetriever, cfg: PipelineConfig):
        self.retriever = retriever
        self.cfg       = cfg
        self._llm      = self._init_llm()

    def _init_llm(self):
        return create_llm_client(self.cfg)

    def answer(
        self,
        question: str,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGAnswer:
        if self._llm is None:
            raise RuntimeError(
                "LLM client not initialised. Check OPENROUTER_API_KEY (or OPENAI_API_KEY)."
            )

        effective_model = model or get_default_model(self.cfg)

        result = self.retriever.retrieve(question, top_k=top_k)
        system, user = self.retriever.build_prompt(result)

        def _call_llm(system_p: str, user_p: str) -> str:
            resp = self._llm.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": system_p},
                    {"role": "user",   "content": user_p},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""

        answer_text = _call_llm(system, user)

        def _retry_fn(prev_answer: str) -> str:
            retry_system, retry_user = self.retriever.build_retry_prompt(
                result, prev_answer
            )
            return _call_llm(retry_system, retry_user)

        check = self.retriever.enforcer.enforce(
            answer=answer_text,
            retrieved_chunks=result.chunks,
            retry_fn=_retry_fn,
        )

        return RAGAnswer(
            question=question,
            answer=check.answer,
            retrieved_chunks=result.chunks,
            citations=result.citations,
            context_block=result.context_block,
        )
