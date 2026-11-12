"""
retrieval/retriever.py
Pulls the top-k most relevant chunks for a query and packages them with
inline citations ready to inject into the LLM prompt.

Phase 2 extends this class with hybrid BM25 + vector search and
a cross-encoder re-ranker — the interface stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.loader import PipelineConfig, create_llm_client, get_default_model
from ingestion.embedder import BaseEmbedder
from store.vector_store import BaseVectorStore, RetrievedChunk


def disambiguate_citation_ids(chunks: List[RetrievedChunk]) -> None:
    """
    If two chunks from different documents happen to share the same 8-char prefix
    and chunk_index, expand their citation_id prefix length until unique.
    """
    by_cid: Dict[str, List[RetrievedChunk]] = {}
    for c in chunks:
        by_cid.setdefault(c.citation_id, []).append(c)

    for cid, group in by_cid.items():
        doc_ids = {c.doc_id for c in group}
        if len(doc_ids) <= 1:
            continue
        for prefix_len in (12, 16, 24, 32, 64):
            candidate_ids = {f"{c.doc_id[:prefix_len]}:{c.chunk_index}" for c in group}
            if len(candidate_ids) == len(doc_ids):
                for c in group:
                    c.citation_id = f"{c.doc_id[:prefix_len]}:{c.chunk_index}"
                break


@dataclass
class RetrievalResult:
    """Everything the generation step needs."""
    query: str
    chunks: List[RetrievedChunk]
    context_block: str
    citations: List[str]


class Retriever:
    """
    Phase 1: pure vector (semantic) retrieval.
    Phase 2: subclassed / extended with BM25 + re-ranking.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        cfg: PipelineConfig,
    ):
        self.embedder     = embedder
        self.vector_store = vector_store
        self.cfg          = cfg

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        Embed the query → search the vector store → return ranked chunks
        wrapped with a formatted context block and citation list.
        """
        retrieval_cfg = self.cfg.retrieval
        k = top_k if top_k is not None else retrieval_cfg.top_k

        query_embedding = self.embedder.embed_query(query)

        chunks = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            score_threshold=retrieval_cfg.score_threshold,
        )

        disambiguate_citation_ids(chunks)
        context_block, citations = self._format_context(
            chunks, include_metadata=retrieval_cfg.include_metadata
        )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            context_block=context_block,
            citations=citations,
        )

    def build_prompt(self, result: RetrievalResult) -> Tuple[str, str]:
        """
        Fill the versioned prompt templates from config.
        Returns (system_prompt, user_prompt).
        """
        prompts = self.cfg.prompts
        user_prompt = prompts.rag_user.format(
            context=result.context_block,
            question=result.query,
        )
        return prompts.rag_system, user_prompt

    @staticmethod
    def _format_context(
        chunks: List[RetrievedChunk],
        include_metadata: bool = True,
    ) -> Tuple[str, List[str]]:
        parts: List[str] = []
        citations: List[str] = []

        for chunk in chunks:
            citation = f"[{chunk.citation_id}]"
            citations.append(citation)
            if include_metadata:
                extra_items = [
                    f"{k}={v}"
                    for k, v in sorted((chunk.metadata or {}).items())
                    if k not in {
                        "embedding_dimensions",
                        "embedding_provider",
                        "embedding_model",
                        "retrieval_score",
                        "raw_rerank_logit",
                        "rerank_score",
                    }
                ]
                extra_str = f", {', '.join(extra_items)}" if extra_items else ""
                header = f"{citation}  (source: {chunk.source}, score: {chunk.score:.4f}{extra_str})"
            else:
                header = f"{citation}"
            parts.append(f"{header}\n{chunk.text}")

        context_block = "\n\n".join(parts)
        return context_block, citations


class RAGPipeline:
    """
    Thin wrapper that chains Retriever → OpenRouter LLM completion.
    Keeps generation logic separate from retrieval so each can be
    tested and swapped independently.
    """

    def __init__(
        self,
        retriever: Retriever,
        cfg: PipelineConfig,
    ):
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
    ) -> "RAGAnswer":
        """Full RAG cycle: retrieve → build prompt → generate → return answer."""
        result = self.retriever.retrieve(question, top_k=top_k)
        system_prompt, user_prompt = self.retriever.build_prompt(result)

        if self._llm is None:
            raise RuntimeError(
                "LLM client not initialised. Check OPENROUTER_API_KEY (or OPENAI_API_KEY)."
            )

        effective_model = model or get_default_model(self.cfg)

        response = self._llm.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
        )

        answer_text = response.choices[0].message.content or ""

        return RAGAnswer(
            question=question,
            answer=answer_text,
            retrieved_chunks=result.chunks,
            citations=result.citations,
            context_block=result.context_block,
        )


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    citations: List[str]
    context_block: str

    def pretty_print(self) -> None:
        """Print a formatted answer to stdout."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(Panel(self.answer, title="Answer", border_style="green"))
            console.print("\n[bold]Sources used:[/bold]")
            for chunk in self.retrieved_chunks:
                console.print(
                    f"  • [{chunk.citation_id}]  {chunk.source}  "
                    f"(score={chunk.score:.4f})"
                )
        except ImportError:
            print(f"Answer:\n{self.answer}")
            print("\nSources:")
            for chunk in self.retrieved_chunks:
                print(f"  [{chunk.citation_id}] {chunk.source}")
