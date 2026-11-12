"""
ingestion/pipeline.py
Orchestrates the full ingestion flow:
    raw documents → chunks → embeddings → vector store → BM25 index

Re-ingestion is source-aware and replaces stale chunks for changed documents.
Calling build_ingestion_pipeline("config/phase2.yaml") wires up the
HybridRetriever automatically; "config/phase1.yaml" still returns the
plain Phase 1 Retriever — no caller changes needed.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from config.loader import PipelineConfig, load_config
from ingestion.chunker import Chunk, TokenChunker
from ingestion.embedder import BaseEmbedder, build_embedder
from store.vector_store import BaseVectorStore, build_vector_store

logger = logging.getLogger(__name__)


class IngestionStats:
    def __init__(
        self,
        num_documents: int,
        num_chunks: int,
        elapsed_seconds: float,
        skipped: Optional[List[dict]] = None,
    ):
        self.num_documents   = num_documents
        self.num_chunks      = num_chunks
        self.elapsed_seconds = elapsed_seconds
        self.skipped         = skipped or []

    def __str__(self) -> str:
        msg = (
            f"Ingested {self.num_documents} document(s) → "
            f"{self.num_chunks} chunks in {self.elapsed_seconds:.1f}s"
        )
        if self.skipped:
            msg += f" ({len(self.skipped)} skipped)"
        return msg


class IngestionPipeline:
    """
    Wires together:
      TokenChunker → BaseEmbedder → BaseVectorStore (→ BM25Index if Phase 2+)

    All components are injected so they can be mocked in tests.
    """

    def __init__(
        self,
        chunker:      TokenChunker,
        embedder:     BaseEmbedder,
        vector_store: BaseVectorStore,
        cfg:          PipelineConfig,
        bm25_index=None,
    ):
        self.chunker      = chunker
        self.embedder     = embedder
        self.vector_store = vector_store
        self.cfg          = cfg
        self.bm25_index   = bm25_index

    def ingest_documents(self, documents: List[dict]) -> IngestionStats:
        """
        Ingest a list of document dicts.

        Each dict must contain:
          - "text"     (str)  – full document content
          - "source"   (str)  – filename, URL, or human label
          - "metadata" (dict, optional)
        """
        t0 = time.time()
        non_empty_docs = [d for d in documents if d.get("text", "").strip()]

        chunks: List[Chunk] = self.chunker.chunk_documents(non_empty_docs)
        if not chunks:
            return IngestionStats(0, 0, time.time() - t0)

        embeddings = self.embedder.embed_chunks(chunks)
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedder returned {len(embeddings)} vector(s) for {len(chunks)} chunk(s)."
            )

        expected_dim = getattr(self.embedder, "dimensions", None)
        if not isinstance(expected_dim, int) or expected_dim <= 0:
            expected_dim = self.cfg.embedding.dimensions if self.cfg.embedding.provider != "local" else None

        first_dim = len(embeddings[0])
        if first_dim <= 0:
            raise ValueError("Embedder returned empty embedding vectors.")
        if expected_dim and len(embeddings[0]) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(embeddings[0])}, expected {expected_dim}."
            )
        for i, vec in enumerate(embeddings):
            if len(vec) != first_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension at index {i}: got {len(vec)}, expected {first_dim}."
                )

        provider_name = getattr(self.embedder, "provider", None)
        if not isinstance(provider_name, str) or provider_name == "unknown":
            provider_name = self.cfg.embedding.provider
        model_name = getattr(self.embedder, "model_name", None)
        if not isinstance(model_name, str) or model_name == "unknown":
            model_name = self.cfg.embedding.model

        for c in chunks:
            c.metadata["embedding_provider"] = provider_name
            c.metadata["embedding_model"] = model_name
            c.metadata["embedding_dimensions"] = first_dim

        # Remove stale chunks for any source being re-ingested before upserting new chunks
        unique_sources = list(dict.fromkeys(d["source"] for d in non_empty_docs if d.get("source")))
        if hasattr(self.vector_store, "delete_by_source"):
            for src in unique_sources:
                self.vector_store.delete_by_source(src)

        self.vector_store.upsert(chunks, embeddings)

        if self.bm25_index is not None:
            self.bm25_index.build_from_vector_store(self.vector_store)

        return IngestionStats(
            num_documents=len(non_empty_docs),
            num_chunks=len(chunks),
            elapsed_seconds=time.time() - t0,
        )

    def _all_chunks(self) -> List[Chunk]:
        """Pull every stored chunk from the vector store for BM25 index rebuilding."""
        if hasattr(self.vector_store, "get_all_chunks"):
            return self.vector_store.get_all_chunks()
        from store.vector_store import _dict_to_chunk
        raw = self.vector_store._collection.get(include=["documents", "metadatas"])
        return [
            _dict_to_chunk({**(meta or {}), "text": doc})
            for doc, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
        ]

    def _parser_settings(self):
        parsers_cfg = getattr(getattr(self.cfg, "ingestion", None), "parsers", None)
        keep_math = getattr(getattr(parsers_cfg, "latex", None), "keep_math", False) if parsers_cfg else False
        ocr_fallback = getattr(getattr(parsers_cfg, "pdf", None), "ocr_fallback", False) if parsers_cfg else False
        enabled_formats = getattr(parsers_cfg, "enabled_formats", None) if parsers_cfg else None
        return keep_math, ocr_fallback, enabled_formats

    def ingest_file(self, path: str, metadata: Optional[dict] = None) -> IngestionStats:
        from ingestion.parsers.registry import get_parser_for
        p = Path(path)
        keep_math, ocr_fallback, enabled_formats = self._parser_settings()
        parser = get_parser_for(
            p,
            keep_math=keep_math,
            ocr_fallback=ocr_fallback,
            enabled_formats=enabled_formats,
        )
        parsed = parser.parse(p)
        if not parsed.text.strip():
            return IngestionStats(
                num_documents=0,
                num_chunks=0,
                elapsed_seconds=0.0,
                skipped=[{"file": str(p), "reason": "No extractable text found"}],
            )
        merged_meta = {**parsed.metadata, **(metadata or {})}
        return self.ingest_documents([{
            "text": parsed.text, "source": parsed.source, "metadata": merged_meta,
        }])

    def ingest_directory(self, directory: str, glob: str = "**/*") -> IngestionStats:
        from ingestion.parsers.base import UnsupportedFormatError
        from ingestion.parsers.registry import get_parser_for
        d = Path(directory)
        docs = []
        skipped = []
        keep_math, ocr_fallback, enabled_formats = self._parser_settings()

        for p in sorted(d.glob(glob)):
            if not p.is_file():
                continue
            try:
                parser = get_parser_for(
                    p,
                    keep_math=keep_math,
                    ocr_fallback=ocr_fallback,
                    enabled_formats=enabled_formats,
                )
                parsed = parser.parse(p)
                if not parsed.text.strip():
                    skipped.append({"file": str(p), "reason": "No extractable text found"})
                    continue
                docs.append({
                    "text": parsed.text,
                    "source": str(p.relative_to(d)),
                    "metadata": {**parsed.metadata, "filepath": str(p)},
                })
            except UnsupportedFormatError as e:
                skipped.append({"file": str(p), "reason": str(e)})
            except Exception as e:
                skipped.append({"file": str(p), "reason": str(e)})

        stats = self.ingest_documents(docs)
        stats.skipped = skipped
        return stats


class RetrieverBundle:
    """Convenience holder returned by build_ingestion_pipeline."""
    def __init__(self, retriever, rag):
        self.retriever = retriever
        self.rag       = rag


def build_ingestion_pipeline(
    config_path: str = "config/phase1.yaml",
    cfg: Optional[PipelineConfig] = None,
) -> tuple[IngestionPipeline, RetrieverBundle]:
    """
    Build and return both ingestion and retrieval components from one config.

    Phase 1 YAML  →  plain vector Retriever + RAGPipeline
    Phase 2+ YAML →  HybridRetriever + HybridRAGPipeline  (auto-detected)
    """
    if cfg is None:
        cfg = load_config(config_path)
    chunker      = TokenChunker(cfg.chunking)
    embedder     = build_embedder(cfg.embedding)
    vector_store = build_vector_store(cfg.vector_store)

    # Validate existing collection embedding compatibility against active embedder
    if hasattr(vector_store, "get_embedding_metadata") and vector_store.count() > 0:
        meta = vector_store.get_embedding_metadata()
        stored_dim = meta.get("embedding_dimensions")
        embedder_dim = getattr(embedder, "dimensions", None)
        if stored_dim and embedder_dim and int(stored_dim) != int(embedder_dim):
            vector_store.close()
            raise ValueError(
                f"Existing collection embedding dimension ({stored_dim}) is incompatible "
                f"with configured embedder '{embedder.model_name}' dimension ({embedder_dim})."
            )

    if cfg.retrieval.hybrid.enabled:
        from retrieval.bm25_index import BM25Index
        from retrieval.hybrid_retriever import HybridRAGPipeline, HybridRetriever

        bm25_index = BM25Index()
        try:
            if vector_store.count() > 0:
                bm25_index.build_from_vector_store(vector_store)
        except Exception as e:
            logger.warning(f"Failed to build initial BM25 index from vector store: {e}")

        pipeline   = IngestionPipeline(
            chunker, embedder, vector_store, cfg, bm25_index=bm25_index
        )
        retriever = HybridRetriever(embedder, vector_store, bm25_index, cfg)
        rag       = HybridRAGPipeline(retriever, cfg)
    else:
        from retrieval.retriever import RAGPipeline, Retriever

        pipeline  = IngestionPipeline(chunker, embedder, vector_store, cfg)
        retriever = Retriever(embedder, vector_store, cfg)
        rag       = RAGPipeline(retriever, cfg)

    return pipeline, RetrieverBundle(retriever=retriever, rag=rag)
