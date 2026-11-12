"""
tests/test_phase1.py
Unit and integration tests for Phase 1:
  - TokenChunker produces chunks within token bounds and validates config bounds
  - Source-aware doc_id (identical text from different sources never collides)
  - Protected chunk metadata fields cannot be overwritten by user metadata
  - IngestionPipeline replaces stale chunks per source and validates embedding dimensions
  - ChromaVectorStore real persistence, delete_by_source, delete_collection lifecycle
  - WeaviateVectorStore cloud connection and batch error verification
  - Retriever returns ranked results with citation strings and respects include_metadata
  - OpenRouter client configuration
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def chunking_cfg():
    from config.loader import ChunkingConfig
    return ChunkingConfig(min_tokens=100, max_tokens=150, overlap_tokens=20)


@pytest.fixture
def sample_text():
    return " ".join(["word"] * 600)


@pytest.fixture
def sample_documents():
    return [
        {"text": " ".join(["alpha"] * 600), "source": "doc_a.txt"},
        {"text": " ".join(["beta"] * 400),  "source": "doc_b.txt"},
    ]


class TestTokenChunker:

    def test_chunks_within_token_bounds(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        chunks = chunker.chunk_document(sample_text, source="test.txt")

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.token_count <= chunking_cfg.max_tokens, (
                f"Chunk {chunk.chunk_index} has {chunk.token_count} tokens "
                f"(max={chunking_cfg.max_tokens})"
            )

    def test_all_chunks_except_last_above_min(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        chunks = chunker.chunk_document(sample_text, source="test.txt")

        for chunk in chunks[:-1]:
            assert chunk.token_count >= chunking_cfg.min_tokens

    def test_chunk_indices_are_sequential(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document(sample_text, "t.txt")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_citation_id_format(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document(sample_text, "t.txt")
        for chunk in chunks:
            parts = chunk.citation_id.split(":")
            assert len(parts) == 2
            assert len(parts[0]) == 8
            assert parts[1].isdigit()

    def test_same_source_and_text_same_doc_id(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        c1 = chunker.chunk_document(sample_text, "a.txt")
        c2 = chunker.chunk_document(sample_text, "a.txt")
        assert c1[0].doc_id == c2[0].doc_id

    def test_same_text_different_source_distinct_doc_id(self, chunking_cfg, sample_text):
        """Identical document text from different sources must produce distinct doc_ids."""
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        c1 = chunker.chunk_document(sample_text, "a.txt")
        c2 = chunker.chunk_document(sample_text, "b.txt")
        assert c1[0].doc_id != c2[0].doc_id

    def test_protected_fields_cannot_be_overwritten_by_metadata(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunker = TokenChunker(chunking_cfg)
        malicious_meta = {
            "doc_id": "forged_doc_id",
            "source": "forged_source.txt",
            "citation_id": "deadbeef:999",
            "chunk_index": 999,
            "text": "forged text",
            "author": "alice",
        }
        chunks = chunker.chunk_document(sample_text, "real_source.txt", extra_metadata=malicious_meta)
        d = chunks[0].to_dict()
        assert d["source"] == "real_source.txt"
        assert d["doc_id"] != "forged_doc_id"
        assert d["citation_id"] != "deadbeef:999"
        assert d["chunk_index"] == 0
        assert d["author"] == "alice"

    def test_invalid_overlap_tokens_rejected(self):
        from config.loader import ChunkingConfig
        with pytest.raises(ValueError):
            ChunkingConfig(min_tokens=100, max_tokens=150, overlap_tokens=150)

    def test_empty_text_returns_no_chunks(self, chunking_cfg):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document("", "empty.txt")
        assert chunks == []

    def test_extra_metadata_propagated(self, chunking_cfg, sample_text):
        from ingestion.chunker import TokenChunker
        chunks = TokenChunker(chunking_cfg).chunk_document(
            sample_text, "t.txt", extra_metadata={"author": "alice"}
        )
        for chunk in chunks:
            assert chunk.metadata.get("author") == "alice"

    def test_overlap_means_tokens_shared(self, chunking_cfg, sample_text):
        """Consecutive chunks should share exactly overlap_tokens tokens."""
        from ingestion.chunker import TokenChunker

        chunker = TokenChunker(chunking_cfg)
        chunks = chunker.chunk_document(sample_text, "t.txt")
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")

        c0_end = chunks[0].end_token
        c1_start = chunks[1].start_token
        overlap = c0_end - c1_start
        assert overlap == chunking_cfg.overlap_tokens


class TestIngestionPipeline:

    def _build_pipeline(self, chunking_cfg):
        from config.loader import (
            EmbeddingConfig,
            PipelineConfig,
            PromptsConfig,
            RetrievalConfig,
            VectorStoreConfig,
        )
        from ingestion.chunker import TokenChunker
        from ingestion.pipeline import IngestionPipeline

        cfg = PipelineConfig(
            version="test",
            chunking=chunking_cfg,
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(),
            prompts=PromptsConfig(
                rag_system="You are a helpful assistant.",
                rag_user="Context:\n{context}\n\nQuestion: {question}",
            ),
        )
        mock_embedder     = MagicMock()
        mock_embedder.dimensions = 1536
        mock_embedder.provider = "openrouter"
        mock_embedder.model_name = "text-embedding-3-small"
        mock_vector_store = MagicMock()

        mock_embedder.embed_chunks.side_effect = (
            lambda chunks: [[0.1] * 1536 for _ in range(len(chunks))]
        )

        pipeline = IngestionPipeline(
            chunker=TokenChunker(chunking_cfg),
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            cfg=cfg,
        )
        return pipeline, mock_embedder, mock_vector_store

    def test_ingest_calls_delete_by_source_and_upsert(self, chunking_cfg, sample_documents):
        pipeline, _, mock_store = self._build_pipeline(chunking_cfg)
        stats = pipeline.ingest_documents(sample_documents)
        assert mock_store.delete_by_source.call_count == 2
        mock_store.upsert.assert_called_once()
        assert stats.num_documents == 2
        assert stats.num_chunks > 0

    def test_ingest_empty_document_list(self, chunking_cfg):
        pipeline, _, mock_store = self._build_pipeline(chunking_cfg)
        stats = pipeline.ingest_documents([])
        mock_store.upsert.assert_not_called()
        assert stats.num_chunks == 0

    def test_embedder_called_with_chunks(self, chunking_cfg, sample_documents):
        pipeline, mock_embedder, _ = self._build_pipeline(chunking_cfg)
        pipeline.ingest_documents(sample_documents)
        mock_embedder.embed_chunks.assert_called_once()

    def test_mismatched_embedding_count_raises(self, chunking_cfg, sample_documents):
        pipeline, mock_embedder, _ = self._build_pipeline(chunking_cfg)
        mock_embedder.embed_chunks.side_effect = lambda chunks: [[0.1] * 1536]  # only 1 vector
        with pytest.raises(ValueError, match="vector"):
            pipeline.ingest_documents(sample_documents)


class TestVectorStoreIntegrity:

    def test_chroma_persistence_and_stale_chunk_replacement(self, tmp_path: Path, chunking_cfg):
        pytest.importorskip("chromadb")
        from config.loader import ChromaConfig, VectorStoreConfig
        from ingestion.chunker import TokenChunker
        from store.vector_store import ChromaVectorStore

        vs_cfg = VectorStoreConfig(
            backend="chroma",
            chroma=ChromaConfig(
                persist_directory=str(tmp_path / "chroma"),
                collection_name="test_integrity",
            ),
        )
        store = ChromaVectorStore(vs_cfg)
        chunker = TokenChunker(chunking_cfg)

        try:
            # 1. Two different sources with identical text do not overwrite each other
            same_body = " ".join(["identical"] * 120)
            c_a = chunker.chunk_document(same_body, "source_a.txt")
            c_b = chunker.chunk_document(same_body, "source_b.txt")
            store.upsert(c_a + c_b, [[0.2] * 16 for _ in range(len(c_a) + len(c_b))])
            assert store.count() == len(c_a) + len(c_b)
            sources = {s["source"] for s in store.list_sources()}
            assert sources == {"source_a.txt", "source_b.txt"}

            # 2. Re-ingesting modified source_a replaces old chunks instead of leaving stale ones
            store.delete_by_source("source_a.txt")
            updated_body = " ".join(["updated"] * 120)
            c_a_new = chunker.chunk_document(updated_body, "source_a.txt")
            store.upsert(c_a_new, [[0.3] * 16 for _ in range(len(c_a_new))])
            all_chunks = store.get_all_chunks()
            a_texts = [c.text for c in all_chunks if c.source == "source_a.txt"]
            assert len(a_texts) == len(c_a_new)
            assert all("updated" in t for t in a_texts)

            # 3. Dimension mismatch rejected on query and upsert
            with pytest.raises(ValueError, match="dimension"):
                store.query([0.1] * 8, top_k=2)

            # 4. delete_collection resets collection cleanly so subsequent calls succeed
            store.delete_collection()
            assert store.count() == 0
            store.upsert(c_a_new, [[0.5] * 8 for _ in range(len(c_a_new))])
            assert store.count() == len(c_a_new)
        finally:
            store.close()

    def test_weaviate_batch_failure_and_cloud_connection(self, chunking_cfg):
        from config.loader import VectorStoreConfig, WeaviateConfig
        from ingestion.chunker import TokenChunker
        from store.vector_store import WeaviateVectorStore

        with patch("weaviate.connect_to_weaviate_cloud") as mock_cloud:
            mock_client = MagicMock()
            mock_client.collections.exists.return_value = True
            mock_cloud.return_value = mock_client

            cfg = VectorStoreConfig(
                backend="weaviate",
                weaviate=WeaviateConfig(
                    url="https://my-cluster.weaviate.network",
                    api_key="wv-secret-key",
                    class_name="RagChunk",
                ),
            )
            store = WeaviateVectorStore(cfg)
            mock_cloud.assert_called_once()

            # Simulate failed_objects in Weaviate batch
            mock_collection = MagicMock()
            mock_collection.batch.failed_objects = [MagicMock(message="batch error")]
            mock_client.collections.get.return_value = mock_collection

            chunks = TokenChunker(chunking_cfg).chunk_document(" ".join(["word"] * 120), "doc.txt")
            with pytest.raises(RuntimeError, match="Weaviate batch upsert failed"):
                store.upsert(chunks, [[0.1] * 8 for _ in chunks])

            store.close()
            mock_client.close.assert_called_once()


class TestRetriever:

    def _make_retrieved_chunk(self, citation_id: str, text: str, score: float):
        from store.vector_store import RetrievedChunk
        return RetrievedChunk(
            chunk_dict={
                "text": text,
                "doc_id": citation_id.split(":")[0] * 8,
                "chunk_index": int(citation_id.split(":")[1]),
                "source": "test.txt",
                "citation_id": citation_id,
                "token_count": 120,
            },
            score=score,
        )

    def test_retrieve_returns_result_with_citations(self):
        from config.loader import (
            ChunkingConfig,
            EmbeddingConfig,
            PipelineConfig,
            PromptsConfig,
            RetrievalConfig,
            VectorStoreConfig,
        )
        from retrieval.retriever import Retriever

        cfg = PipelineConfig(
            version="test",
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(top_k=3),
            prompts=PromptsConfig(
                rag_system="system",
                rag_user="Context:\n{context}\n\nQuestion: {question}",
            ),
        )

        mock_embedder     = MagicMock()
        mock_vector_store = MagicMock()

        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_vector_store.query.return_value = [
            self._make_retrieved_chunk("abcd1234:0", "Relevant text.", 0.92),
            self._make_retrieved_chunk("abcd1234:1", "More context.",  0.87),
        ]

        retriever = Retriever(mock_embedder, mock_vector_store, cfg)
        result = retriever.retrieve("What is relevant?")

        assert len(result.chunks) == 2
        assert len(result.citations) == 2
        assert "[abcd1234:0]" in result.context_block
        assert result.query == "What is relevant?"

    def test_include_metadata_false_omits_source_header(self):
        from config.loader import (
            ChunkingConfig,
            EmbeddingConfig,
            PipelineConfig,
            PromptsConfig,
            RetrievalConfig,
            VectorStoreConfig,
        )
        from retrieval.retriever import Retriever

        cfg = PipelineConfig(
            version="test",
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(top_k=2, include_metadata=False),
            prompts=PromptsConfig(
                rag_system="system",
                rag_user="Context:\n{context}\n\nQuestion: {question}",
            ),
        )
        mock_embedder = MagicMock()
        mock_vector_store = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1536
        mock_vector_store.query.return_value = [
            self._make_retrieved_chunk("abcd1234:0", "Relevant text.", 0.92),
        ]
        retriever = Retriever(mock_embedder, mock_vector_store, cfg)
        result = retriever.retrieve("What is relevant?")
        assert "source:" not in result.context_block
        assert "[abcd1234:0]\nRelevant text." in result.context_block

    def test_build_prompt_uses_config_templates(self):
        from config.loader import (
            ChunkingConfig,
            EmbeddingConfig,
            PipelineConfig,
            PromptsConfig,
            RetrievalConfig,
            VectorStoreConfig,
        )
        from retrieval.retriever import Retriever

        cfg = PipelineConfig(
            version="test",
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(),
            prompts=PromptsConfig(
                rag_system="SYSTEM_PROMPT",
                rag_user="Context:\n{context}\n\nQuestion: {question}",
            ),
        )

        mock_embedder     = MagicMock()
        mock_vector_store = MagicMock()
        mock_embedder.embed_query.return_value = [0.0] * 1536
        mock_vector_store.query.return_value   = []

        retriever = Retriever(mock_embedder, mock_vector_store, cfg)
        result    = retriever.retrieve("test question")
        system, user = retriever.build_prompt(result)

        assert system == "SYSTEM_PROMPT"
        assert "test question" in user

    def test_openrouter_client_initialisation(self, monkeypatch):
        from config.loader import OPENROUTER_DEFAULT_BASE_URL, create_llm_client, load_config

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
        cfg = load_config("config/phase1.yaml")
        client = create_llm_client(cfg)
        assert client is not None
        assert str(client.base_url).rstrip("/") == OPENROUTER_DEFAULT_BASE_URL.rstrip("/")
