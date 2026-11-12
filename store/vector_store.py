"""
store/vector_store.py
Abstraction layer over the vector database.

Phase 1  →  ChromaDB  (local, zero-infra, persistent on disk)
Phase 1+ →  Weaviate  (local Docker or Weaviate Cloud)

The VectorStore interface is intentionally minimal so the retrieval layer
never touches database internals directly.
"""
from __future__ import annotations

import os
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional

from config.loader import VectorStoreConfig
from ingestion.chunker import PROTECTED_CHUNK_FIELDS, Chunk, sanitize_metadata


def _validate_upsert_inputs(
    chunks: List[Chunk],
    embeddings: List[List[float]],
    expected_dim: Optional[int] = None,
) -> int:
    """Validate that chunks and embeddings have matching counts and consistent dimensions."""
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Upsert mismatch: received {len(chunks)} chunk(s) but {len(embeddings)} embedding vector(s)."
        )
    if not embeddings:
        return expected_dim or 0

    first_dim = len(embeddings[0])
    if first_dim <= 0:
        raise ValueError("Embedding vectors must be non-empty.")
    if expected_dim is not None and first_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: vector has dimension {first_dim}, "
            f"but collection expects {expected_dim}."
        )
    for idx, vec in enumerate(embeddings):
        if len(vec) != first_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions in batch: index 0 has {first_dim}, "
                f"index {idx} has {len(vec)}."
            )
    return first_dim


class RetrievedChunk:
    """A chunk returned by a similarity search, decorated with a score."""

    def __init__(self, chunk_dict: dict, score: float):
        self.text: str        = chunk_dict["text"]
        self.doc_id: str      = chunk_dict["doc_id"]
        self.chunk_index: int = int(chunk_dict["chunk_index"])
        self.source: str      = chunk_dict["source"]
        self.citation_id: str = chunk_dict.get("citation_id") or f"{self.doc_id[:8]}:{self.chunk_index}"
        self.token_count: int = int(chunk_dict.get("token_count", 0))
        self.score: float     = float(score)
        self.metadata: dict   = sanitize_metadata(chunk_dict)

    def to_dict(self) -> dict:
        safe_meta = sanitize_metadata(self.metadata)
        return {
            **safe_meta,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "citation_id": self.citation_id,
            "token_count": self.token_count,
            "score": self.score,
        }

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(citation={self.citation_id!r}, "
            f"score={self.score:.4f}, tokens={self.token_count})"
        )


def _dict_to_chunk(d: dict) -> Chunk:
    return Chunk(
        text        = d.get("text", ""),
        doc_id      = d.get("doc_id", ""),
        chunk_index = int(d.get("chunk_index", 0)),
        source      = d.get("source", ""),
        token_count = int(d.get("token_count", 0)),
        start_token = int(d.get("start_token", 0)),
        end_token   = int(d.get("end_token", 0)),
        metadata    = sanitize_metadata(d),
    )


class BaseVectorStore:
    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        raise NotImplementedError

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        raise NotImplementedError

    def get_all_chunks(self) -> List[Chunk]:
        raise NotImplementedError

    def list_sources(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def delete_by_source(self, source: str) -> None:
        raise NotImplementedError

    def delete_document(self, doc_id: str) -> None:
        raise NotImplementedError

    def delete_collection(self) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        """Release any underlying database connections."""
        pass

    def __enter__(self) -> "BaseVectorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ChromaVectorStore(BaseVectorStore):
    """
    Persists chunks + embeddings in a local ChromaDB collection.
    Data survives restarts — re-ingestion replaces stale chunks per source.
    """

    def __init__(self, cfg: VectorStoreConfig):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Run: pip install chromadb")

        chroma_cfg = cfg.chroma
        self._collection_name = chroma_cfg.collection_name
        self._client = chromadb.PersistentClient(
            path=chroma_cfg.persist_directory
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_dimensions: Optional[int] = None
        self._embedding_provider: Optional[str] = None
        self._embedding_model: Optional[str] = None
        self._load_existing_embedding_metadata()

    def _load_existing_embedding_metadata(self) -> None:
        """Inspect existing records to recover the collection's embedding dimension and model."""
        try:
            if self._collection.count() > 0:
                sample = self._collection.get(limit=1, include=["metadatas", "embeddings"])
                metas = sample.get("metadatas") or []
                if metas and metas[0]:
                    meta0 = metas[0]
                    if "embedding_dimensions" in meta0:
                        self._embedding_dimensions = int(meta0["embedding_dimensions"])
                    if "embedding_provider" in meta0:
                        self._embedding_provider = str(meta0["embedding_provider"])
                    if "embedding_model" in meta0:
                        self._embedding_model = str(meta0["embedding_model"])
                embs = sample.get("embeddings")
                if self._embedding_dimensions is None and embs is not None and len(embs) > 0:
                    self._embedding_dimensions = len(embs[0])
        except Exception:
            pass

    def get_embedding_metadata(self) -> Dict[str, Any]:
        """Return the tracked embedding space metadata for this collection."""
        if self._embedding_dimensions is None and self.count() > 0:
            self._load_existing_embedding_metadata()
        return {
            "embedding_dimensions": self._embedding_dimensions,
            "embedding_provider": self._embedding_provider,
            "embedding_model": self._embedding_model,
        }

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Upsert chunks into Chroma.
        IDs are derived from doc_id + chunk_index; counts and dimensions are validated.
        """
        if not chunks and not embeddings:
            return

        if self._embedding_dimensions is None and self.count() > 0:
            self._load_existing_embedding_metadata()

        dim = _validate_upsert_inputs(chunks, embeddings, expected_dim=self._embedding_dimensions)

        incoming_provider = chunks[0].metadata.get("embedding_provider") if chunks else None
        incoming_model = chunks[0].metadata.get("embedding_model") if chunks else None
        if (
            self._embedding_model
            and incoming_model
            and self.count() > 0
            and self._embedding_model != incoming_model
        ):
            raise ValueError(
                f"Collection '{self._collection_name}' was built with embedding model "
                f"'{self._embedding_model}', cannot append vectors from '{incoming_model}'."
            )

        ids       = [f"{c.doc_id}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = []
        for c in chunks:
            m = c.to_dict()
            m.pop("text", None)
            m["embedding_dimensions"] = dim
            metadatas.append(m)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        self._embedding_dimensions = dim
        if incoming_provider:
            self._embedding_provider = str(incoming_provider)
        if incoming_model:
            self._embedding_model = str(incoming_model)

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        """Return top_k most similar chunks above score_threshold."""
        if top_k <= 0 or self.count() == 0:
            return []

        if self._embedding_dimensions is None:
            self._load_existing_embedding_metadata()
        if self._embedding_dimensions is not None and len(query_embedding) != self._embedding_dimensions:
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) does not match "
                f"collection dimension ({self._embedding_dimensions})."
            )

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        retrieved: List[RetrievedChunk] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - (dist / 2.0)

            if score < score_threshold:
                continue

            chunk_dict = {**(meta or {}), "text": doc}
            retrieved.append(RetrievedChunk(chunk_dict, score))

        return retrieved

    def get_all_chunks(self) -> List[Chunk]:
        if self.count() == 0:
            return []
        raw = self._collection.get(include=["documents", "metadatas"])
        return [
            _dict_to_chunk({**(meta or {}), "text": doc})
            for doc, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
        ]

    def list_sources(self) -> List[Dict[str, Any]]:
        if self.count() == 0:
            return []
        raw = self._collection.get(include=["metadatas"])
        sources_seen = set()
        sources_list = []
        for meta in raw.get("metadatas", []):
            if not meta:
                continue
            src = meta.get("source", "unknown")
            if src not in sources_seen:
                sources_seen.add(src)
                sources_list.append({
                    "source": src,
                    "doc_id": meta.get("doc_id", ""),
                    "format": meta.get("format", ""),
                })
        return sources_list

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks belonging to the given source identifier."""
        if self.count() == 0:
            return
        self._collection.delete(where={"source": source})
        if self.count() == 0:
            self._embedding_dimensions = None
            self._embedding_provider = None
            self._embedding_model = None

    def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to the given document ID."""
        if self.count() == 0:
            return
        self._collection.delete(where={"doc_id": doc_id})
        if self.count() == 0:
            self._embedding_dimensions = None
            self._embedding_provider = None
            self._embedding_model = None

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_dimensions = None
        self._embedding_provider = None
        self._embedding_model = None

    def count(self) -> int:
        return self._collection.count()


class WeaviateVectorStore(BaseVectorStore):
    """
    Weaviate backend.
    Supports both local Weaviate instances and Weaviate Cloud clusters.
    """

    def __init__(self, cfg: VectorStoreConfig):
        try:
            import weaviate
        except ImportError:
            raise ImportError("Run: pip install weaviate-client")

        wv_cfg = cfg.weaviate
        raw_url = wv_cfg.url if "://" in wv_cfg.url else f"http://{wv_cfg.url}"
        parsed = urllib.parse.urlparse(raw_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 8080)
        api_key = (
            wv_cfg.resolve_api_key()
            if hasattr(wv_cfg, "resolve_api_key")
            else (wv_cfg.api_key or os.getenv("WEAVIATE_API_KEY"))
        )

        is_cloud = bool(
            api_key
            or any(
                host.endswith(suffix)
                for suffix in (".weaviate.network", ".weaviate.cloud", ".semi.network", ".gcp.weaviate.cloud")
            )
            or (parsed.scheme == "https" and host not in ("localhost", "127.0.0.1"))
        )

        if is_cloud:
            import weaviate.classes.init as wvi

            auth_creds = wvi.Auth.api_key(api_key) if api_key else None
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=raw_url,
                auth_credentials=auth_creds,
            )
        else:
            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=getattr(wv_cfg, "grpc_port", 50051),
            )

        self._class_name = wv_cfg.class_name
        self._embedding_dimensions: Optional[int] = None
        self._embedding_provider: Optional[str] = None
        self._embedding_model: Optional[str] = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create Weaviate class if it doesn't exist."""
        import weaviate.classes as wvc

        if not self._client.collections.exists(self._class_name):
            self._client.collections.create(
                name=self._class_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(name="text",                 data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="doc_id",               data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="chunk_index",          data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="source",               data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="citation_id",          data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="token_count",          data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="start_token",          data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="end_token",            data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="embedding_dimensions", data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="embedding_provider",   data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="embedding_model",      data_type=wvc.config.DataType.TEXT),
                ],
            )

    def get_embedding_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_dimensions": self._embedding_dimensions,
            "embedding_provider": self._embedding_provider,
            "embedding_model": self._embedding_model,
        }

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        if not chunks and not embeddings:
            return

        dim = _validate_upsert_inputs(chunks, embeddings, expected_dim=self._embedding_dimensions)

        collection = self._client.collections.get(self._class_name)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, embeddings):
                obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.doc_id}_{chunk.chunk_index}")
                props = chunk.to_dict()
                props["embedding_dimensions"] = dim
                batch.add_object(
                    uuid=obj_uuid,
                    properties=props,
                    vector=vector,
                )
            batch_errors = getattr(batch, "number_errors", 0) or 0

        failed_objects = getattr(collection.batch, "failed_objects", None) or []
        if failed_objects or batch_errors > 0:
            raise RuntimeError(
                f"Weaviate batch upsert failed for {len(failed_objects) or batch_errors} object(s): "
                f"{failed_objects[:3]}"
            )

        self._embedding_dimensions = dim
        if chunks:
            if "embedding_provider" in chunks[0].metadata:
                self._embedding_provider = str(chunks[0].metadata["embedding_provider"])
            if "embedding_model" in chunks[0].metadata:
                self._embedding_model = str(chunks[0].metadata["embedding_model"])

    def query(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> List[RetrievedChunk]:
        if top_k <= 0:
            return []
        if self._embedding_dimensions is not None and len(query_embedding) != self._embedding_dimensions:
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) does not match "
                f"Weaviate collection dimension ({self._embedding_dimensions})."
            )
        collection = self._client.collections.get(self._class_name)
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=["certainty"],
        )
        retrieved = []
        for obj in response.objects:
            score = obj.metadata.certainty or 0.0
            if score >= score_threshold:
                retrieved.append(RetrievedChunk(dict(obj.properties), score))
        return retrieved

    def get_all_chunks(self) -> List[Chunk]:
        collection = self._client.collections.get(self._class_name)
        chunks = []
        for item in collection.iterator():
            chunks.append(_dict_to_chunk(dict(item.properties)))
        return chunks

    def list_sources(self) -> List[Dict[str, Any]]:
        chunks = self.get_all_chunks()
        sources_seen = set()
        sources_list = []
        for c in chunks:
            if c.source not in sources_seen:
                sources_seen.add(c.source)
                sources_list.append({
                    "source": c.source,
                    "doc_id": c.doc_id,
                    "format": c.metadata.get("format", ""),
                })
        return sources_list

    def delete_by_source(self, source: str) -> None:
        import weaviate.classes as wvc

        collection = self._client.collections.get(self._class_name)
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("source").equal(source)
        )

    def delete_document(self, doc_id: str) -> None:
        import weaviate.classes as wvc

        collection = self._client.collections.get(self._class_name)
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("doc_id").equal(doc_id)
        )

    def delete_collection(self) -> None:
        self._client.collections.delete(self._class_name)
        self._ensure_schema()
        self._embedding_dimensions = None
        self._embedding_provider = None
        self._embedding_model = None

    def count(self) -> int:
        collection = self._client.collections.get(self._class_name)
        return collection.aggregate.over_all(total_count=True).total_count

    def close(self) -> None:
        if getattr(self, "_client", None) is not None:
            self._client.close()


def build_vector_store(cfg: VectorStoreConfig) -> BaseVectorStore:
    if cfg.backend == "chroma":
        return ChromaVectorStore(cfg)
    elif cfg.backend == "weaviate":
        return WeaviateVectorStore(cfg)
    else:
        raise ValueError(f"Unknown vector store backend: {cfg.backend}")
