"""
ingestion/embedder.py
Converts text chunks into embedding vectors.

Supports:
  - OpenRouter / OpenAI-compatible embedding endpoints (default)
  - Local sentence-transformers (no API key needed)

Batching and transient-error retries are handled automatically.
"""
from __future__ import annotations

import os
import time
from typing import List

from dotenv import load_dotenv

from config.loader import OPENROUTER_DEFAULT_BASE_URL, EmbeddingConfig, get_api_key
from ingestion.chunker import Chunk

load_dotenv()


def _is_transient_error(exc: Exception) -> bool:
    """Return True if the API exception is a rate-limit, timeout, connection, or 5xx error."""
    cls_name = type(exc).__name__.lower()
    if any(k in cls_name for k in ("ratelimit", "timeout", "connection", "internalserver", "apiconnection", "serviceunavailable")):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("rate", "429", "timeout", "timed out", "temporarily unavailable", "502", "503", "504", "connection")
    )


class BaseEmbedder:
    provider: str = "unknown"
    model_name: str = "unknown"
    dimensions: int = 0

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        return self.embed_texts([c.text for c in chunks])

    def embed_query(self, query: str) -> List[float]:
        res = self.embed_texts([query])
        if not res or not res[0]:
            raise RuntimeError("Embedder returned an empty vector for query.")
        return res[0]


class OpenRouterEmbedder(BaseEmbedder):
    """
    Calls OpenRouter's (or OpenAI-compatible) embedding endpoint in batches.
    Requires OPENROUTER_API_KEY (or OPENAI_API_KEY) in environment or .env.
    """

    def __init__(self, cfg: EmbeddingConfig, max_retries: int = 3):
        if cfg.batch_size <= 0:
            raise ValueError("embedding.batch_size must be > 0")
        if cfg.dimensions <= 0:
            raise ValueError("embedding.dimensions must be > 0")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Run: pip install openai")

        api_key = get_api_key()
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY (or OPENAI_API_KEY) is not set. "
                "Add it to your .env file or export it in your shell."
            )

        base_url = (
            os.getenv("OPENROUTER_BASE_URL")
            or getattr(cfg, "base_url", None)
            or OPENROUTER_DEFAULT_BASE_URL
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.cfg = cfg
        self.max_retries = max(1, max_retries)
        self.provider = cfg.provider
        self.model_name = cfg.model
        self.dimensions = cfg.dimensions

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, batching and retrying transient API errors."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = self.cfg.batch_size
        if batch_size <= 0:
            raise ValueError("embedding.batch_size must be > 0")

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(self.max_retries):
                try:
                    try:
                        response = self.client.embeddings.create(
                            model=self.cfg.model,
                            input=batch,
                            dimensions=self.cfg.dimensions,
                        )
                    except Exception as inner_exc:
                        # Some OpenRouter upstream embedding providers do not accept `dimensions`
                        if "dimension" in str(inner_exc).lower() and not _is_transient_error(inner_exc):
                            response = self.client.embeddings.create(
                                model=self.cfg.model,
                                input=batch,
                            )
                        else:
                            raise

                    batch_vectors = [item.embedding for item in response.data]
                    if len(batch_vectors) != len(batch):
                        raise ValueError(
                            f"Embedding count mismatch: sent {len(batch)} texts, got {len(batch_vectors)} vectors."
                        )
                    for vec in batch_vectors:
                        if not vec:
                            raise ValueError("Received an empty embedding vector from API.")
                        if self.cfg.dimensions and len(vec) != self.cfg.dimensions:
                            raise ValueError(
                                f"Embedding dimension mismatch: expected {self.cfg.dimensions}, got {len(vec)}."
                            )

                    all_embeddings.extend(batch_vectors)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1 and _is_transient_error(e):
                        time.sleep(2 ** attempt)
                    else:
                        raise

        return all_embeddings


# Backward-compatible alias
OpenAIEmbedder = OpenRouterEmbedder


class LocalEmbedder(BaseEmbedder):
    """
    Uses a local sentence-transformers model.
    No API key required — good for offline / zero-cost usage.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, cfg: EmbeddingConfig):
        if cfg.batch_size <= 0:
            raise ValueError("embedding.batch_size must be > 0")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        model_name = cfg.model if cfg.provider == "local" else self.DEFAULT_MODEL
        self.model = SentenceTransformer(model_name)
        self.batch_size = cfg.batch_size
        self.cfg = cfg
        self.provider = "local"
        self.model_name = model_name

        actual_dim = self.model.get_sentence_embedding_dimension() or cfg.dimensions
        if cfg.provider == "local" and cfg.dimensions not in (actual_dim, 1536):
            raise ValueError(
                f"Configured embedding.dimensions ({cfg.dimensions}) does not match "
                f"local model '{model_name}' output dimension ({actual_dim})."
            )
        self.dimensions = int(actual_dim)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        result = embeddings.tolist()
        if len(result) != len(texts):
            raise ValueError(
                f"LocalEmbedder returned {len(result)} vectors for {len(texts)} texts."
            )
        for vec in result:
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"LocalEmbedder dimension mismatch: expected {self.dimensions}, got {len(vec)}."
                )
        return result


def build_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    """Return the correct embedder based on config."""
    if cfg.provider in ("openrouter", "openai"):
        return OpenRouterEmbedder(cfg)
    elif cfg.provider == "local":
        return LocalEmbedder(cfg)
    else:
        raise ValueError(f"Unknown embedding provider: {cfg.provider}")
