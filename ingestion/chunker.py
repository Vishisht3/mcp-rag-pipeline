"""
ingestion/chunker.py
Splits raw documents into token-bounded chunks (500-800 tokens) with a
sliding-window overlap so context is never cut mid-thought.

Each chunk carries full provenance metadata so the retriever can surface
precise citations later:  source -> doc_id -> chunk_index -> token span.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken

from config.loader import ChunkingConfig

PROTECTED_CHUNK_FIELDS = frozenset(
    {
        "text",
        "doc_id",
        "chunk_index",
        "source",
        "token_count",
        "start_token",
        "end_token",
        "citation_id",
        "score",
    }
)


def sanitize_metadata(metadata: Optional[dict]) -> dict:
    """Return a copy of metadata with all system-owned protected keys removed."""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if k not in PROTECTED_CHUNK_FIELDS}


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    text: str
    doc_id: str
    chunk_index: int
    source: str
    token_count: int
    start_token: int
    end_token: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = sanitize_metadata(self.metadata)

    @property
    def citation_id(self) -> str:
        """Human-readable citation key: <doc_id_short>:<chunk_index>"""
        return f"{self.doc_id[:8]}:{self.chunk_index}"

    def to_dict(self) -> dict:
        safe_meta = sanitize_metadata(self.metadata)
        return {
            **safe_meta,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "token_count": self.token_count,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "citation_id": self.citation_id,
        }


class TokenChunker:
    """
    Splits a document into chunks of [min_tokens, max_tokens] with an
    overlap_tokens sliding window.
    """

    def __init__(self, cfg: ChunkingConfig):
        if cfg.max_tokens <= 0 or cfg.min_tokens <= 0 or cfg.min_tokens > cfg.max_tokens:
            raise ValueError("Invalid chunking token bounds: require 0 < min_tokens <= max_tokens.")
        if cfg.overlap_tokens < 0 or cfg.overlap_tokens >= cfg.max_tokens:
            raise ValueError("overlap_tokens must be >= 0 and strictly less than max_tokens.")
        self.cfg = cfg
        self.enc = tiktoken.get_encoding(cfg.tokenizer)

    def chunk_document(
        self,
        text: str,
        source: str,
        extra_metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Chunk a single document string.

        Args:
            text:           Raw document text.
            source:         Human-readable identifier (filename, URL, etc.).
            extra_metadata: Any additional key/values stored on every chunk.

        Returns:
            Ordered list of Chunk objects.
        """
        step = self.cfg.max_tokens - self.cfg.overlap_tokens
        if step <= 0:
            raise ValueError("overlap_tokens must be strictly less than max_tokens.")

        doc_id = _stable_hash(text, source=source)
        tokens = self.enc.encode(text)
        metadata = sanitize_metadata(extra_metadata)

        if not tokens:
            return []

        windows: List[tuple] = []

        start = 0
        while start < len(tokens):
            end = min(start + self.cfg.max_tokens, len(tokens))
            windows.append((start, end))
            if end == len(tokens):
                break
            start += step

        if len(windows) > 1:
            last_start, last_end = windows[-1]
            if (last_end - last_start) < self.cfg.min_tokens:
                prev_start, _ = windows[-2]
                # Only merge when the combined span stays within max_tokens.
                if (last_end - prev_start) <= self.cfg.max_tokens:
                    windows.pop()
                    windows[-1] = (prev_start, last_end)

        chunks: List[Chunk] = []
        for idx, (start, end) in enumerate(windows):
            chunk_tokens = tokens[start:end]
            chunk_text = self.enc.decode(chunk_tokens)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    doc_id=doc_id,
                    chunk_index=idx,
                    source=source,
                    token_count=len(chunk_tokens),
                    start_token=start,
                    end_token=end,
                    metadata=dict(metadata),
                )
            )

        return chunks

    def chunk_documents(self, documents: List[dict]) -> List[Chunk]:
        """
        Chunk a list of document dicts.

        Each dict must have:
          - "text"   (str)  - document content
          - "source" (str)  - identifier
          - "metadata" (dict, optional)
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.chunk_document(
                text=doc["text"],
                source=doc["source"],
                extra_metadata=doc.get("metadata"),
            )
            all_chunks.extend(chunks)
        return all_chunks


def _stable_hash(text: str, source: str = "") -> str:
    """SHA-256 hash of source + text, hex-encoded. Stable across runs and source-aware."""
    payload = f"{source}\x00{text}" if source else text
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
