"""
mcp_server/server.py
Custom MCPServer exposing the project's Agentic RAG pipeline and ingestion tools over MCP SDK v2.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agent.mcp_client import MCPClientManager
from agent.orchestrator import AgenticRAGPipeline
from config.loader import load_config
from ingestion.pipeline import build_ingestion_pipeline

load_dotenv()


def _resolve_within_allowlist(given_path: str, allowlist_root: str) -> Path:
    """
    Resolves given_path and verifies it resides inside allowlist_root (after symlink resolution)
    and refers to an existing regular file.
    Raises ValueError if path attempts traversal outside allowlist_root or is not a regular file.
    """
    root = Path(allowlist_root).resolve()
    target = Path(given_path).resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path traversal rejected: '{given_path}' resolves outside the allowed root directory '{root}'."
        )

    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")

    if not target.is_file():
        raise ValueError(f"Target path is not a regular file: '{target}'.")

    return target


def create_server(config_path: str = "config/phase4.yaml") -> Any:
    """
    Factory creating an MCP SDK v2 MCPServer instance wired to the RAG and Agentic RAG pipelines.
    """
    try:
        from mcp.server import MCPServer
    except ImportError:
        raise ImportError("mcp>=2.3.0 package is required. Install via 'pip install \"mcp>=2.3.0\"'.")

    cfg = load_config(config_path)
    pipeline, bundle = build_ingestion_pipeline(config_path, cfg=cfg)
    allowlist_root = cfg.mcp.server.ingest_allowlist_root

    mcp_manager = MCPClientManager(cfg.mcp.servers)
    agentic_pipeline = AgenticRAGPipeline(bundle.retriever, mcp_manager, cfg)

    @asynccontextmanager
    async def _server_lifespan(_server: MCPServer):
        try:
            yield {}
        finally:
            await mcp_manager.close_all()
            if hasattr(pipeline.vector_store, "close"):
                pipeline.vector_store.close()

    mcp = MCPServer("rag-pipeline", lifespan=_server_lifespan)
    mcp._pipeline = pipeline  # type: ignore[attr-defined]
    mcp._bundle = bundle  # type: ignore[attr-defined]
    mcp._mcp_manager = mcp_manager  # type: ignore[attr-defined]
    mcp._agentic_pipeline = agentic_pipeline  # type: ignore[attr-defined]

    @mcp.tool()
    async def rag_query(
        question: str,
        top_k: Optional[int] = None,
        agentic: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using the ingested document corpus and (when agentic mode is enabled)
        external MCP tools such as web search.
        Returns answer text, inline citations, retrieved chunks, and agentic tool trace.
        """
        use_agentic = cfg.agent.enabled if agentic is None else bool(agentic)
        if use_agentic:
            ans = await agentic_pipeline.aanswer(question)
            return {
                "mode": "agentic",
                "answer": ans.answer,
                "citations": ans.citations,
                "chunks": [
                    c.to_dict() if hasattr(c, "to_dict") else {
                        "citation_id": c.citation_id,
                        "source": c.source,
                        "score": c.score,
                        "text": c.text,
                    }
                    for c in ans.retrieved_chunks
                ],
                "tool_trace": ans.tool_trace,
                "web_sources": ans.web_sources,
                "mcp_status": ans.mcp_status,
            }

        answer = await asyncio.to_thread(bundle.rag.answer, question, top_k=top_k)
        return {
            "mode": "rag",
            "answer": answer.answer,
            "citations": answer.citations,
            "chunks": [
                c.to_dict() if hasattr(c, "to_dict") else {
                    "citation_id": c.citation_id,
                    "source": c.source,
                    "score": c.score,
                    "text": c.text,
                }
                for c in answer.retrieved_chunks
            ],
        }

    @mcp.tool()
    def ingest_document(path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest a document into the corpus.
        'path' must resolve to a regular file inside the configured allow-listed root.
        """
        safe_path = _resolve_within_allowlist(path, allowlist_root)
        stats = pipeline.ingest_file(str(safe_path), metadata=metadata)
        return {
            "num_documents": stats.num_documents,
            "num_chunks": stats.num_chunks,
            "elapsed_seconds": stats.elapsed_seconds,
            "skipped": stats.skipped,
        }

    @mcp.tool()
    def list_ingested_sources() -> List[Dict[str, Any]]:
        """
        List all sources currently ingested in the vector store collection.
        """
        if hasattr(pipeline.vector_store, "list_sources"):
            return pipeline.vector_store.list_sources()
        raw = pipeline.vector_store._collection.get(include=["metadatas"])
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

    @mcp.tool()
    def delete_ingested_source(source: str) -> Dict[str, Any]:
        """
        Delete all chunks associated with a given source identifier from the vector store.
        """
        if hasattr(pipeline.vector_store, "delete_by_source"):
            pipeline.vector_store.delete_by_source(source)
            if pipeline.bm25_index is not None:
                pipeline.bm25_index.build_from_vector_store(pipeline.vector_store)
            return {"deleted_source": source, "status": "ok"}
        raise RuntimeError("Vector store backend does not support delete_by_source.")

    return mcp
