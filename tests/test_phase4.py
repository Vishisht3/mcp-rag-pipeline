"""
tests/test_phase4.py
Unit and integration tests for Phase 4:
  - Multi-format ingestion parsers (txt, md, pdf + ocr_fallback, docx, tex)
  - Parser registry & magic byte fallback
  - Ingestion directory wide glob & skipped files tracking
  - Agentic orchestrator, routing strategies, event-loop safety, and guaranteed cleanup
  - MCP Client Manager (MCP SDK v2: input_schema, is_error, structured_content, streamable_http_client, negotiate_auto, env validation)
  - Custom MCPServer (v2) end-to-end in-process client/server integration & path-traversal security
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.loader import (
    AgentConfig,
    ChunkingConfig,
    CitationEnforcementConfig,
    EmbeddingConfig,
    IngestionConfig,
    MCPConfig,
    MCPOwnServerConfig,
    MCPServerConnectionConfig,
    PipelineConfig,
    PromptsConfig,
    RetrievalConfig,
    VectorStoreConfig,
    load_config,
)
from ingestion.parsers.base import UnsupportedFormatError
from ingestion.parsers.docx_parser import DocxParser
from ingestion.parsers.latex_parser import LatexParser
from ingestion.parsers.pdf_parser import PDFParser
from ingestion.parsers.registry import get_parser_for
from ingestion.parsers.text_parser import TextParser
from store.vector_store import RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "sample.pdf"
    if not p.exists():
        try:
            import pypdf
            writer = pypdf.PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with open(p, "wb") as f:
                writer.write(f)
        except Exception:
            p.write_bytes(b"%PDF-1.4 sample pdf content")
    return p


@pytest.fixture
def sample_docx_path(fixtures_dir: Path) -> Path:
    p = fixtures_dir / "sample.docx"
    if not p.exists():
        try:
            import docx
            doc = docx.Document()
            doc.add_paragraph("Sample Word paragraph text.")
            doc.save(str(p))
        except Exception:
            p.write_bytes(b"PK\x03\x04 sample docx content")
    return p


@pytest.fixture
def sample_tex_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "sample.tex"


@pytest.fixture
def phase4_cfg() -> PipelineConfig:
    return PipelineConfig(
        version="4.0.0",
        chunking=ChunkingConfig(min_tokens=50, max_tokens=100, overlap_tokens=10),
        embedding=EmbeddingConfig(),
        vector_store=VectorStoreConfig(),
        retrieval=RetrievalConfig(top_k=3, score_threshold=0.75),
        citation_enforcement=CitationEnforcementConfig(enabled=True, min_citations_required=1),
        prompts=PromptsConfig(
            rag_system="rag system",
            rag_user="Context:\n{context}\n\nQuestion: {question}",
            agent_system="agent system prompt",
        ),
        ingestion=IngestionConfig(),
        agent=AgentConfig(enabled=True, strategy="auto", max_tool_iterations=4, confidence_threshold=0.75),
        mcp=MCPConfig(
            servers=[
                MCPServerConnectionConfig(
                    name="web_search",
                    enabled=True,
                    command="npx",
                    args=["-y", "tavily-mcp"],
                    tool_allowlist=["tavily_search", "tavily_extract"],
                )
            ],
            server=MCPOwnServerConfig(ingest_allowlist_root="./ingest_data"),
        ),
    )


# ── Test Parsers ──────────────────────────────────────────────────────────────

class TestParsers:
    def test_text_parser(self, fixtures_dir: Path):
        parser = TextParser()
        txt_doc = parser.parse(fixtures_dir / "sample.txt")
        assert "Plain text file" in txt_doc.text
        assert txt_doc.metadata["format"] == "txt"

        md_doc = parser.parse(fixtures_dir / "sample.md")
        assert "Markdown Sample Document" in md_doc.text
        assert md_doc.metadata["format"] == "md"

    def test_latex_parser(self, sample_tex_path: Path):
        pytest.importorskip("pylatexenc")
        parser = LatexParser()
        parsed = parser.parse(sample_tex_path)
        assert "Sample LaTeX Document" in parsed.text
        assert r"\documentclass" not in parsed.text
        assert parsed.metadata["format"] == "tex"

    def test_pdf_parser_missing_dep(self, monkeypatch, sample_pdf_path: Path):
        import sys
        parser = PDFParser()
        monkeypatch.setitem(sys.modules, "pypdf", None)
        with pytest.raises(ImportError) as exc_info:
            parser.parse(sample_pdf_path)
        assert "pypdf" in str(exc_info.value)

    def test_docx_parser_missing_dep(self, monkeypatch, sample_docx_path: Path):
        import sys
        parser = DocxParser()
        monkeypatch.setitem(sys.modules, "docx", None)
        with pytest.raises(ImportError) as exc_info:
            parser.parse(sample_docx_path)
        assert "python-docx" in str(exc_info.value)

    def test_parser_registry(self, fixtures_dir: Path):
        parser_txt = get_parser_for(fixtures_dir / "sample.txt")
        assert isinstance(parser_txt, TextParser)

        parser_tex = get_parser_for(fixtures_dir / "sample.tex")
        assert isinstance(parser_tex, LatexParser)

        parser_pdf = get_parser_for(fixtures_dir / "sample.pdf", ocr_fallback=True)
        assert isinstance(parser_pdf, PDFParser)
        assert parser_pdf.ocr_fallback is True

        with pytest.raises(UnsupportedFormatError):
            get_parser_for(Path("unknown_file.xyz_unsupported"))

    def test_scanned_pdf_returns_empty_text(self, tmp_path: Path):
        """Scanned PDF with 0 extractable text should return ParsedDocument with empty text."""
        pytest.importorskip("pypdf")
        import pypdf
        pdf_file = tmp_path / "scanned.pdf"
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        with open(pdf_file, "wb") as f:
            writer.write(f)

        parser = PDFParser()
        doc = parser.parse(pdf_file)
        assert doc.text == ""
        assert doc.metadata["page_count"] == 1


# ── Test Ingestion Multi-Format ───────────────────────────────────────────────

class TestIngestionMultiFormat:
    def test_ingest_directory_skips_unsupported(self, tmp_path: Path, phase4_cfg: PipelineConfig):
        from ingestion.chunker import TokenChunker
        from ingestion.pipeline import IngestionPipeline

        (tmp_path / "doc1.txt").write_text("Hello text file context.")
        (tmp_path / "doc2.unknown_ext").write_text("Binary or unknown file.")

        mock_embedder = MagicMock()
        mock_embedder.dimensions = 1536
        mock_embedder.provider = "openrouter"
        mock_embedder.model_name = "text-embedding-3-small"
        mock_store = MagicMock()
        mock_embedder.embed_chunks.side_effect = lambda chunks: [[0.1] * 1536 for _ in range(len(chunks))]

        pipeline = IngestionPipeline(
            chunker=TokenChunker(phase4_cfg.chunking),
            embedder=mock_embedder,
            vector_store=mock_store,
            cfg=phase4_cfg,
        )

        stats = pipeline.ingest_directory(str(tmp_path))
        assert stats.num_documents == 1
        assert len(stats.skipped) == 1
        assert "doc2.unknown_ext" in stats.skipped[0]["file"]


# ── Test Citation Enforcer Agentic ───────────────────────────────────────────

class TestAgenticCitationEnforcement:
    def _make_retrieved_chunk(self, cid: str) -> RetrievedChunk:
        return RetrievedChunk(
            {
                "text": "sample text",
                "doc_id": cid.split(":")[0],
                "chunk_index": int(cid.split(":")[1]),
                "source": "source.txt",
                "citation_id": cid,
            },
            score=0.9,
        )

    def test_mixed_citations_valid(self, phase4_cfg: PipelineConfig):
        from retrieval.citation_enforcer import CitationEnforcer

        enforcer = CitationEnforcer(phase4_cfg.citation_enforcement)
        chunks = [self._make_retrieved_chunk("abcd1234:0")]
        web_sources = [{"url": "https://example.com"}]

        answer = "Local claim [abcd1234:0] and web claim [web:1]."
        res = enforcer.check_agentic(answer, chunks, web_sources)
        assert res.is_valid
        assert len(res.found_citations) == 2

    def test_hallucinated_web_citation_fails(self, phase4_cfg: PipelineConfig):
        from retrieval.citation_enforcer import CitationEnforcer

        enforcer = CitationEnforcer(phase4_cfg.citation_enforcement)
        chunks = [self._make_retrieved_chunk("abcd1234:0")]
        web_sources = [{"url": "https://example.com"}]

        answer = "Local claim [abcd1234:0] and bad web claim [web:5]."
        res = enforcer.check_agentic(answer, chunks, web_sources)
        assert not res.is_valid
        assert "web:5" in res.invalid_citations


# ── Test Agentic Orchestrator ────────────────────────────────────────────────

class TestAgenticOrchestrator:
    def test_agentic_pipeline_local_only(self, phase4_cfg: PipelineConfig):
        from agent.orchestrator import AgenticRAGPipeline

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = MagicMock(
            chunks=[
                RetrievedChunk(
                    {
                        "text": "RAG reduces hallucinations",
                        "doc_id": "abcd1234",
                        "chunk_index": 0,
                        "source": "rag.txt",
                        "citation_id": "abcd1234:0",
                    },
                    score=0.95,
                )
            ],
            citations=["[abcd1234:0]"],
        )
        mock_retriever.enforcer = MagicMock()
        mock_retriever.enforcer.check_agentic.side_effect = lambda ans, chunks, web: MagicMock(
            is_valid=True, answer=ans
        )
        mock_retriever.enforcer.enforce_agentic.side_effect = lambda ans, chunks, web, retry_fn=None: MagicMock(
            is_valid=True, answer=ans
        )
        mock_mcp_manager = MagicMock()
        mock_mcp_manager.connect_all = AsyncMock(return_value={"web_search": {"connected": True, "tools": []}})
        mock_mcp_manager.close_all = AsyncMock()
        mock_mcp_manager.get_tool_schemas.return_value = []

        pipeline = AgenticRAGPipeline(mock_retriever, mock_mcp_manager, phase4_cfg)

        mock_llm = MagicMock()
        pipeline._llm = mock_llm

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "search_local_documents"
        tool_call.function.arguments = json.dumps({"query": "RAG"})

        msg1 = MagicMock()
        msg1.tool_calls = [tool_call]
        msg1.content = None

        msg2 = MagicMock()
        msg2.tool_calls = None
        msg2.content = "RAG combines retrieval with generation [abcd1234:0]."

        resp1 = MagicMock(choices=[MagicMock(message=msg1)])
        resp2 = MagicMock(choices=[MagicMock(message=msg2)])
        mock_llm.chat.completions.create.side_effect = [resp1, resp2]

        ans = pipeline.answer("What is RAG?")
        assert "[abcd1234:0]" in ans.answer
        assert len(ans.tool_trace) == 1
        assert ans.tool_trace[0]["tool"] == "search_local_documents"
        mock_mcp_manager.close_all.assert_awaited_once()

    def test_agentic_pipeline_closes_mcp_on_llm_exception(self, phase4_cfg: PipelineConfig):
        from agent.orchestrator import AgenticRAGPipeline

        mock_retriever = MagicMock()
        mock_mcp_manager = MagicMock()
        mock_mcp_manager.connect_all = AsyncMock(return_value={})
        mock_mcp_manager.close_all = AsyncMock()
        mock_mcp_manager.get_tool_schemas.return_value = []

        pipeline = AgenticRAGPipeline(mock_retriever, mock_mcp_manager, phase4_cfg)
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.side_effect = RuntimeError("LLM network failure")
        pipeline._llm = mock_llm

        with pytest.raises(RuntimeError, match="LLM network failure"):
            pipeline.answer("Will close_all still run?")

        mock_mcp_manager.close_all.assert_awaited_once()


# ── Test MCP Client Manager (MCP SDK v2) ──────────────────────────────────────

class TestMCPClientManagerV2:
    def test_phase4_yaml_tavily_allowlist_has_real_tool_names(self):
        cfg = load_config("config/phase4.yaml")
        web_srv = next(s for s in cfg.mcp.servers if s.name == "web_search")
        assert "tavily_search" in web_srv.tool_allowlist
        assert "tavily_extract" in web_srv.tool_allowlist

    def test_missing_env_var_raises_value_error(self, monkeypatch):
        from agent.mcp_client import _substitute_env_vars
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            _substitute_env_vars({"TAVILY_API_KEY": "${TAVILY_API_KEY}"}, server_name="web_search")

    def test_duplicate_server_names_rejected(self):
        from agent.mcp_client import MCPClientManager
        s1 = MCPServerConnectionConfig(name="dup", command="npx")
        s2 = MCPServerConnectionConfig(name="dup", command="npx")
        with pytest.raises(ValueError, match="Duplicate MCP server name"):
            MCPClientManager([s1, s2])

    def test_v2_tool_input_schema_is_error_and_structured_content(self):
        from mcp.types import CallToolResult, TextContent, Tool
        from agent.mcp_client import MCPClientManager

        srv_cfg = MCPServerConnectionConfig(
            name="web_search",
            enabled=True,
            transport="streamable-http",
            url="http://localhost:9999/mcp",
            tool_allowlist=["tavily_search"],
        )
        mgr = MCPClientManager([srv_cfg])

        real_v2_tool = Tool(
            name="tavily_search",
            description="Search the web via Tavily",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        filtered_tool = Tool(
            name="unlisted_tool",
            description="Should be filtered by allowlist",
            input_schema={"type": "object", "properties": {}},
        )

        mock_session = MagicMock()
        mock_session.send_discover = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[real_v2_tool, filtered_tool]))

        structured_res = CallToolResult(
            content=[TextContent(type="text", text="https://example.com/article")],
            structured_content={"results": [{"url": "https://example.com/article", "title": "Example"}]},
            is_error=False,
        )
        error_res = CallToolResult(
            content=[TextContent(type="text", text="Rate limit exceeded")],
            is_error=True,
        )
        mock_session.call_tool = AsyncMock(side_effect=[structured_res, error_res])

        @asynccontextmanager
        async def fake_http_client(url):
            yield (MagicMock(), MagicMock())

        @asynccontextmanager
        async def fake_client_session(r, w):
            yield mock_session

        async def _run():
            with (
                patch("mcp.client.streamable_http.streamable_http_client", side_effect=fake_http_client) as mock_http,
                patch("mcp.ClientSession", side_effect=fake_client_session),
                patch("mcp.client._probe.negotiate_auto", new=AsyncMock()) as mock_neg,
            ):
                status = await mgr.connect_all()
                mock_http.assert_called_once_with("http://localhost:9999/mcp")
                mock_neg.assert_awaited_once_with(mock_session)

                assert status["web_search"]["connected"] is True
                schemas = mgr.get_tool_schemas()
                assert len(schemas) == 1
                fn_schema = schemas[0]["function"]
                assert fn_schema["name"] == "web_search__tavily_search"
                assert "query" in fn_schema["parameters"]["properties"]
                assert fn_schema["parameters"]["required"] == ["query"]

                # Successful call with structured_content
                ok_out = await mgr.call_tool("web_search__tavily_search", {"query": "MCP v2"})
                assert ok_out["is_error"] is False
                assert "error" not in ok_out
                assert ok_out["structured_content"]["results"][0]["url"] == "https://example.com/article"

                # Error call with v2 is_error=True
                err_out = await mgr.call_tool("web_search__tavily_search", {"query": "fail"})
                assert err_out["is_error"] is True
                assert "Rate limit exceeded" in err_out["error"]

                await mgr.close_all()

        asyncio.run(_run())


# ── Test MCP Server (v2 MCPServer) & Security ─────────────────────────────────

class TestMCPServerSecurityAndIntegration:
    def test_path_traversal_rejection(self, tmp_path: Path):
        from mcp_server.server import _resolve_within_allowlist

        allowlist_root = tmp_path / "allowed"
        allowlist_root.mkdir()
        secret_file = tmp_path / "secret.txt"
        secret_file.write_text("secret content")

        with pytest.raises(ValueError) as exc_info:
            _resolve_within_allowlist(str(secret_file), str(allowlist_root))
        assert "Path traversal rejected" in str(exc_info.value)

    def test_directory_path_rejected_as_non_regular_file(self, tmp_path: Path):
        from mcp_server.server import _resolve_within_allowlist

        allowlist_root = tmp_path / "allowed"
        allowlist_root.mkdir()
        subdir = allowlist_root / "subdir"
        subdir.mkdir()

        with pytest.raises(ValueError, match="regular file"):
            _resolve_within_allowlist(str(subdir), str(allowlist_root))

    def test_valid_path_resolves(self, tmp_path: Path):
        from mcp_server.server import _resolve_within_allowlist

        allowlist_root = tmp_path / "allowed"
        allowlist_root.mkdir()
        valid_file = allowlist_root / "doc.txt"
        valid_file.write_text("valid content")

        resolved = _resolve_within_allowlist(str(valid_file), str(allowlist_root))
        assert resolved == valid_file.resolve()

    def test_mcp_server_v2_end_to_end_in_process(self, tmp_path: Path):
        """Verify create_server() builds a real MCP SDK v2 MCPServer and serves tools over mcp.Client."""
        from mcp import Client
        from mcp.server import MCPServer
        from mcp_server.server import create_server

        allow_dir = tmp_path / "ingest_allowed"
        allow_dir.mkdir()
        doc_path = allow_dir / "notes.txt"
        doc_path.write_text("Retrieval augmented generation grounds answers in retrieved documents. " * 20)

        cfg_file = tmp_path / "test_phase4.yaml"
        raw_cfg = Path("config/phase4.yaml").read_text(encoding="utf-8")
        raw_cfg = raw_cfg.replace("./chroma_db", str(tmp_path / "chroma_mcp").replace("\\", "/"))
        raw_cfg = raw_cfg.replace("./ingest_data", str(allow_dir).replace("\\", "/"))
        raw_cfg = raw_cfg.replace("enabled: true\n      transport: \"stdio\"", "enabled: false\n      transport: \"stdio\"")
        cfg_file.write_text(raw_cfg, encoding="utf-8")

        mock_embedder = MagicMock()
        mock_embedder.dimensions = 16
        mock_embedder.provider = "local"
        mock_embedder.model_name = "mock-embed"
        mock_embedder.embed_chunks.side_effect = lambda chunks: [[0.25] * 16 for _ in chunks]
        mock_embedder.embed_query.return_value = [0.25] * 16

        with patch("ingestion.pipeline.build_embedder", return_value=mock_embedder):
            server = create_server(config_path=str(cfg_file))
            assert isinstance(server, MCPServer)

            # Mock LLM on the server's agentic pipeline to test rag_query tool end-to-end
            mock_llm = MagicMock()
            msg = MagicMock()
            msg.tool_calls = None
            msg.content = "Grounded answer from agentic server [abcd1234:0]."
            mock_llm.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=msg)])
            server._agentic_pipeline._llm = mock_llm
            server._agentic_pipeline.retriever.enforcer.cfg.enabled = False

            async def _exercise_server():
                async with Client(server, mode="auto") as client:
                    tools_res = await client.list_tools()
                    tool_names = {t.name for t in tools_res.tools}
                    assert {"rag_query", "ingest_document", "list_ingested_sources", "delete_ingested_source"} <= tool_names

                    ingest_res = await client.call_tool("ingest_document", {"path": str(doc_path)})
                    assert not ingest_res.is_error

                    sources_res = await client.call_tool("list_ingested_sources", {})
                    assert not sources_res.is_error
                    assert "notes.txt" in str(sources_res.content)

                    query_res = await client.call_tool("rag_query", {"question": "What does RAG do?"})
                    assert not query_res.is_error
                    assert "Grounded answer from agentic server" in str(query_res.content)

                    del_res = await client.call_tool("delete_ingested_source", {"source": "notes.txt"})
                    assert not del_res.is_error

            asyncio.run(_exercise_server())
            server._pipeline.vector_store.close()
