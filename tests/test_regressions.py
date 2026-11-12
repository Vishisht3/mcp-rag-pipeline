"""
tests/test_regressions.py
Regression tests for the five bugs reported against the post-audit upload:

  1. --retrieval-only / no-API-key runs must gate CI on context_recall only.
  2. Local-embedding fallback must use an isolated collection (no 1536-d vs 384-d clash).
  3. Empty vector store auto-seeds SAMPLE_DOCS (unless --ingest-dir / --no-auto-seed).
  4. `python -m mcp_server --http --port N` and Weaviate `api_key_env` / `index_name` keys.
  5. Agentic citation retry / enforcement must run off the event-loop thread.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.loader import WeaviateConfig, load_config


# ── Bug 1 ─────────────────────────────────────────────────────────────────────

class TestRetrievalOnlyThresholdScope:

    def test_retrieval_scope_keeps_only_context_recall(self):
        from config.loader import ThresholdsConfig
        from eval.run_eval import select_active_thresholds

        active = select_active_thresholds(ThresholdsConfig(context_recall=0.6), retrieval_scope=True)
        assert active == {"context_recall": 0.6}

    def test_full_scope_keeps_all_metrics(self):
        from config.loader import ThresholdsConfig
        from eval.run_eval import select_active_thresholds

        active = select_active_thresholds(ThresholdsConfig(), retrieval_scope=False)
        assert set(active) == {"faithfulness", "answer_relevance", "context_recall", "citation_coverage"}

    def test_retrieval_only_report_passes_when_recall_meets_threshold(self):
        from config.loader import ThresholdsConfig
        from eval.run_eval import select_active_thresholds
        from eval.scorer import SampleScore, build_report

        scores = [
            SampleScore(
                id=f"q{i}", question="q", answer="", ground_truth="gt",
                faithfulness=None, answer_relevance=None, context_recall=0.9,
                citation_coverage=None, tool_selection_accuracy=None,
            )
            for i in range(3)
        ]
        retrieval_report = build_report(scores, select_active_thresholds(ThresholdsConfig(), True))
        assert retrieval_report.ci_passed
        assert retrieval_report.threshold_breaches == []

        # Same scores under full gating must still fail on the unmeasured metrics (dry-run semantics).
        full_report = build_report(scores, select_active_thresholds(ThresholdsConfig(), False))
        assert not full_report.ci_passed
        assert {"faithfulness", "answer_relevance", "citation_coverage"} <= set(full_report.threshold_breaches)


# ── Bug 2 ─────────────────────────────────────────────────────────────────────

class TestLocalCollectionIsolation:

    def test_collection_and_class_names_are_suffixed_once(self):
        from eval.run_eval import _isolate_local_collection

        cfg = load_config("config/phase3.yaml").model_copy(deep=True)
        base_chroma = cfg.vector_store.chroma.collection_name
        base_class = cfg.vector_store.weaviate.class_name

        _isolate_local_collection(cfg, 384)
        _isolate_local_collection(cfg, 384)  # idempotent

        assert cfg.vector_store.chroma.collection_name == f"{base_chroma}_local384"
        assert cfg.vector_store.weaviate.class_name == f"{base_class}Local384"

    def test_local_fallback_does_not_touch_existing_api_collection(self, tmp_path):
        """A persisted 1536-d collection must not break an offline 384-d run."""
        pytest.importorskip("chromadb")
        from config.loader import ChromaConfig, VectorStoreConfig
        from eval.run_eval import _isolate_local_collection
        from ingestion.chunker import Chunk
        from store.vector_store import build_vector_store

        vs_cfg = VectorStoreConfig(
            backend="chroma",
            chroma=ChromaConfig(persist_directory=str(tmp_path / "db"), collection_name="rag_phase3"),
        )
        api_store = build_vector_store(vs_cfg)
        chunk = Chunk(text="hello world", doc_id="d" * 16, chunk_index=0, source="a.txt",
                      token_count=2, start_token=0, end_token=2)
        api_store.upsert([chunk], [[0.1] * 1536])
        api_count = api_store.count()
        if hasattr(api_store, "close"):
            api_store.close()

        cfg = SimpleNamespace(vector_store=vs_cfg.model_copy(deep=True))
        _isolate_local_collection(cfg, 384)
        local_store = build_vector_store(cfg.vector_store)
        assert local_store.count() == 0
        local_store.upsert([chunk], [[0.2] * 384])  # must not raise a dimension mismatch
        assert local_store.count() == 1
        if hasattr(local_store, "close"):
            local_store.close()

        reopened = build_vector_store(vs_cfg)
        assert reopened.count() == api_count
        if hasattr(reopened, "close"):
            reopened.close()


# ── Bug 1 + 2 + 3 end-to-end through eval.run_eval.main ───────────────────────

class TestRunEvalMainOffline:

    def _run_main(self, monkeypatch, tmp_path, argv, initial_count=0):
        import eval.run_eval as run_eval

        dataset = run_eval.load_dataset("eval/dataset.jsonl")
        by_question = {s["question"]: s for s in dataset}

        cfg = load_config("config/phase3.yaml").model_copy(deep=True)
        cfg.evaluation.results_path = str(tmp_path / "results.jsonl")
        cfg.evaluation.report_path = str(tmp_path / "report.json")
        cfg.evaluation.ci.sample_size = None
        cfg.embedding.provider = "openrouter"
        original_collection = cfg.vector_store.chroma.collection_name

        state = {"count": initial_count, "seeded": None, "cfg_seen": None}

        def fake_retrieve(question):
            s = by_question[question]
            sources = s.get("source_docs") or ["unknown.txt"]
            chunks = [SimpleNamespace(source=src, citation_id=f"{'a' * 8}:{i}")
                      for i, src in enumerate(sources)]
            ctx = "\n".join(f"[{c.citation_id}]  (source: {c.source}, score: 0.9)\n{s['ground_truth']}"
                            for c in chunks)
            return SimpleNamespace(chunks=chunks, context_block=ctx)

        vector_store = MagicMock()
        vector_store.count.side_effect = lambda: state["count"]

        def fake_ingest(docs):
            state["seeded"] = docs
            state["count"] = len(docs)
            return {"documents": len(docs)}

        ingestion = MagicMock()
        ingestion.vector_store = vector_store
        ingestion.ingest_documents.side_effect = fake_ingest
        bundle = SimpleNamespace(
            retriever=SimpleNamespace(vector_store=vector_store, retrieve=fake_retrieve),
            rag=None,
        )

        def fake_build(config_path, cfg=None):
            state["cfg_seen"] = cfg
            return ingestion, bundle

        monkeypatch.setattr(run_eval, "load_config", lambda path: cfg)
        monkeypatch.setattr(run_eval, "get_api_key", lambda: None)
        monkeypatch.setattr(run_eval, "build_ingestion_pipeline", fake_build)
        monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])

        with pytest.raises(SystemExit) as exc:
            run_eval.main()
        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) \
            if (tmp_path / "report.json").exists() else None
        return exc.value.code, state, report, original_collection

    def test_retrieval_only_offline_passes_ci_and_auto_seeds(self, monkeypatch, tmp_path):
        code, state, report, original = self._run_main(
            monkeypatch, tmp_path, ["--config", "config/phase3.yaml", "--retrieval-only"]
        )
        assert code == 0, report
        # Bug 3: empty store seeded with the baseline corpus.
        from main import SAMPLE_DOCS
        assert state["seeded"] == SAMPLE_DOCS
        # Bug 2: local fallback uses an isolated collection.
        assert state["cfg_seen"].embedding.provider == "local"
        assert state["cfg_seen"].vector_store.chroma.collection_name == f"{original}_local384"
        # Bug 1: only context_recall gates CI.
        assert report["threshold_breaches"] == []

    def test_no_auto_seed_fails_on_empty_store(self, monkeypatch, tmp_path):
        code, state, _, _ = self._run_main(
            monkeypatch, tmp_path,
            ["--config", "config/phase3.yaml", "--retrieval-only", "--no-auto-seed"],
        )
        assert code == 1
        assert state["seeded"] is None

    def test_existing_corpus_is_not_reseeded(self, monkeypatch, tmp_path):
        code, state, _, _ = self._run_main(
            monkeypatch, tmp_path, ["--config", "config/phase3.yaml", "--retrieval-only"],
            initial_count=5,
        )
        assert code == 0
        assert state["seeded"] is None

    def test_dry_run_still_fails_on_unmeasured_metrics(self, monkeypatch, tmp_path):
        code, _, report, _ = self._run_main(
            monkeypatch, tmp_path, ["--config", "config/phase3.yaml", "--dry-run"]
        )
        assert code == 1
        assert "faithfulness" in report["threshold_breaches"]


class TestCIWorkflow:

    def test_workflow_does_not_ingest_test_fixtures(self):
        text = open(".github/workflows/eval.yml", encoding="utf-8").read()
        assert "tests/fixtures" not in text
        assert "--retrieval-only" in text

    def test_job_level_if_does_not_reference_secrets(self):
        import yaml

        wf = yaml.safe_load(open(".github/workflows/eval.yml", encoding="utf-8"))
        for job in wf["jobs"].values():
            assert "secrets." not in str(job.get("if", ""))


# ── Bug 4 ─────────────────────────────────────────────────────────────────────

class TestMCPServerCLI:

    def _cfg(self, transport="stdio", port=8005):
        return SimpleNamespace(mcp=SimpleNamespace(server=SimpleNamespace(transport=transport, http_port=port)))

    def test_http_flag_forces_streamable_http_with_config_port(self):
        from mcp_server.__main__ import build_arg_parser, resolve_transport

        args = build_arg_parser().parse_args(["--config", "config/phase4.yaml", "--http"])
        assert resolve_transport(args, self._cfg()) == ("streamable-http", 8005)

    def test_port_override_implies_http(self):
        from mcp_server.__main__ import build_arg_parser, resolve_transport

        args = build_arg_parser().parse_args(["--port", "9100"])
        assert resolve_transport(args, self._cfg()) == ("streamable-http", 9100)

    def test_default_uses_config_transport(self):
        from mcp_server.__main__ import build_arg_parser, resolve_transport

        args = build_arg_parser().parse_args([])
        assert resolve_transport(args, self._cfg("stdio"))[0] == "stdio"
        assert resolve_transport(args, self._cfg("streamable-http"))[0] == "streamable-http"

    def test_invalid_port_and_conflicts_rejected(self):
        from mcp_server.__main__ import build_arg_parser, resolve_transport

        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            resolve_transport(parser.parse_args(["--port", "70000"]), self._cfg())
        with pytest.raises(SystemExit):
            resolve_transport(parser.parse_args(["--stdio", "--port", "9000"]), self._cfg())
        with pytest.raises(SystemExit):
            parser.parse_args(["--http", "--stdio"])

    def test_main_passes_port_to_server_run(self, monkeypatch):
        import mcp_server.__main__ as entry

        fake_server = MagicMock()
        fake_server._pipeline = None
        monkeypatch.setattr(entry, "create_server", lambda config_path: fake_server)
        entry.main(["--config", "config/phase4.yaml", "--http", "--port", "9123"])
        fake_server.run.assert_called_once_with(transport="streamable-http", port=9123)


class TestWeaviateConfigAliases:

    def test_index_name_alias_maps_to_class_name(self):
        assert WeaviateConfig(index_name="DocsIndex").class_name == "DocsIndex"
        assert WeaviateConfig(class_name="Other").class_name == "Other"

    def test_conflicting_aliases_rejected(self):
        with pytest.raises(ValueError):
            WeaviateConfig(class_name="A", index_name="B")

    def test_api_key_env_resolution(self, monkeypatch):
        monkeypatch.setenv("MY_WV_KEY", "secret-123")
        monkeypatch.delenv("WEAVIATE_API_KEY", raising=False)
        assert WeaviateConfig(api_key_env="MY_WV_KEY").resolve_api_key() == "secret-123"
        assert WeaviateConfig(api_key="literal", api_key_env="MY_WV_KEY").resolve_api_key() == "literal"
        assert WeaviateConfig().resolve_api_key() is None

    def test_readme_style_yaml_loads(self, tmp_path):
        base = open("config/phase1.yaml", encoding="utf-8").read()
        cfg_path = tmp_path / "wv.yaml"
        import yaml

        data = yaml.safe_load(base)
        data.setdefault("vector_store", {})["backend"] = "weaviate"
        data["vector_store"]["weaviate"] = {
            "url": "http://localhost:8080",
            "api_key_env": "WEAVIATE_API_KEY",
            "index_name": "RagChunk",
        }
        cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        cfg = load_config(str(cfg_path))
        assert cfg.vector_store.weaviate.class_name == "RagChunk"
        assert cfg.vector_store.weaviate.api_key_env == "WEAVIATE_API_KEY"


# ── Bug 5 ─────────────────────────────────────────────────────────────────────

class TestAgenticEnforcementOffLoop:

    def test_enforce_agentic_and_retry_run_off_event_loop_thread(self):
        from agent.orchestrator import AgenticRAGPipeline

        cfg = load_config("config/phase4.yaml")
        threads = {}

        retriever = MagicMock()
        retriever.enforcer = MagicMock()

        def fake_enforce(ans, chunks, web, retry_fn=None):
            threads["enforce"] = threading.get_ident()
            new_ans = retry_fn(ans)  # exercise the synchronous retry LLM call
            return SimpleNamespace(is_valid=True, answer=new_ans)

        retriever.enforcer.enforce_agentic.side_effect = fake_enforce

        mcp_manager = MagicMock()
        mcp_manager.connect_all = AsyncMock(return_value={})
        mcp_manager.close_all = AsyncMock()
        mcp_manager.get_tool_schemas.return_value = []

        pipeline = AgenticRAGPipeline(retriever, mcp_manager, cfg)

        final_msg = SimpleNamespace(tool_calls=None, content="Uncited answer.")
        retry_msg = SimpleNamespace(tool_calls=None, content="Cited answer [web:1].")
        responses = iter([
            SimpleNamespace(choices=[SimpleNamespace(message=final_msg)]),
            SimpleNamespace(choices=[SimpleNamespace(message=retry_msg)]),
        ])

        def fake_create(**kwargs):
            threads.setdefault("llm_calls", []).append(threading.get_ident())
            return next(responses)

        pipeline._llm = MagicMock()
        pipeline._llm.chat.completions.create.side_effect = fake_create

        async def _run():
            threads["loop"] = threading.get_ident()
            return await pipeline.aanswer("What is RAG?")

        ans = asyncio.run(_run())

        assert ans.answer == "Cited answer [web:1]."
        assert threads["enforce"] != threads["loop"]
        assert all(t != threads["loop"] for t in threads["llm_calls"])
        mcp_manager.close_all.assert_awaited_once()

