"""
eval/run_eval.py
────────────────────────────────────────────────────────────────────────────────
Offline evaluation script. Wired to CI via .github/workflows/eval.yml.

Flow:
  1. Load eval dataset from eval/dataset.jsonl
  2. For each QA pair, run the RAG pipeline to get an answer + context
  3. Score each answer with FaithfulnessScorer (faithfulness, relevance,
     recall, citation coverage, and optional tool selection)
  4. Write per-sample results to eval/results.jsonl
  5. Write aggregate report to eval/report.json
  6. Exit 0 if all thresholds pass, Exit 1 if any breach or unmeasured metric (CI gate)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Add project root to path so imports resolve regardless of cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.loader import get_api_key, load_config
from eval.scorer import FaithfulnessScorer, SampleScore, build_report
from ingestion.pipeline import build_ingestion_pipeline

SOURCE_HEADER_PATTERN = re.compile(r"\(source:\s*([^,\)]+)")


# ── Helpers ───────────────────────────────────────────────────────────────────

RETRIEVAL_ONLY_METRICS = ("context_recall",)


def select_active_thresholds(thresholds, retrieval_scope: bool) -> dict:
    """
    Return the thresholds that should gate CI for this run.

    In retrieval-only scope (``--retrieval-only`` or no API key outside ``--dry-run``)
    only retrieval metrics are measurable, so only they gate CI. Full runs and
    ``--dry-run`` gate on every configured metric (dry-run therefore still fails on
    unmeasured metrics, which is the intended behaviour).
    """
    all_thresholds = (
        thresholds.model_dump() if hasattr(thresholds, "model_dump") else dict(thresholds)
    )
    if retrieval_scope:
        return {k: all_thresholds[k] for k in RETRIEVAL_ONLY_METRICS if k in all_thresholds}
    return all_thresholds


def _isolate_local_collection(cfg, dims: int) -> None:
    """
    Suffix the vector-store collection/class name with the local embedding dimension
    so a local-embedding run never reuses a collection built with a different
    embedding model (e.g. 1536-d OpenRouter/OpenAI vectors already on disk).
    Idempotent: calling it twice does not double-suffix.
    """
    chroma_suffix = f"_local{dims}"
    chroma_cfg = cfg.vector_store.chroma
    if not chroma_cfg.collection_name.endswith(chroma_suffix):
        chroma_cfg.collection_name += chroma_suffix

    # Weaviate class names must be PascalCase alphanumerics.
    weaviate_suffix = f"Local{dims}"
    wv_cfg = cfg.vector_store.weaviate
    if not wv_cfg.class_name.endswith(weaviate_suffix):
        wv_cfg.class_name += weaviate_suffix

    active = (
        chroma_cfg.collection_name
        if cfg.vector_store.backend == "chroma"
        else wv_cfg.class_name
    )
    print(f"  Using isolated local-embedding collection: {active}")


def load_dataset(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_rag_for_sample(
    sample: dict,
    retriever,
) -> Tuple[str, str, List[str], List[str]]:
    """
    Run the retrieval pipeline for a single QA pair.
    Returns (answer_text, context_block, retrieved_sources, valid_citations).
    """
    if retriever is None:
        return "", "", [], []

    result = retriever.retrieve(sample["question"])
    retrieved_sources = [c.source for c in result.chunks]
    valid_citations = [c.citation_id for c in result.chunks]
    return "", result.context_block, retrieved_sources, valid_citations


def run_full_rag(
    sample: dict,
    rag,
    agentic_pipeline=None,
) -> Tuple[str, str, List[str], List[str], List[str]]:
    """Run retrieve + generate. Returns (answer, context, actual_tools, retrieved_sources, valid_citations)."""
    if agentic_pipeline is not None:
        ans_obj = agentic_pipeline.answer(sample["question"])
        tools_used = [step["tool"] for step in ans_obj.tool_trace]
        ctx = "\n\n".join(
            f"[{c.citation_id}]  (source: {c.source}, score: {c.score:.4f})\n{c.text}"
            for c in ans_obj.retrieved_chunks
        )
        retrieved_sources = [c.source for c in ans_obj.retrieved_chunks]
        valid_cites = [c.citation_id for c in ans_obj.retrieved_chunks] + [
            f"web:{i+1}" for i in range(len(ans_obj.web_sources))
        ]
        return ans_obj.answer, ctx, tools_used, retrieved_sources, valid_cites

    if rag is None:
        return "", "", [], [], []

    answer_obj = rag.answer(sample["question"])
    retrieved_sources = [c.source for c in answer_obj.retrieved_chunks]
    valid_cites = [c.citation_id for c in answer_obj.retrieved_chunks]
    return (
        answer_obj.answer,
        answer_obj.context_block,
        ["search_local_documents"],
        retrieved_sources,
        valid_cites,
    )


def _lexical_recall(
    ground_truth: str,
    context: str,
    source_docs: Optional[List[str]] = None,
    retrieved_sources: Optional[List[str]] = None,
) -> float:
    """
    Deterministic lexical overlap + source attribution recall for retrieval evaluation.
    Verifies both token coverage of ground_truth and that expected source_docs were retrieved.
    """
    if source_docs is not None and len(source_docs) == 0:
        return 1.0
    if not context.strip():
        return 0.0

    # Check source document attribution when expected source_docs are specified
    source_factor = 1.0
    if source_docs:
        actual_sources = set()
        if retrieved_sources is not None:
            actual_sources.update(s.strip() for s in retrieved_sources if s)
        header_sources = SOURCE_HEADER_PATTERN.findall(context)
        actual_sources.update(s.strip() for s in header_sources if s)

        if actual_sources or retrieved_sources is not None:
            matched_docs = [
                exp_doc for exp_doc in source_docs
                if exp_doc in actual_sources
                or any(exp_doc in actual or actual in exp_doc for actual in actual_sources)
            ]
            source_factor = len(matched_docs) / len(source_docs)
            if source_factor == 0.0:
                return 0.0

    trans = str.maketrans("", "", string.punctuation)
    gt_tokens = {w for w in ground_truth.lower().translate(trans).split() if len(w) > 2}
    if not gt_tokens:
        return round(1.0 * source_factor, 4)
    ctx_tokens = set(context.lower().translate(trans).split())
    token_recall = len(gt_tokens & ctx_tokens) / len(gt_tokens)
    return round(token_recall * source_factor, 4)


# ── Main eval loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Script — Phase 3 & 4")
    parser.add_argument("--config",   default="config/phase3.yaml")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Random sample size (default: full dataset)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Skip RAG + LLM calls; metrics will be marked unmeasured (None)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Score retrieved context recall without LLM generation")
    parser.add_argument("--agentic",  action="store_true",
                        help="Evaluate using AgenticRAGPipeline")
    parser.add_argument("--ingest-dir", default=None,
                        help="Optional directory of documents to ingest before running evaluation")
    parser.add_argument("--no-auto-seed", action="store_true",
                        help="Fail instead of seeding the baseline SAMPLE_DOCS corpus when the "
                             "vector store is empty (use when evaluating your own corpus)")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    eval_cfg = cfg.evaluation

    has_api_key = bool(get_api_key())
    # Retrieval-only scoring applies when explicitly requested, or implicitly when no
    # API key is available (generation/LLM-judge metrics cannot be measured).
    # --dry-run is deliberately excluded so it keeps exercising unmeasured-metric gating.
    retrieval_scope = args.retrieval_only or (not has_api_key and not args.dry_run)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    dataset_path = eval_cfg.dataset_path
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run:  python eval/build_dataset.py --mode seed")
        sys.exit(1)

    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} QA pairs from {dataset_path}")

    # ── 2. Enforce minimum dataset size ───────────────────────────────────────
    min_size = eval_cfg.ci.min_dataset_size
    if len(dataset) < min_size:
        print(f"ERROR: Dataset has {len(dataset)} records; minimum is {min_size}")
        sys.exit(1)

    # ── 3. Optional random sample ─────────────────────────────────────────────
    sample_size = args.sample or eval_cfg.ci.sample_size
    if sample_size and sample_size < len(dataset):
        dataset = random.sample(dataset, sample_size)
        print(f"Sampled {sample_size} records for evaluation")

    # ── 4. Build pipeline ─────────────────────────────────────────────────────
    agentic_pipeline = None
    ingestion = None
    bundle = None

    try:
        if args.dry_run:
            print("DRY RUN — skipping RAG pipeline initialisation (metrics will be unmeasured)")
            retriever = None
            rag       = None
        else:
            if not has_api_key and cfg.embedding.provider in ("openrouter", "openai"):
                local_dims = 384
                print(f"No OPENROUTER_API_KEY / OPENAI_API_KEY detected — using local embeddings ({local_dims}-d)")
                cfg.embedding.provider = "local"
                cfg.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
                cfg.embedding.dimensions = local_dims
                # Isolate the local embedding space so it never collides with an existing
                # API-embedded (e.g. 1536-d) collection persisted under the configured name.
                _isolate_local_collection(cfg, local_dims)

            print(f"Building pipeline from {args.config}...")
            ingestion, bundle = build_ingestion_pipeline(args.config, cfg=cfg)

            if args.ingest_dir:
                print(f"Ingesting corpus from {args.ingest_dir}...")
                stats = ingestion.ingest_directory(args.ingest_dir)
                print(f"  {stats}")

            if bundle.retriever.vector_store.count() == 0:
                if args.ingest_dir or args.no_auto_seed:
                    print(
                        "ERROR: Vector store collection is empty. "
                        "Ingest your corpus before running evaluation or pass --ingest-dir <path>.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print("Vector store is empty — ingesting baseline SAMPLE_DOCS (the corpus "
                      "eval/dataset.jsonl was written against)...")
                from main import SAMPLE_DOCS
                stats = ingestion.ingest_documents(SAMPLE_DOCS)
                print(f"  {stats}")

            retriever = bundle.retriever
            rag       = bundle.rag if (not retrieval_scope and has_api_key) else None

            use_agentic = args.agentic or cfg.agent.enabled
            if use_agentic and not retrieval_scope and has_api_key:
                from agent.mcp_client import MCPClientManager
                from agent.orchestrator import AgenticRAGPipeline
                mcp_manager = MCPClientManager(cfg.mcp.servers)
                agentic_pipeline = AgenticRAGPipeline(retriever, mcp_manager, cfg)

        # ── 5. Run RAG for each sample ────────────────────────────────────────
        print(f"\nRunning RAG pipeline on {len(dataset)} samples...")
        scored_inputs = []

        for i, sample in enumerate(dataset):
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(dataset)}...")

            actual_tools: List[str] = []
            retrieved_sources: List[str] = []
            valid_citations: List[str] = []
            try:
                if args.dry_run or retrieval_scope or rag is None:
                    answer, context, retrieved_sources, valid_citations = run_rag_for_sample(
                        sample, retriever
                    )
                    actual_tools = []
                else:
                    (
                        answer,
                        context,
                        actual_tools,
                        retrieved_sources,
                        valid_citations,
                    ) = run_full_rag(sample, rag, agentic_pipeline=agentic_pipeline)
            except Exception as e:
                answer  = ""
                context = f"ERROR: {e}"

            scored_inputs.append({
                "id":                sample["id"],
                "question":          sample["question"],
                "answer":            answer,
                "context":           context,
                "ground_truth":      sample["ground_truth"],
                "source_docs":       sample.get("source_docs", []),
                "expected_source":   sample.get("expected_source"),
                "actual_tools":      actual_tools,
                "retrieved_sources": retrieved_sources,
                "valid_citations":   valid_citations,
            })

        # ── 6. Score ──────────────────────────────────────────────────────────
        print(f"\nScoring {len(scored_inputs)} answers...")
        if args.dry_run or retrieval_scope:
            scores: List[SampleScore] = []
            for s in scored_inputs:
                recall_val: Optional[float] = None if args.dry_run else _lexical_recall(
                    s["ground_truth"],
                    s["context"],
                    s.get("source_docs"),
                    s.get("retrieved_sources"),
                )
                scores.append(
                    SampleScore(
                        id=s["id"],
                        question=s["question"],
                        answer=s["answer"],
                        ground_truth=s["ground_truth"],
                        faithfulness=None,
                        answer_relevance=None,
                        context_recall=recall_val,
                        citation_coverage=None,
                        tool_selection_accuracy=None,
                        expected_source=s.get("expected_source"),
                        actual_tools=s.get("actual_tools", []),
                    )
                )
        else:
            scorer = FaithfulnessScorer(cfg)
            scores = scorer.score_batch(
                scored_inputs,
                max_workers=eval_cfg.scorer.max_workers,
            )

        # ── 7. Write per-sample results ───────────────────────────────────────
        results_path = Path(eval_cfg.results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with results_path.open("w", encoding="utf-8") as f:
            for score in scores:
                f.write(json.dumps(asdict(score)) + "\n")

        print(f"Results written to {results_path}")

        # ── 8. Build and write aggregate report ───────────────────────────────
        # Retrieval-only runs gate strictly on context_recall; generation metrics are
        # reported as unmeasured (None) but are not CI thresholds in that mode.
        active_thresholds = select_active_thresholds(eval_cfg.thresholds, retrieval_scope)
        report = build_report(scores, active_thresholds)
        report.print_summary()

        report_path = Path(eval_cfg.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        print(f"Report written to {report_path}")

        # ── 9. CI gate ────────────────────────────────────────────────────────
        if eval_cfg.ci.fail_on_threshold_breach and not report.ci_passed:
            print(f"\n❌ CI FAILED — thresholds breached / unmeasured: {report.threshold_breaches}")
            sys.exit(1)

        print("✅ CI PASSED")
        sys.exit(0)
    finally:
        if ingestion is not None and hasattr(ingestion.vector_store, "close"):
            ingestion.vector_store.close()


if __name__ == "__main__":
    main()
