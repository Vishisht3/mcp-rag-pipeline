# MCP-Powered Agentic RAG Pipeline

A modular Retrieval-Augmented Generation (RAG) and MCP v2 agentic system with multi-format document ingestion, hybrid BM25 + dense vector search, cross-encoder reranking, sentence-level citation enforcement, external MCP server delegation, and automated faithfulness evaluation.

Configured out of the box for **OpenRouter** (`https://openrouter.ai/api/v1`) so you can run generation and LLM-as-judge evaluation with any free-tier OpenRouter model (defaulting to `openrouter/free`), alongside local `sentence-transformers` or OpenRouter embeddings.

## Quick Start

```bash
git clone https://github.com/Vishisht3/mcp-rag-pipeline.git
cd mcp-rag-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Configure OpenRouter (supports any free-tier model, defaults to openrouter/free)
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env
# Optional: override the default free model
echo "OPENROUTER_MODEL=openrouter/free" >> .env

python main.py                                    # Full RAG demo (uses OPENROUTER_API_KEY)
python main.py --retrieval-only                   # Retrieval only, no API key required
python main.py --config config/phase4.yaml        # Phase 4: Agentic RAG with MCP tools
python -m mcp_server --config config/phase4.yaml  # Standalone MCP v2 server (stdio or HTTP)
```

## Architecture

```
mcp-rag-pipeline/
├── config/
│   ├── loader.py              # Typed Pydantic config + OpenRouter client factory
│   ├── phase1.yaml            # Phase 1: Core RAG
│   ├── phase2.yaml            # Phase 2: Hybrid search + cross-encoder reranking
│   ├── phase3.yaml            # Phase 3: Evaluation + CI quality gates
│   └── phase4.yaml            # Phase 4: Multi-format ingestion + Agentic MCP RAG
├── ingestion/
│   ├── chunker.py             # Source-aware sliding-window token chunker
│   ├── embedder.py            # OpenRouter / OpenAI-compatible & local embeddings
│   ├── pipeline.py            # Idempotent chunk -> validate -> embed -> store pipeline
│   └── parsers/               # Multi-format parsers (.txt, .md, .pdf, .docx, .tex)
├── store/
│   └── vector_store.py        # ChromaDB & Weaviate (local + Cloud) vector backends
├── retrieval/
│   ├── retriever.py           # Semantic top-k retrieval + disambiguated citations
│   ├── hybrid_retriever.py    # Collision-safe BM25 + vector score fusion
│   ├── reranker.py            # Cross-encoder reranking with score preservation
│   ├── bm25_index.py          # Saturation-normalised Okapi BM25 keyword index
│   └── citation_enforcer.py   # Sentence-level & lexical-support citation validation
├── agent/
│   ├── orchestrator.py        # ReAct agent loop over local RAG + external MCP tools
│   ├── tools.py               # Local retrieval tool with calibrated confidence
│   └── mcp_client.py          # MCP SDK v2 client manager (stdio & streamable-http)
├── mcp_server/
│   ├── server.py              # Standalone MCP SDK v2 server (MCPServer)
│   └── __main__.py            # CLI entrypoint for stdio and streamable-http
├── eval/
│   ├── build_dataset.py       # Golden QA dataset generator, seeder, and validator
│   ├── run_eval.py            # Evaluation runner with strict CI threshold gating
│   ├── scorer.py              # LLM-as-judge faithfulness & citation coverage scorer
│   └── dataset.jsonl          # 60-pair golden evaluation dataset
├── tests/
│   ├── test_phase1.py         # Chunker, vector store integrity, retriever tests
│   ├── test_phase2.py         # BM25, hybrid fusion, reranker, citation enforcer tests
│   ├── test_phase3.py         # Dataset validation, scorer, and CI gating tests
│   ├── test_phase4.py         # Parsers, MCP v2 client/server, and orchestrator tests
│   └── test_regressions.py    # Eval gating, collection isolation, CLI, async-safety regressions
├── .github/workflows/
│   └── eval.yml               # GitHub Actions unit test + retrieval evaluation gate
├── main.py                    # End-to-end CLI demo
└── requirements.txt
```

## Phases

Each phase is driven by its YAML configuration file:

### Phase 1 — Core RAG

```bash
python main.py --config config/phase1.yaml
```

- Source-aware sliding-window token chunking (500–800 tokens, 100-token overlap)
- Protected chunk metadata sanitisation and idempotent re-ingestion (`delete_by_source`)
- OpenRouter (`openai/text-embedding-3-small`) or local (`all-MiniLM-L6-v2`) embeddings with dimension tracking
- ChromaDB or Weaviate vector store with cosine similarity search
- Inline citation formatting (`[doc_id:chunk_index]`)

### Phase 2 — Hybrid Search and Reranking

```bash
python main.py --config config/phase2.yaml
```

- Hybrid retrieval: saturation-normalised BM25 keyword scores fused with vector similarity (`vector_weight` / `bm25_weight`)
- Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Post-generation citation enforcement: validates citation syntax, retrieved-context membership, multi-sentence citation coverage, and lexical claim support

### Phase 3 — Evaluation and CI

```bash
python eval/run_eval.py --config config/phase3.yaml
```

- LLM-as-judge scoring via OpenRouter (`openrouter/free` by default) across four metrics
- Strict CI threshold gating (unmeasured metrics in `--retrieval-only` or `--dry-run` mode are reported as `null` rather than fabricated scores)
- GitHub Actions workflow (`.github/workflows/eval.yml`)

### Phase 4 — Multi-Format Ingestion, Agentic RAG, and MCP v2 Server

```bash
# Run agentic RAG pipeline
python main.py --config config/phase4.yaml

# Run standalone MCP v2 server (stdio transport)
python -m mcp_server --config config/phase4.yaml

# Run standalone MCP v2 server (streamable-http on mcp.server.http_port, default 8005)
python -m mcp_server --config config/phase4.yaml --http

# Override the HTTP port (implies --http)
python -m mcp_server --config config/phase4.yaml --port 9000
```

- Multi-format document ingestion (`.txt`, `.md`, `.pdf` with optional OCR fallback, `.docx`, `.tex`)
- `MCPClientManager` built on MCP SDK v2 (`mcp>=2.3.0`) with `2026-07-28` protocol auto-negotiation, `streamable_http_client`, `tool.input_schema`, `res.is_error`, and `res.structured_content`
- `AgenticRAGPipeline` combines local hybrid retrieval (`search_local_documents`) with external MCP servers (such as Tavily `tavily_search` / `tavily_extract`)
- Standalone MCP server (`MCPServer`) exposing `rag_query`, `ingest_document`, `list_ingested_sources`, and `delete_ingested_source`

## Evaluation Metrics

| Metric | What it measures | Default threshold |
|--------|-----------------|-------------------|
| Faithfulness | Are all answer claims supported by the retrieved context? | 0.75 |
| Answer Relevance | Does the answer directly address the question? | 0.70 |
| Context Recall | Does retrieved context cover the ground truth from expected sources? | 0.65 |
| Citation Coverage | Do substantive answer sentences cite valid retrieved chunks? | 0.80 |

```bash
# Seed or validate the 60-pair evaluation dataset
python eval/build_dataset.py --mode seed --output eval/dataset.jsonl
python eval/build_dataset.py --mode validate --output eval/dataset.jsonl

# Full evaluation (requires OPENROUTER_API_KEY)
python eval/run_eval.py --config config/phase3.yaml

# Retrieval-only evaluation (no LLM generation cost; gates CI on context_recall only)
python eval/run_eval.py --config config/phase3.yaml --retrieval-only

# Evaluate against your own corpus (fail instead of auto-seeding if nothing is ingested)
python eval/run_eval.py --config config/phase3.yaml --ingest-dir path/to/docs --no-auto-seed
```

Notes:
- On a fresh clone the vector store is empty, so `run_eval.py` seeds the baseline `SAMPLE_DOCS` corpus from `main.py` — the documents `eval/dataset.jsonl` was written against. Pass `--ingest-dir` or `--no-auto-seed` to disable this.
- Without an API key, `run_eval.py` falls back to local 384-d embeddings in an isolated collection (e.g. `rag_phase3_local384`), so it never clashes with an existing 1536-d OpenRouter/OpenAI collection.
- `--retrieval-only` (and key-less runs) report generation metrics as `null` and gate CI on `context_recall` only. `--dry-run` still fails on every unmeasured metric.

## Switching Vector Stores

Swap from ChromaDB to local Weaviate or Weaviate Cloud in any YAML config:

```yaml
vector_store:
  backend: "weaviate"
  weaviate:
    url: "http://localhost:8080"          # or "https://<cluster>.weaviate.cloud"
    api_key_env: "WEAVIATE_API_KEY"       # optional for local, required for Weaviate Cloud
    index_name: "RagChunk"                # alias of class_name
```

`api_key_env` names the environment variable holding the key (a literal `api_key` also works but is discouraged). If neither is set, `WEAVIATE_API_KEY` is used when present.

## Running Tests

```bash
pytest tests/ -v
```

111 tests pass across all 4 phases plus `tests/test_regressions.py`.
