"""
config/loader.py
Loads versioned YAML config files and exposes them as typed Pydantic models.
Swap phases by changing the YAML path — all downstream code stays identical.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import yaml
from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

load_dotenv()

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openrouter/free"


def get_api_key() -> Optional[str]:
    """Return the configured OpenRouter API key (with fallback to OPENAI_API_KEY)."""
    return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")


def get_base_url(cfg: Optional["PipelineConfig"] = None) -> str:
    """Return the OpenRouter-compatible API base URL."""
    env_url = os.getenv("OPENROUTER_BASE_URL")
    if env_url:
        return env_url
    if cfg is not None and hasattr(cfg, "llm") and cfg.llm and cfg.llm.base_url:
        return cfg.llm.base_url
    return OPENROUTER_DEFAULT_BASE_URL


def get_default_model(cfg: Optional["PipelineConfig"] = None, fallback: str = OPENROUTER_DEFAULT_MODEL) -> str:
    """Return the default LLM model from env or config."""
    env_model = os.getenv("OPENROUTER_MODEL")
    if env_model:
        return env_model
    if cfg is not None and hasattr(cfg, "llm") and cfg.llm and cfg.llm.model:
        return cfg.llm.model
    return fallback


def create_llm_client(cfg: Optional["PipelineConfig"] = None) -> Any:
    """
    Create an OpenRouter-configured OpenAI SDK client.
    Returns None if neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set.
    """
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        base_url = get_base_url(cfg)
        headers: Dict[str, str] = {}
        app_name = getattr(getattr(cfg, "llm", None), "app_name", None) or "mcp-rag-pipeline"
        site_url = getattr(getattr(cfg, "llm", None), "site_url", None) or os.getenv("OPENROUTER_SITE_URL")
        if app_name:
            headers["X-Title"] = app_name
        if site_url:
            headers["HTTP-Referer"] = site_url

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers or None,
        )
    except Exception:
        return None


# ── Sub-models ────────────────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    provider: Literal["openrouter", "openai"] = "openrouter"
    base_url: str = OPENROUTER_DEFAULT_BASE_URL
    model: str = OPENROUTER_DEFAULT_MODEL
    app_name: Optional[str] = "mcp-rag-pipeline"
    site_url: Optional[str] = None


class ChunkingConfig(BaseModel):
    min_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 100
    tokenizer: str = "cl100k_base"

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ChunkingConfig":
        if self.min_tokens <= 0:
            raise ValueError("chunking.min_tokens must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("chunking.max_tokens must be > 0")
        if self.min_tokens > self.max_tokens:
            raise ValueError("chunking.min_tokens must be <= chunking.max_tokens")
        if self.overlap_tokens < 0:
            raise ValueError("chunking.overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("chunking.overlap_tokens must be strictly less than chunking.max_tokens")
        return self


class EmbeddingConfig(BaseModel):
    provider: Literal["openrouter", "openai", "local"] = "openrouter"
    base_url: str = OPENROUTER_DEFAULT_BASE_URL
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64

    @model_validator(mode="after")
    def _validate_embedding(self) -> "EmbeddingConfig":
        if self.batch_size <= 0:
            raise ValueError("embedding.batch_size must be > 0")
        if self.dimensions <= 0:
            raise ValueError("embedding.dimensions must be > 0")
        return self


class ChromaConfig(BaseModel):
    persist_directory: str = "./chroma_db"
    collection_name: str = "rag_phase1"


class WeaviateConfig(BaseModel):
    """
    Weaviate connection settings.

    Accepted YAML keys:
      - ``class_name`` or its alias ``index_name``
      - ``api_key`` (literal key; discouraged) or ``api_key_env`` (name of the
        environment variable holding the key, e.g. ``WEAVIATE_API_KEY``)
    """
    model_config = ConfigDict(populate_by_name=True)

    url: str = "http://localhost:8080"
    class_name: str = Field(
        default="RagChunk",
        validation_alias=AliasChoices("class_name", "index_name"),
    )
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    grpc_port: int = 50051

    @model_validator(mode="before")
    @classmethod
    def _reject_conflicting_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and "class_name" in data and "index_name" in data:
            if data["class_name"] != data["index_name"]:
                raise ValueError(
                    "vector_store.weaviate: 'class_name' and 'index_name' are aliases; "
                    "specify only one (or identical values)."
                )
        return data

    def resolve_api_key(self) -> Optional[str]:
        """Literal api_key wins, then the env var named by api_key_env, then WEAVIATE_API_KEY."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env) or None
        return os.getenv("WEAVIATE_API_KEY") or None


class VectorStoreConfig(BaseModel):
    backend: Literal["chroma", "weaviate"] = "chroma"
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)


class HybridConfig(BaseModel):
    enabled: bool = False
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    candidate_multiplier: int = 4

    @model_validator(mode="after")
    def _validate_hybrid(self) -> "HybridConfig":
        if not (0.0 <= self.vector_weight <= 1.0):
            raise ValueError("hybrid.vector_weight must be between 0.0 and 1.0")
        if not (0.0 <= self.bm25_weight <= 1.0):
            raise ValueError("hybrid.bm25_weight must be between 0.0 and 1.0")
        if (self.vector_weight + self.bm25_weight) <= 0.0:
            raise ValueError("hybrid.vector_weight + hybrid.bm25_weight must be > 0.0")
        if self.candidate_multiplier < 1:
            raise ValueError("hybrid.candidate_multiplier must be >= 1")
        return self


class RerankerConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_after_rerank: int = 5

    @model_validator(mode="after")
    def _validate_reranker(self) -> "RerankerConfig":
        if self.top_k_after_rerank < 1:
            raise ValueError("reranker.top_k_after_rerank must be >= 1")
        return self


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.0
    include_metadata: bool = True
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    @model_validator(mode="after")
    def _validate_retrieval(self) -> "RetrievalConfig":
        if self.top_k < 1:
            raise ValueError("retrieval.top_k must be >= 1")
        if not (0.0 <= self.score_threshold <= 1.0):
            raise ValueError("retrieval.score_threshold must be between 0.0 and 1.0")
        return self


class CitationEnforcementConfig(BaseModel):
    enabled: bool = False
    min_citations_required: int = 1
    citation_pattern: str = r"\[([a-f0-9]{8,64}:\d+)\]"
    on_violation: Literal["retry", "warn", "raise"] = "warn"
    max_retries: int = 2
    require_all_sentences_cited: bool = True
    verify_lexical_support: bool = True

    @model_validator(mode="after")
    def _validate_citations(self) -> "CitationEnforcementConfig":
        if self.min_citations_required < 0:
            raise ValueError("citation_enforcement.min_citations_required must be >= 0")
        if self.max_retries < 0:
            raise ValueError("citation_enforcement.max_retries must be >= 0")
        return self


# ── Phase 3: Evaluation ───────────────────────────────────────────────────────

class ScorerConfig(BaseModel):
    model: str = OPENROUTER_DEFAULT_MODEL
    batch_size: int = 10
    max_workers: int = 4

    @model_validator(mode="after")
    def _validate_scorer(self) -> "ScorerConfig":
        if self.batch_size < 1:
            raise ValueError("scorer.batch_size must be >= 1")
        if self.max_workers < 1:
            raise ValueError("scorer.max_workers must be >= 1")
        return self


class CIConfig(BaseModel):
    fail_on_threshold_breach: bool = True
    min_dataset_size: int = 20
    sample_size: Optional[int] = None


class ThresholdsConfig(BaseModel):
    faithfulness: float = 0.75
    answer_relevance: float = 0.70
    context_recall: float = 0.65
    citation_coverage: float = 0.80


class EvaluationConfig(BaseModel):
    dataset_path: str = "eval/dataset.jsonl"
    results_path: str = "eval/results.jsonl"
    report_path: str = "eval/report.json"
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    ci: CIConfig = Field(default_factory=CIConfig)


# ── Phase 4: Multi-format Ingestion, Agent & MCP ────────────────────────────

class ParserPDFConfig(BaseModel):
    ocr_fallback: bool = False


class ParserLatexConfig(BaseModel):
    keep_math: bool = False


class ParsersConfig(BaseModel):
    enabled_formats: list[str] = Field(
        default_factory=lambda: ["txt", "md", "pdf", "docx", "tex"]
    )
    pdf: ParserPDFConfig = Field(default_factory=ParserPDFConfig)
    latex: ParserLatexConfig = Field(default_factory=ParserLatexConfig)


class IngestionConfig(BaseModel):
    parsers: ParsersConfig = Field(default_factory=ParsersConfig)


class AgentConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["auto", "rag_only", "rag_then_fallback"] = "auto"
    max_tool_iterations: int = 4
    confidence_threshold: float = 0.75

    @model_validator(mode="after")
    def _validate_agent(self) -> "AgentConfig":
        if self.max_tool_iterations < 1:
            raise ValueError("agent.max_tool_iterations must be >= 1")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("agent.confidence_threshold must be between 0.0 and 1.0")
        return self


class MCPServerConnectionConfig(BaseModel):
    name: str
    enabled: bool = True
    transport: Literal["stdio", "streamable-http"] = "stdio"
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = None
    tool_allowlist: Optional[list[str]] = None


class MCPOwnServerConfig(BaseModel):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    http_port: int = 8765
    ingest_allowlist_root: str = "./ingest_data"


class MCPConfig(BaseModel):
    servers: list[MCPServerConnectionConfig] = Field(default_factory=list)
    server: MCPOwnServerConfig = Field(default_factory=MCPOwnServerConfig)

    @model_validator(mode="after")
    def _validate_unique_server_names(self) -> "MCPConfig":
        seen = set()
        for srv in self.servers:
            if srv.name in seen:
                raise ValueError(f"Duplicate MCP server name in configuration: '{srv.name}'")
            seen.add(srv.name)
        return self


DEFAULT_FAITHFULNESS_JUDGE = """You are an impartial evaluation judge. Your task is to assess whether an AI-generated answer is faithful to the provided context.
Context:
{context}

Answer to evaluate:
{answer}

Return ONLY valid JSON:
{{
  "claims": [{{"claim": "<claim text>", "supported": true, "evidence": "<quote or 'none'>"}}],
  "faithfulness_score": <float 0.0-1.0>,
  "reasoning": "<one sentence summary>"
}}"""

DEFAULT_RELEVANCE_JUDGE = """You are an impartial evaluation judge assessing answer relevance.
Question: {question}
Answer: {answer}

Return ONLY valid JSON:
{{
  "relevance_score": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}"""

DEFAULT_RECALL_JUDGE = """You are evaluating whether retrieved context covers the ground-truth answer.
GT: {ground_truth}
Context: {context}

Return ONLY valid JSON:
{{
  "recall_score": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}}"""


class PromptsConfig(BaseModel):
    rag_system: str
    rag_user: str
    rag_retry: Optional[str] = None
    agent_system: Optional[str] = None
    faithfulness_judge: str = DEFAULT_FAITHFULNESS_JUDGE
    answer_relevance_judge: str = DEFAULT_RELEVANCE_JUDGE
    context_recall_judge: str = DEFAULT_RECALL_JUDGE


class PipelineConfig(BaseModel):
    version: str
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    prompts: PromptsConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)
    citation_enforcement: CitationEnforcementConfig = Field(
        default_factory=CitationEnforcementConfig
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


# ── Loader (mtime-aware cache) ────────────────────────────────────────────────

_CONFIG_CACHE: Dict[str, Tuple[int, PipelineConfig]] = {}


def load_config(yaml_path: str = "config/phase1.yaml", reload: bool = False) -> PipelineConfig:
    """
    Load and validate config from a YAML file.
    Automatically reloads if the file's modification time changes on disk
    or when reload=True is passed.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    resolved_key = str(path.resolve())
    mtime_ns = path.stat().st_mtime_ns

    if not reload and resolved_key in _CONFIG_CACHE:
        cached_mtime, cached_cfg = _CONFIG_CACHE[resolved_key]
        if cached_mtime == mtime_ns:
            return cached_cfg.model_copy(deep=True)

    with path.open(encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    cfg = PipelineConfig(**raw)
    _CONFIG_CACHE[resolved_key] = (mtime_ns, cfg)
    return cfg.model_copy(deep=True)


def clear_config_cache() -> None:
    """Clear the configuration cache."""
    _CONFIG_CACHE.clear()


load_config.cache_clear = clear_config_cache  # type: ignore[attr-defined]
