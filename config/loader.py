"""
config/loader.py
Loads versioned YAML config files and exposes them as typed Pydantic models.
Swap phases by changing the YAML path — all downstream code stays identical.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ── Sub-models ────────────────────────────────────────────────────────────────

class ChunkingConfig(BaseModel):
    min_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 100
    tokenizer: str = "cl100k_base"


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "local"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64


class ChromaConfig(BaseModel):
    persist_directory: str = "./chroma_db"
    collection_name: str = "rag_phase1"


class WeaviateConfig(BaseModel):
    url: str = "http://localhost:8080"
    class_name: str = "RagChunk"


class VectorStoreConfig(BaseModel):
    backend: Literal["chroma", "weaviate"] = "chroma"
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)


class HybridConfig(BaseModel):
    enabled: bool = False
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    candidate_multiplier: int = 4


class RerankerConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_after_rerank: int = 5


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.0
    include_metadata: bool = True
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class CitationEnforcementConfig(BaseModel):
    enabled: bool = False
    min_citations_required: int = 1
    citation_pattern: str = r"\[([a-f0-9]{8}:\d+)\]"
    on_violation: Literal["retry", "warn", "raise"] = "warn"
    max_retries: int = 2


# ── Phase 3: Evaluation ───────────────────────────────────────────────────────

class ScorerConfig(BaseModel):
    model: str = "gpt-4o-mini"
    batch_size: int = 10
    max_workers: int = 4


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


class PromptsConfig(BaseModel):
    rag_system: str
    rag_user: str
    rag_retry: Optional[str] = None
    faithfulness_judge: str = "Context: {context}\nAnswer: {answer}"
    answer_relevance_judge: str = "Question: {question}\nAnswer: {answer}"
    context_recall_judge: str = "GT: {ground_truth}\nContext: {context}"


class PipelineConfig(BaseModel):
    version: str
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    prompts: PromptsConfig
    citation_enforcement: CitationEnforcementConfig = Field(
        default_factory=CitationEnforcementConfig
    )
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def load_config(yaml_path: str = "config/phase1.yaml") -> PipelineConfig:
    """
    Load and validate config from a YAML file.
    Results are cached — safe to call repeatedly without re-reading disk.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open() as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    return PipelineConfig(**raw)