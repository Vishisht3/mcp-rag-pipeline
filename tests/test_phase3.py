"""
tests/test_phase3.py
Unit tests for Phase 3:
  - Dataset validation (schema, duplicates, edge cases, generation deduplication)
  - FaithfulnessScorer (mocked LLM — no API calls)
  - Report builder (thresholds, breach detection, unmeasured metrics, CI gate)
  - Citation coverage scoring (verified against retrieved context)
  - Source-aware lexical recall
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_cfg():
    """Minimal PipelineConfig with Phase 3 eval settings."""
    class ScorerCfg(BaseModel):
        model: str = "openrouter/free"
        batch_size: int = 10
        max_workers: int = 2

    class CICfg(BaseModel):
        fail_on_threshold_breach: bool = True
        min_dataset_size: int = 5
        sample_size: Optional[int] = None

    class ThresholdsCfg(BaseModel):
        faithfulness: float = 0.75
        answer_relevance: float = 0.70
        context_recall: float = 0.65
        citation_coverage: float = 0.80

    class EvalCfg(BaseModel):
        dataset_path: str = "eval/dataset.jsonl"
        results_path: str = "eval/results.jsonl"
        report_path: str = "eval/report.json"
        thresholds: ThresholdsCfg = ThresholdsCfg()
        scorer: ScorerCfg = ScorerCfg()
        ci: CICfg = CICfg()

    cfg = MagicMock()
    cfg.evaluation = EvalCfg()
    cfg.prompts.faithfulness_judge   = "Context: {context}\nAnswer: {answer}"
    cfg.prompts.answer_relevance_judge = "Question: {question}\nAnswer: {answer}"
    cfg.prompts.context_recall_judge = "GT: {ground_truth}\nContext: {context}"
    return cfg


@pytest.fixture
def sample_scores():
    from eval.scorer import SampleScore
    return [
        SampleScore("q001", "Q1", "A1 [abcd1234:0]", "GT1",
                    faithfulness=0.9, answer_relevance=0.85,
                    context_recall=0.80, citation_coverage=1.0),
        SampleScore("q002", "Q2", "A2", "GT2",
                    faithfulness=0.6, answer_relevance=0.5,
                    context_recall=0.4, citation_coverage=0.0),
        SampleScore("q003", "Q3", "A3 [efgh5678:1]", "GT3",
                    faithfulness=0.8, answer_relevance=0.75,
                    context_recall=0.70, citation_coverage=1.0),
    ]


# ── Dataset validation ────────────────────────────────────────────────────────

class TestDatasetValidation:

    def _valid_record(self, id_="q001"):
        return {
            "id": id_,
            "question": "What is artificial intelligence?",
            "ground_truth": "AI is intelligence demonstrated by machines.",
            "source_docs": ["ai_overview.txt"],
            "difficulty": "easy",
            "category": "factual",
        }

    def test_valid_record_passes(self):
        from eval.build_dataset import validate_record
        errors = validate_record(self._valid_record(), 0)
        assert errors == []

    def test_missing_field_raises_error(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        del record["ground_truth"]
        errors = validate_record(record, 0)
        assert any("ground_truth" in e for e in errors)

    def test_invalid_difficulty(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        record["difficulty"] = "impossible"
        errors = validate_record(record, 0)
        assert any("difficulty" in e for e in errors)

    def test_invalid_category(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        record["category"] = "nonsense"
        errors = validate_record(record, 0)
        assert any("category" in e for e in errors)

    def test_question_too_short(self):
        from eval.build_dataset import validate_record
        record = self._valid_record()
        record["question"] = "AI?"
        errors = validate_record(record, 0)
        assert any("short" in e for e in errors)

    def test_all_60_seed_records_valid(self):
        from eval.build_dataset import SEED_DATASET, validate_record
        all_errors = []
        for i, record in enumerate(SEED_DATASET):
            all_errors.extend(validate_record(record, i))
        assert all_errors == [], f"Seed dataset errors: {all_errors}"

    def test_seed_dataset_has_no_duplicate_ids(self):
        from eval.build_dataset import SEED_DATASET
        ids = [r["id"] for r in SEED_DATASET]
        assert len(ids) == len(set(ids))

    def test_seed_dataset_size(self):
        from eval.build_dataset import SEED_DATASET
        assert len(SEED_DATASET) >= 50

    def test_seed_has_all_categories(self):
        from eval.build_dataset import SEED_DATASET
        cats = {r["category"] for r in SEED_DATASET}
        assert cats == {"factual", "reasoning", "multi-hop", "edge-case"}

    def test_seed_has_all_difficulties(self):
        from eval.build_dataset import SEED_DATASET
        diffs = {r["difficulty"] for r in SEED_DATASET}
        assert diffs == {"easy", "medium", "hard"}

    def test_cmd_generate_deduplicates_intra_batch_and_uses_corpus(self, tmp_path: Path):
        from eval.build_dataset import cmd_generate
        from ingestion.chunker import Chunk

        out_file = tmp_path / "dataset.jsonl"
        chunks = [
            Chunk(
                text="Custom corpus document about quantum computing qubits.",
                doc_id="12345678" * 8,
                chunk_index=0,
                source="quantum_notes.txt",
                token_count=8,
                start_token=0,
                end_token=8,
            )
        ]

        llm_lines = "\n".join([
            '{"id": "q001", "question": "What are quantum computing qubits used for?", "ground_truth": "They are used in quantum computing.", "source_docs": ["quantum_notes.txt"], "difficulty": "easy", "category": "factual"}',
            '{"id": "q001", "question": "What are quantum computing qubits used for?", "ground_truth": "Duplicate question in same batch.", "source_docs": ["quantum_notes.txt"], "difficulty": "easy", "category": "factual"}',
            '{"id": "q001", "question": "Which source discusses quantum computing qubits?", "ground_truth": "The quantum_notes.txt document.", "source_docs": ["quantum_notes.txt"], "difficulty": "medium", "category": "factual"}',
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=llm_lines))]
        )

        with patch("eval.build_dataset.create_llm_client", return_value=mock_client):
            cmd_generate(str(out_file), n=3, config_path="config/phase3.yaml", corpus_chunks=chunks)

        lines = [l for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2
        import json
        recs = [json.loads(l) for l in lines]
        assert recs[0]["id"] != recs[1]["id"]
        assert recs[0]["source_docs"] == ["quantum_notes.txt"]


# ── Citation coverage & lexical recall ───────────────────────────────────────

class TestCitationCoverage:

    def _scorer(self, cfg):
        from eval.scorer import FaithfulnessScorer
        scorer = FaithfulnessScorer.__new__(FaithfulnessScorer)
        scorer.cfg = cfg
        return scorer

    def test_answer_with_citation_scores_1(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("Answer here [abcd1234:0].") == 1.0

    def test_answer_without_citation_scores_0(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("No citations here.") == 0.0

    def test_multiple_citations_still_scores_1(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("[abcd1234:0] and [efgh5678:1]") == 1.0

    def test_malformed_citation_scores_0(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        assert scorer._score_citations("See [notvalid] for details.") == 0.0

    def test_fabricated_citation_not_in_context_scores_0(self, minimal_cfg):
        scorer = self._scorer(minimal_cfg)
        context = "[abcd1234:0]  (source: doc.txt, score: 0.9000)\nReal retrieved context."
        assert scorer._score_citations("Answer with fake citation [deadbeef:9].", context=context) == 0.0

    def test_lexical_recall_verifies_expected_source_docs(self):
        from eval.run_eval import _lexical_recall

        gt = "ChromaDB persists data to disk automatically."
        context_wrong_source = "[abcd1234:0]  (source: wrong_file.txt, score: 0.9)\nChromaDB persists data to disk automatically."
        context_right_source = "[abcd1234:0]  (source: vector_databases.txt, score: 0.9)\nChromaDB persists data to disk automatically."

        assert _lexical_recall(gt, context_wrong_source, source_docs=["vector_databases.txt"]) == 0.0
        assert _lexical_recall(gt, context_right_source, source_docs=["vector_databases.txt"]) == 1.0


# ── Scorer (mocked LLM) ───────────────────────────────────────────────────────

class TestFaithfulnessScorer:

    def _build_scorer(self, minimal_cfg):
        from eval.scorer import FaithfulnessScorer
        scorer = FaithfulnessScorer.__new__(FaithfulnessScorer)
        scorer.cfg = minimal_cfg

        mock_client = MagicMock()
        scorer._client = mock_client

        def fake_judge(prompt: str) -> dict:
            if "GT:" in prompt:
                return {"recall_score": 0.75, "reasoning": "Most ground truth is covered."}
            elif "Question:" in prompt:
                return {"relevance_score": 0.85, "reasoning": "Directly answers the question."}
            else:
                return {
                    "faithfulness_score": 0.9,
                    "claims": [{"claim": "AI is intelligent", "supported": True, "evidence": "..."}],
                    "reasoning": "All claims supported.",
                }

        scorer._call_judge = fake_judge
        return scorer

    def test_score_sample_returns_all_metrics(self, minimal_cfg):
        scorer = self._build_scorer(minimal_cfg)
        result = scorer.score_sample(
            "q001", "What is AI?", "AI is intelligent machines [abcd1234:0].",
            "AI context here.", "AI is intelligence by machines."
        )
        assert result.faithfulness     > 0
        assert result.answer_relevance > 0
        assert result.context_recall   > 0
        assert result.citation_coverage == 1.0
        assert result.error is None

    def test_score_sample_no_citation(self, minimal_cfg):
        scorer = self._build_scorer(minimal_cfg)
        result = scorer.score_sample(
            "q002", "What is AI?", "AI is intelligent machines.",
            "context", "ground truth"
        )
        assert result.citation_coverage == 0.0

    def test_score_sample_handles_error_gracefully(self, minimal_cfg):
        scorer = self._build_scorer(minimal_cfg)
        scorer._call_judge = MagicMock(side_effect=Exception("API error"))
        result = scorer.score_sample("q003", "Q?", "A.", "ctx", "gt")
        assert result.error is not None
        assert result.faithfulness == 0.0


# ── Report builder ────────────────────────────────────────────────────────────

class TestReportBuilder:

    def test_report_averages_correct(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.75, "answer_relevance": 0.70,
                      "context_recall": 0.65, "citation_coverage": 0.80}
        report = build_report(sample_scores, thresholds)
        assert abs(report.avg_faithfulness - (0.9 + 0.6 + 0.8) / 3) < 1e-6
        assert report.total == 3

    def test_threshold_breach_detected(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.99, "answer_relevance": 0.99,
                      "context_recall": 0.99, "citation_coverage": 0.99}
        report = build_report(sample_scores, thresholds)
        assert len(report.threshold_breaches) > 0
        assert not report.ci_passed

    def test_unmeasured_metrics_fail_ci_gate(self):
        from eval.scorer import SampleScore, build_report
        unmeasured = [
            SampleScore(
                "q001", "Q1", "", "GT1",
                faithfulness=None,
                answer_relevance=None,
                context_recall=0.85,
                citation_coverage=None,
            )
        ]
        thresholds = {
            "faithfulness": 0.75,
            "answer_relevance": 0.70,
            "context_recall": 0.65,
            "citation_coverage": 0.80,
        }
        report = build_report(unmeasured, thresholds)
        assert not report.ci_passed
        assert "faithfulness" in report.threshold_breaches
        assert "answer_relevance" in report.threshold_breaches
        assert "citation_coverage" in report.threshold_breaches
        assert report.passed == 0

    def test_all_pass_when_thresholds_low(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.0, "answer_relevance": 0.0,
                      "context_recall": 0.0, "citation_coverage": 0.0}
        report = build_report(sample_scores, thresholds)
        assert report.ci_passed

    def test_empty_scores_raises(self):
        from eval.scorer import build_report
        with pytest.raises(ValueError):
            build_report([], {})

    def test_passed_count(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.75, "answer_relevance": 0.70,
                      "context_recall": 0.65, "citation_coverage": 0.80}
        report = build_report(sample_scores, thresholds)
        assert report.passed == 2
        assert report.failed == 1

    def test_report_to_dict_includes_ci_passed(self, sample_scores):
        from eval.scorer import build_report
        thresholds = {"faithfulness": 0.0, "answer_relevance": 0.0,
                      "context_recall": 0.0, "citation_coverage": 0.0}
        report = build_report(sample_scores, thresholds)
        d = report.to_dict()
        assert "ci_passed" in d
        assert d["ci_passed"] is True
