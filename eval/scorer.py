"""
eval/scorer.py
LLM-as-judge faithfulness scorer (OpenRouter compatible).

Measures four metrics per QA pair:
  1. faithfulness       — are all answer claims supported by the context?
  2. answer_relevance   — does the answer address the question?
  3. context_recall     — does the context cover the ground truth?
  4. citation_coverage  — does the answer contain valid, verified inline citations?

All judge prompts live in config/phase3.yaml — never hardcoded here.
Uses concurrent.futures for batched parallel scoring.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Set

from config.loader import PipelineConfig, create_llm_client, get_default_model
from retrieval.citation_enforcer import _split_factual_sentences


def _parse_json_response(raw_text: str) -> dict:
    """Parse JSON from LLM output, stripping markdown code fences if present."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ── Per-sample result ─────────────────────────────────────────────────────────

@dataclass
class SampleScore:
    id: str
    question: str
    answer: str
    ground_truth: str
    faithfulness: Optional[float]      = 0.0
    answer_relevance: Optional[float]  = 0.0
    context_recall: Optional[float]    = 0.0
    citation_coverage: Optional[float] = 0.0
    tool_selection_accuracy: Optional[float] = None
    expected_source: Optional[str] = None
    actual_tools: list          = field(default_factory=list)
    faithfulness_claims: list   = field(default_factory=list)
    faithfulness_reasoning: str = ""
    relevance_reasoning: str    = ""
    recall_reasoning: str       = ""
    error: Optional[str]        = None

    def passed(self, thresholds: dict) -> bool:
        for metric_name, attr_val in (
            ("faithfulness", self.faithfulness),
            ("answer_relevance", self.answer_relevance),
            ("context_recall", self.context_recall),
            ("citation_coverage", self.citation_coverage),
        ):
            req = thresholds.get(metric_name, 0.0)
            if attr_val is None:
                if req > 0.0 or metric_name in thresholds:
                    return False
            elif attr_val < req:
                return False

        if "tool_selection" in thresholds:
            if self.tool_selection_accuracy is None:
                if self.expected_source:
                    return False
            elif self.tool_selection_accuracy < thresholds["tool_selection"]:
                return False

        return True


# ── Aggregate report ──────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    avg_faithfulness: Optional[float]
    avg_answer_relevance: Optional[float]
    avg_context_recall: Optional[float]
    avg_citation_coverage: Optional[float]
    avg_tool_selection: Optional[float]
    thresholds: dict
    threshold_breaches: List[str]   # metric names that fell below threshold or were unmeasured
    samples: List[SampleScore]

    @property
    def ci_passed(self) -> bool:
        return len(self.threshold_breaches) == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci_passed"] = self.ci_passed
        d.pop("samples")    # stored separately in results.jsonl
        return d

    def print_summary(self) -> None:
        def _fmt(val: Optional[float]) -> str:
            return f"{val:.3f}" if val is not None else "N/A (unmeasured)"

        status = "✅ PASSED" if self.ci_passed else "❌ FAILED"
        print(f"\n{'─'*60}")
        print(f"Eval Report  {status}")
        print(f"{'─'*60}")
        print(f"  Total samples : {self.total}")
        print(f"  Passed        : {self.passed}  Failed: {self.failed}")
        print("")
        print(f"  Faithfulness      : {_fmt(self.avg_faithfulness)}  "
              f"(threshold={self.thresholds.get('faithfulness', '?')})")
        print(f"  Answer Relevance  : {_fmt(self.avg_answer_relevance)}  "
              f"(threshold={self.thresholds.get('answer_relevance', '?')})")
        print(f"  Context Recall    : {_fmt(self.avg_context_recall)}  "
              f"(threshold={self.thresholds.get('context_recall', '?')})")
        print(f"  Citation Coverage : {_fmt(self.avg_citation_coverage)}  "
              f"(threshold={self.thresholds.get('citation_coverage', '?')})")
        if self.avg_tool_selection is not None:
            print(f"  Tool Selection    : {_fmt(self.avg_tool_selection)}  "
                  f"(threshold={self.thresholds.get('tool_selection', '?')})")
        if self.threshold_breaches:
            print(f"\n  ⚠ Breached / unmeasured thresholds: {', '.join(self.threshold_breaches)}")
        print(f"{'─'*60}\n")


# ── Scorer ────────────────────────────────────────────────────────────────────

class FaithfulnessScorer:
    """
    Scores RAG answers using an OpenRouter LLM judge.
    All prompts are loaded from the versioned config file.
    """

    CITATION_PATTERN = re.compile(r"\[([a-f0-9]{8,64}:\d+|web:\d+)\]")

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._client = self._init_client()

    def _init_client(self):
        return create_llm_client(self.cfg)

    # ── Public API ────────────────────────────────────────────────────────────

    def score_sample(
        self,
        sample_id: str,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        expected_source: Optional[str] = None,
        actual_tools: Optional[List[str]] = None,
        valid_citations: Optional[List[str]] = None,
    ) -> SampleScore:
        """Score a single QA pair across all metrics including tool selection if expected_source is provided."""
        result = SampleScore(
            id=sample_id,
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            expected_source=expected_source,
            actual_tools=actual_tools or [],
        )

        try:
            # 1. Faithfulness
            faith = self._score_faithfulness(answer, context)
            result.faithfulness           = float(faith.get("faithfulness_score", 0.0))
            result.faithfulness_claims    = faith.get("claims", [])
            result.faithfulness_reasoning = faith.get("reasoning", "")

            # 2. Answer relevance
            rel = self._score_relevance(question, answer)
            result.answer_relevance    = float(rel.get("relevance_score", 0.0))
            result.relevance_reasoning = rel.get("reasoning", "")

            # 3. Context recall
            recall = self._score_recall(ground_truth, context)
            result.context_recall   = float(recall.get("recall_score", 0.0))
            result.recall_reasoning = recall.get("reasoning", "")

            # 4. Citation coverage (verified against retrieved context / valid citation IDs)
            result.citation_coverage = self._score_citations(
                answer, context=context, valid_citations=valid_citations
            )

            # 5. Tool selection metric (if expected_source is provided)
            if expected_source:
                result.tool_selection_accuracy = self.score_tool_selection(
                    actual_tools=actual_tools or [],
                    expected_source=expected_source,
                )

        except Exception as e:
            result.error = str(e)

        return result

    def score_batch(
        self,
        samples: List[dict],
        max_workers: int = 4,
    ) -> List[SampleScore]:
        """
        Score a list of dicts, each containing:
          id, question, answer, context, ground_truth, (optional) expected_source,
          (optional) actual_tools, (optional) valid_citations
        """
        results: List[SampleScore] = []
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = {
                pool.submit(
                    self.score_sample,
                    s["id"], s["question"], s["answer"],
                    s["context"], s["ground_truth"],
                    s.get("expected_source"), s.get("actual_tools"),
                    s.get("valid_citations"),
                ): s["id"]
                for s in samples
            }
            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r.id)
        return results

    # ── Private: LLM calls & Tool Selection ───────────────────────────────────

    @staticmethod
    def score_tool_selection(actual_tools: List[str], expected_source: Optional[str]) -> Optional[float]:
        if not expected_source:
            return None

        has_local = any(t == "search_local_documents" or "rag_query" in t for t in actual_tools)
        has_web = any(t != "search_local_documents" and "rag_query" not in t for t in actual_tools)

        exp = expected_source.lower()
        if exp == "local":
            return 1.0 if has_local and not has_web else 0.0
        elif exp == "web":
            return 1.0 if has_web and not has_local else 0.0
        elif exp == "both":
            return 1.0 if has_local and has_web else 0.0
        elif exp == "either":
            return 1.0 if (has_local or has_web) else 0.0
        return 0.0

    def _call_judge(self, prompt: str) -> dict:
        """Call the judge LLM via OpenRouter and parse JSON response."""
        if self._client is None:
            raise RuntimeError("OPENROUTER_API_KEY (or OPENAI_API_KEY) not set for LLM judge.")
        eval_cfg = self.cfg.evaluation
        model_name = getattr(getattr(eval_cfg, "scorer", None), "model", None) or get_default_model(self.cfg)

        for attempt in range(3):
            try:
                try:
                    resp = self._client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    # Fallback for free-tier models on OpenRouter that reject response_format
                    resp = self._client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                return _parse_json_response(resp.choices[0].message.content or "")
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def _score_faithfulness(self, answer: str, context: str) -> dict:
        prompt = self.cfg.prompts.faithfulness_judge.format(
            context=context, answer=answer
        )
        return self._call_judge(prompt)

    def _score_relevance(self, question: str, answer: str) -> dict:
        prompt = self.cfg.prompts.answer_relevance_judge.format(
            question=question, answer=answer
        )
        return self._call_judge(prompt)

    def _score_recall(self, ground_truth: str, context: str) -> dict:
        prompt = self.cfg.prompts.context_recall_judge.format(
            ground_truth=ground_truth, context=context
        )
        return self._call_judge(prompt)

    def _score_citations(
        self,
        answer: str,
        context: Optional[str] = None,
        valid_citations: Optional[List[str]] = None,
    ) -> float:
        """
        Verify both citation syntax, citation provenance against retrieved context/sources,
        and sentence-level coverage across multi-sentence answers.
        """
        found = self.CITATION_PATTERN.findall(answer or "")
        if not found:
            return 0.0

        found_unique = list(dict.fromkeys(found))

        # Normalize valid citation IDs from explicit list or context headers
        valid_set: Set[str] = set()
        if valid_citations is not None:
            for vc in valid_citations:
                cleaned = vc.strip().lstrip("[").rstrip("]")
                if cleaned:
                    valid_set.add(cleaned)
        elif context:
            ctx_ids = self.CITATION_PATTERN.findall(context)
            valid_set.update(ctx_ids)

        if valid_set:
            valid_found = [cid for cid in found_unique if cid in valid_set]
            if not valid_found:
                return 0.0
            validity_ratio = len(valid_found) / len(found_unique)
        else:
            validity_ratio = 1.0

        sentences = _split_factual_sentences(answer)
        if len(sentences) > 1:
            cited_sents = sum(
                1 for s in sentences
                if any(
                    (cid in valid_set if valid_set else True)
                    for cid in self.CITATION_PATTERN.findall(s)
                )
            )
            sentence_coverage = cited_sents / len(sentences)
        else:
            sentence_coverage = 1.0

        return round(validity_ratio * sentence_coverage, 4)


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(
    scores: List[SampleScore],
    thresholds: dict,
) -> EvalReport:
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot build report from empty score list")

    def _avg_metric(attr: str) -> Optional[float]:
        vals = [getattr(s, attr) for s in scores if getattr(s, attr) is not None]
        return (sum(vals) / len(vals)) if vals else None

    avg_faith  = _avg_metric("faithfulness")
    avg_rel    = _avg_metric("answer_relevance")
    avg_recall = _avg_metric("context_recall")
    avg_cit    = _avg_metric("citation_coverage")
    avg_ts     = _avg_metric("tool_selection_accuracy")

    breaches: List[str] = []
    for metric_key, avg_val in (
        ("faithfulness", avg_faith),
        ("answer_relevance", avg_rel),
        ("context_recall", avg_recall),
        ("citation_coverage", avg_cit),
    ):
        if metric_key in thresholds:
            if avg_val is None:
                breaches.append(metric_key)
            elif avg_val < thresholds[metric_key]:
                breaches.append(metric_key)

    if "tool_selection" in thresholds:
        if avg_ts is None:
            if any(s.expected_source for s in scores):
                breaches.append("tool_selection")
        elif avg_ts < thresholds["tool_selection"]:
            breaches.append("tool_selection")

    passed = sum(1 for s in scores if s.passed(thresholds))

    return EvalReport(
        total=n,
        passed=passed,
        failed=n - passed,
        avg_faithfulness=avg_faith,
        avg_answer_relevance=avg_rel,
        avg_context_recall=avg_recall,
        avg_citation_coverage=avg_cit,
        avg_tool_selection=avg_ts,
        thresholds=thresholds,
        threshold_breaches=breaches,
        samples=scores,
    )
