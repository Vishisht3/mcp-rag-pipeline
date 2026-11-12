"""
retrieval/citation_enforcer.py
Post-generation citation validation and retry loop.

After the LLM produces an answer, this module:
  1. Scans the answer for inline citations matching [xxxxxxxx:N] or [web:N]
  2. Verifies each citation ID exists in the retrieved chunks / web sources
  3. Verifies every factual sentence in multi-sentence answers carries a citation
  4. Verifies lexical support between cited sentences and substantive chunks
  5. If violations are found → retry / warn / raise (configurable)
"""
from __future__ import annotations

import re
import string
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from config.loader import CitationEnforcementConfig
from store.vector_store import RetrievedChunk


@dataclass
class CitationCheckResult:
    answer: str
    is_valid: bool
    found_citations: List[str]
    invalid_citations: List[str]
    missing_count: int
    violation_messages: List[str] = field(default_factory=list)


WEB_PATTERN = re.compile(r"\[web:(\d+)\]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "have", "has", "had", "not", "but", "what", "which", "when", "where", "who",
        "how", "why", "can", "could", "would", "should", "will", "into", "over",
        "more", "some", "such", "than", "then", "them", "they", "their", "there",
        "here", "also", "only", "both", "each", "other", "about", "between",
    }
)


def _content_tokens(text: str) -> Set[str]:
    cleaned = re.sub(r"\[[^\]]+\]", " ", text.lower())
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    return {w for w in cleaned.split() if len(w) >= 3 and w not in STOPWORDS}


def _is_abstention(text: str) -> bool:
    low = text.lower().strip()
    return any(
        phrase in low
        for phrase in (
            "don't have enough information",
            "do not have enough information",
            "insufficient information",
            "don't know",
            "do not know",
            "cannot answer",
        )
    )


def _split_factual_sentences(answer: str) -> List[str]:
    raw_parts = [p.strip() for p in SENTENCE_SPLIT_PATTERN.split(answer.strip()) if p.strip()]
    factual = []
    for part in raw_parts:
        if _is_abstention(part):
            continue
        # Strip citations before counting words
        words = re.sub(r"\[[^\]]+\]", "", part).split()
        if len(words) >= 3:
            factual.append(part)
    return factual


class CitationEnforcer:
    """
    Validates that an LLM answer contains well-formed inline citations
    that reference the actually-retrieved chunks or web sources, that every
    factual sentence is cited, and that cited chunks lexically support the claim.
    """

    def __init__(self, cfg: CitationEnforcementConfig):
        self.cfg = cfg
        self._pattern = re.compile(cfg.citation_pattern)

    def _check_sentence_coverage_and_support(
        self,
        answer: str,
        chunk_map: Dict[str, RetrievedChunk],
        allow_web: bool = False,
    ) -> tuple[int, List[str]]:
        violations: List[str] = []
        uncited_count = 0

        sentences = _split_factual_sentences(answer)
        require_all = getattr(self.cfg, "require_all_sentences_cited", True)
        verify_support = getattr(self.cfg, "verify_lexical_support", True)

        if require_all and len(sentences) > 1:
            for s in sentences:
                has_local = bool(self._pattern.search(s))
                has_web = bool(WEB_PATTERN.search(s)) if allow_web else False
                if not (has_local or has_web):
                    uncited_count += 1
            if uncited_count > 0:
                violations.append(
                    f"{uncited_count} of {len(sentences)} factual sentence(s) lack an inline citation."
                )

        if verify_support and sentences:
            for s in sentences:
                cited_ids = self._pattern.findall(s)
                if not cited_ids:
                    continue
                sent_tokens = _content_tokens(s)
                if len(sent_tokens) < 2:
                    continue
                # Check if at least one cited valid chunk with substantive content shares terms
                valid_cited_chunks = [chunk_map[cid] for cid in cited_ids if cid in chunk_map]
                substantive_chunks = [
                    c for c in valid_cited_chunks if len(_content_tokens(c.text)) >= 5
                ]
                if substantive_chunks:
                    supported = any(
                        bool(sent_tokens & _content_tokens(c.text))
                        for c in substantive_chunks
                    )
                    if not supported:
                        violations.append(
                            f"Cited chunk(s) {cited_ids} do not lexically support claim: '{s[:80]}'"
                        )

        return uncited_count, violations

    def check(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> CitationCheckResult:
        """
        Check an answer against the retrieved chunks (local-only).
        """
        chunk_map: Dict[str, RetrievedChunk] = {c.citation_id: c for c in retrieved_chunks}
        valid_ids: Set[str] = set(chunk_map.keys())
        found = self._pattern.findall(answer)
        found_unique = list(dict.fromkeys(found))

        invalid = [cid for cid in found_unique if cid not in valid_ids]
        missing = max(0, self.cfg.min_citations_required - len(found_unique))

        violations: List[str] = []
        if missing > 0:
            violations.append(
                f"Answer contains {len(found_unique)} citation(s); "
                f"at least {self.cfg.min_citations_required} required."
            )
        if invalid:
            violations.append(
                f"Hallucinated citation IDs (not in retrieved set): {invalid}"
            )

        uncited_sents, claim_violations = self._check_sentence_coverage_and_support(
            answer, chunk_map, allow_web=False
        )
        missing += uncited_sents
        violations.extend(claim_violations)

        return CitationCheckResult(
            answer=answer,
            is_valid=(len(violations) == 0),
            found_citations=found_unique,
            invalid_citations=invalid,
            missing_count=missing,
            violation_messages=violations,
        )

    def check_agentic(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        web_sources: Optional[List[dict]] = None,
    ) -> CitationCheckResult:
        """
        Check an agentic answer supporting both local [doc_id:chunk_index] and web [web:n] citations.
        """
        chunk_map: Dict[str, RetrievedChunk] = {c.citation_id: c for c in retrieved_chunks}
        valid_local_ids: Set[str] = set(chunk_map.keys())
        local_found = self._pattern.findall(answer)
        web_found_indices = WEB_PATTERN.findall(answer)

        local_found_unique = list(dict.fromkeys(local_found))
        web_found_unique = list(dict.fromkeys(web_found_indices))

        invalid_local = [cid for cid in local_found_unique if cid not in valid_local_ids]

        num_web_sources = len(web_sources or [])
        invalid_web = [
            f"web:{idx}" for idx in web_found_unique
            if not (idx.isdigit() and 1 <= int(idx) <= num_web_sources)
        ]

        total_citations = len(local_found_unique) + len(web_found_unique)
        missing = max(0, self.cfg.min_citations_required - total_citations)

        violations: List[str] = []
        if missing > 0:
            violations.append(
                f"Answer contains {total_citations} citation(s); "
                f"at least {self.cfg.min_citations_required} required."
            )
        if invalid_local:
            violations.append(f"Hallucinated local citation IDs: {invalid_local}")
        if invalid_web:
            violations.append(f"Hallucinated web citation IDs: {invalid_web}")

        uncited_sents, claim_violations = self._check_sentence_coverage_and_support(
            answer, chunk_map, allow_web=True
        )
        missing += uncited_sents
        violations.extend(claim_violations)

        all_found = [f"[{cid}]" for cid in local_found_unique] + [f"[web:{idx}]" for idx in web_found_unique]
        all_invalid = invalid_local + invalid_web

        return CitationCheckResult(
            answer=answer,
            is_valid=(len(violations) == 0),
            found_citations=all_found,
            invalid_citations=all_invalid,
            missing_count=missing,
            violation_messages=violations,
        )

    def enforce(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        retry_fn: Optional[Callable[[str], str]] = None,
    ) -> CitationCheckResult:
        """
        Check and — if configured — retry or raise on violation (local RAG).
        """
        if not self.cfg.enabled:
            return CitationCheckResult(
                answer=answer,
                is_valid=True,
                found_citations=self._pattern.findall(answer),
                invalid_citations=[],
                missing_count=0,
            )

        result = self.check(answer, retrieved_chunks)
        if result.is_valid:
            return result

        if self.cfg.on_violation == "raise":
            raise CitationViolationError(result)

        if self.cfg.on_violation == "warn":
            warnings.warn(
                "Citation violation: " + "; ".join(result.violation_messages),
                stacklevel=2,
            )
            return result

        if self.cfg.on_violation == "retry" and retry_fn is not None:
            for _ in range(self.cfg.max_retries):
                new_answer = retry_fn(result.answer)
                result = self.check(new_answer, retrieved_chunks)
                if result.is_valid:
                    return result
            warnings.warn(
                f"Citation enforcement: {self.cfg.max_retries} retries exhausted. "
                "Returning best available answer.",
                stacklevel=2,
            )

        return result

    def enforce_agentic(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        web_sources: Optional[List[dict]] = None,
        retry_fn: Optional[Callable[[str], str]] = None,
    ) -> CitationCheckResult:
        """
        Check and — if configured — retry or raise on violation for agentic answers.
        """
        if not self.cfg.enabled:
            return CitationCheckResult(
                answer=answer,
                is_valid=True,
                found_citations=[],
                invalid_citations=[],
                missing_count=0,
            )

        result = self.check_agentic(answer, retrieved_chunks, web_sources)
        if result.is_valid:
            return result

        if self.cfg.on_violation == "raise":
            raise CitationViolationError(result)

        if self.cfg.on_violation == "warn":
            warnings.warn(
                "Citation violation: " + "; ".join(result.violation_messages),
                stacklevel=2,
            )
            return result

        if self.cfg.on_violation == "retry" and retry_fn is not None:
            for _ in range(self.cfg.max_retries):
                new_answer = retry_fn(result.answer)
                result = self.check_agentic(new_answer, retrieved_chunks, web_sources)
                if result.is_valid:
                    return result
            warnings.warn(
                f"Citation enforcement: {self.cfg.max_retries} retries exhausted. "
                "Returning best available answer.",
                stacklevel=2,
            )

        return result


class CitationViolationError(Exception):
    def __init__(self, result: CitationCheckResult):
        self.result = result
        super().__init__(
            "Citation enforcement failed:\n"
            + "\n".join(f"  • {v}" for v in result.violation_messages)
        )
