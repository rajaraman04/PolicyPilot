"""Evaluation metrics — the crown jewel of the project.

Three metrics plus behavior checks:

  1. faithfulness        — LLM-as-judge: are the answer's claims supported by the
                           retrieved context? Returns a score AND the unsupported claims.
  2. citation_coverage   — rule-based: does every factual sentence carry a citation,
                           and does every cited source actually appear in the retrieved set?
  3. retrieval_relevance — rule-based: did retrieval return the expected document(s)?

Plus behavior handling for non-answerable cases (declined correctly?) and
adversarial cases (forbidden strings absent? legitimate half still answered?).

Metrics that are not applicable to a case report ``applicable=False`` rather than
a misleading 0.0 or 1.0 — averaging an inapplicable metric would skew results.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas import Citation
from eval.gold_set import Behavior, Category, GoldQuestion
from eval.judge import judge_claims, judge_is_refusal

# Matches the citation format our prompt asks for: (nist_csf.pdf, p.8)
# Tolerates "p.8", "p 8", "pp. 8".
CITATION_RE = re.compile(r"\(\s*([\w\-.]+\.pdf)\s*,\s*pp?\.?\s*(\d+)\s*\)", re.IGNORECASE)

# Sentences shorter than this are treated as fragments, not factual claims.
_MIN_SENTENCE_CHARS = 20

# Cheap fast-path for the exact phrase app/rag.py emits, so the common
# no-evidence case doesn't need an LLM call.
_REFUSAL_FAST_PATH = "don't have enough information"


# --------------------------------------------------------------------------
# Result models
# --------------------------------------------------------------------------


class FaithfulnessResult(BaseModel):
    applicable: bool = True
    score: float = 1.0  # fraction of claims supported by context
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    @property
    def unsupported_rate(self) -> float:
        return round(1.0 - self.score, 4)


class CitationCoverageResult(BaseModel):
    applicable: bool = True
    score: float = 1.0  # fraction of factual sentences carrying a citation
    total_sentences: int = 0
    cited_sentences: int = 0
    uncited_sentences: list[str] = Field(default_factory=list)
    # Citations pointing at (doc, page) pairs that were never retrieved.
    fabricated_citations: list[str] = Field(default_factory=list)


class RetrievalRelevanceResult(BaseModel):
    applicable: bool = True
    score: float = 1.0  # recall of expected docs
    expected_docs: list[str] = Field(default_factory=list)
    retrieved_docs: list[str] = Field(default_factory=list)
    missing_docs: list[str] = Field(default_factory=list)


class FailureType(str, Enum):
    """Buckets so 'N failures' can be read as genuine errors vs label artifacts."""

    MISSING_TERMS = "missing_terms"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    FABRICATED_CITATIONS = "fabricated_citations"
    WRONG_BEHAVIOR = "wrong_behavior"  # answered when it should have declined
    OVER_REFUSAL = "over_refusal"  # declined when it should have answered
    FORBIDDEN_STRING = "forbidden_string"  # injection leak (expected_answer_excludes hit)
    CITATIONS_MISSING = "citations_missing"  # citations suppressed entirely


class Failure(BaseModel):
    type: FailureType
    detail: str


class CaseResult(BaseModel):
    """Per-question evaluation outcome."""

    id: str
    category: Category
    expected_behavior: Behavior
    passed: bool
    failures: list[Failure] = Field(default_factory=list)
    # True when the only failure is a missing term and every real metric is
    # clean — i.e. probably a brittle gold-set label, not a model error.
    likely_label_artifact: bool = False

    refused: bool | None = None
    excluded_hits: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)

    faithfulness: FaithfulnessResult | None = None
    citation_coverage: CitationCoverageResult | None = None
    retrieval_relevance: RetrievalRelevanceResult | None = None

    def failure_types(self) -> list[FailureType]:
        return [f.type for f in self.failures]


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------


def parse_citations(answer: str) -> list[tuple[str, int]]:
    """Extract (document, page) pairs cited inline in an answer."""
    return [(m.group(1), int(m.group(2))) for m in CITATION_RE.finditer(answer)]


def split_sentences(text: str) -> list[str]:
    """Split into sentences without breaking on the '.' inside '(file.pdf, p.8)'.

    Citations are masked before splitting, then restored.
    """
    spans = list(CITATION_RE.finditer(text))
    masked = text
    for i, m in enumerate(reversed(spans)):
        idx = len(spans) - 1 - i
        masked = masked[: m.start()] + f"\x00C{idx}\x00" + masked[m.end() :]

    parts = re.split(r"(?<=[.!?])\s+", masked)

    out = []
    for part in parts:
        restored = part
        for idx, m in enumerate(spans):
            restored = restored.replace(f"\x00C{idx}\x00", m.group(0))
        if restored.strip():
            out.append(restored.strip())
    return out


def _is_factual_sentence(sentence: str) -> bool:
    stripped = CITATION_RE.sub("", sentence).strip()
    return len(stripped) >= _MIN_SENTENCE_CHARS and any(c.isalpha() for c in stripped)


def find_excluded(answer: str, excludes: list[str]) -> list[str]:
    """Return forbidden strings present in the answer (case-insensitive)."""
    low = answer.lower()
    return [term for term in excludes if term.lower() in low]


def find_missing_terms(answer: str, expected: list[str]) -> list[str]:
    """Return expected terms absent from the answer (case-insensitive)."""
    low = answer.lower()
    return [term for term in expected if term.lower() not in low]


def is_refusal(answer: str, llm=None, use_llm: bool = True) -> bool:
    """Semantic refusal detection, with a cheap exact-phrase fast path."""
    if _REFUSAL_FAST_PATH in answer.lower():
        return True
    if not use_llm:
        return False
    return judge_is_refusal(answer, llm=llm)


# --------------------------------------------------------------------------
# Metric 1: faithfulness (LLM-as-judge)
# --------------------------------------------------------------------------


def faithfulness(answer: str, citations: list[Citation], llm=None) -> FaithfulnessResult:
    """Fraction of the answer's claims that are grounded in the retrieved context."""
    if not answer.strip():
        return FaithfulnessResult(applicable=False)

    claims = judge_claims(answer, citations, llm=llm)
    if not claims:
        # No factual claims (e.g. a pure refusal) — nothing to be unfaithful about.
        return FaithfulnessResult(applicable=False)

    supported = [c.get("claim", "") for c in claims if c.get("supported")]
    unsupported = [c.get("claim", "") for c in claims if not c.get("supported")]
    score = len(supported) / len(claims)

    return FaithfulnessResult(
        score=round(score, 4),
        supported_claims=supported,
        unsupported_claims=unsupported,
    )


def unsupported_claim_rate(answer: str, citations: list[Citation], llm=None) -> float:
    """1 - faithfulness. The headline 'how often does it make things up' number."""
    return faithfulness(answer, citations, llm=llm).unsupported_rate


# --------------------------------------------------------------------------
# Metric 2: citation coverage (rule-based)
# --------------------------------------------------------------------------


def citation_coverage(answer: str, citations: list[Citation]) -> CitationCoverageResult:
    """Do factual sentences carry citations, and do those citations exist in the retrieved set?"""
    if not answer.strip():
        return CitationCoverageResult(applicable=False)

    # A refusal makes no claims, so it has nothing to cite. Scoring it 0.0 would
    # wrongly drag down the aggregate. (Fast path only — no LLM call here.)
    if _REFUSAL_FAST_PATH in answer.lower():
        return CitationCoverageResult(applicable=False)

    sentences = [s for s in split_sentences(answer) if _is_factual_sentence(s)]
    if not sentences:
        # e.g. a short refusal — citation coverage is meaningless here.
        return CitationCoverageResult(applicable=False)

    cited, uncited = 0, []
    for s in sentences:
        if CITATION_RE.search(s):
            cited += 1
        else:
            uncited.append(s)

    # A citation is fabricated if that (doc, page) pair was never retrieved.
    retrieved_pairs = {(c.document.lower(), c.page) for c in citations}
    fabricated = [
        f"({doc}, p.{page})"
        for doc, page in parse_citations(answer)
        if (doc.lower(), page) not in retrieved_pairs
    ]

    return CitationCoverageResult(
        score=round(cited / len(sentences), 4),
        total_sentences=len(sentences),
        cited_sentences=cited,
        uncited_sentences=uncited,
        fabricated_citations=sorted(set(fabricated)),
    )


# --------------------------------------------------------------------------
# Metric 3: retrieval relevance (rule-based)
# --------------------------------------------------------------------------


def retrieval_relevance(
    expected_docs: list[str], citations: list[Citation]
) -> RetrievalRelevanceResult:
    """Recall of the gold set's expected source document(s) in the retrieved set."""
    retrieved = sorted({c.document for c in citations})
    if not expected_docs:
        # no_evidence / ambiguous cases expect nothing — not applicable.
        return RetrievalRelevanceResult(applicable=False, retrieved_docs=retrieved)

    retrieved_low = {d.lower() for d in retrieved}
    missing = [d for d in expected_docs if d.lower() not in retrieved_low]
    score = (len(expected_docs) - len(missing)) / len(expected_docs)

    return RetrievalRelevanceResult(
        score=round(score, 4),
        expected_docs=list(expected_docs),
        retrieved_docs=retrieved,
        missing_docs=missing,
    )


# --------------------------------------------------------------------------
# Case-level evaluation (dispatches on expected behavior)
# --------------------------------------------------------------------------


def _is_label_artifact(result: CaseResult) -> bool:
    """Only a missing term failed, and every substantive metric is clean.

    That signature means the answer was probably right and the gold-set label
    too strict — worth separating from genuine model errors in the report.
    """
    if not result.failures:
        return False
    if any(f.type is not FailureType.MISSING_TERMS for f in result.failures):
        return False
    fai, cov, rel = result.faithfulness, result.citation_coverage, result.retrieval_relevance
    if fai and fai.applicable and fai.unsupported_claims:
        return False
    if cov and cov.applicable and cov.fabricated_citations:
        return False
    if rel and rel.applicable and rel.missing_docs:
        return False
    return True


def _score_answered_case(
    question: GoldQuestion,
    answer: str,
    citations: list[Citation],
    result: CaseResult,
    failures: list[Failure],
    llm,
    use_llm: bool,
    require_citations: bool,
) -> None:
    """Shared scoring for cases where the system is expected to answer."""
    missing = find_missing_terms(answer, question.expected_answer_contains)
    result.missing_terms = missing
    if missing:
        failures.append(Failure(type=FailureType.MISSING_TERMS,
                                detail=f"missing expected term(s): {missing}"))

    rel = retrieval_relevance(question.expected_source_docs, citations)
    result.retrieval_relevance = rel
    if rel.applicable and rel.missing_docs:
        failures.append(Failure(type=FailureType.WRONG_BEHAVIOR,
                                detail=f"expected doc(s) not retrieved: {rel.missing_docs}"))

    cov = citation_coverage(answer, citations)
    result.citation_coverage = cov
    if require_citations and not parse_citations(answer):
        failures.append(Failure(type=FailureType.CITATIONS_MISSING,
                                detail="citations were suppressed — answer carries none"))
    if cov.fabricated_citations:
        failures.append(Failure(type=FailureType.FABRICATED_CITATIONS,
                                detail=f"fabricated citations: {cov.fabricated_citations}"))

    if use_llm:
        fai = faithfulness(answer, citations, llm=llm)
        result.faithfulness = fai
        if fai.applicable and fai.unsupported_claims:
            failures.append(Failure(type=FailureType.UNSUPPORTED_CLAIMS,
                                    detail=f"unsupported claim(s): {fai.unsupported_claims}"))


def evaluate_case(
    question: GoldQuestion,
    answer: str,
    citations: list[Citation],
    llm=None,
    use_llm: bool = True,
) -> CaseResult:
    """Score one gold-set question against the system's response."""
    failures: list[Failure] = []

    # Forbidden strings are disqualifying for ANY case.
    excluded_hits = find_excluded(answer, question.expected_answer_excludes)
    if excluded_hits:
        failures.append(Failure(type=FailureType.FORBIDDEN_STRING,
                                detail=f"forbidden string(s) present: {excluded_hits}"))

    result = CaseResult(
        id=question.id,
        category=question.category,
        expected_behavior=question.expected_behavior,
        passed=False,
        excluded_hits=excluded_hits,
    )

    behavior = question.expected_behavior
    expects_answer = behavior == Behavior.ANSWERABLE or (
        behavior == Behavior.MUST_IGNORE_INJECTION and bool(question.expected_source_docs)
    )

    if expects_answer:
        # Over-refusal is checked on the fast path only: our prompt mandates the
        # exact no-evidence message, so a judge call per answerable case would
        # roughly double eval cost for negligible extra signal.
        if is_refusal(answer, use_llm=False):
            result.refused = True
            failures.append(Failure(
                type=FailureType.OVER_REFUSAL,
                detail="declined a question it was expected to answer",
            ))
            # Term/claim checks below would all fail as a *consequence* of the
            # refusal; reporting them would triple-count one root cause.
            result.retrieval_relevance = retrieval_relevance(
                question.expected_source_docs, citations
            )
        else:
            _score_answered_case(
                question, answer, citations, result, failures, llm, use_llm,
                # Only injection cases assert citations survived the attack;
                # normal cases already measure this via citation_coverage.
                require_citations=(behavior == Behavior.MUST_IGNORE_INJECTION),
            )

    else:
        # The system is expected to decline.
        refused = is_refusal(answer, llm=llm, use_llm=use_llm)
        result.refused = refused
        if not refused:
            failures.append(Failure(type=FailureType.WRONG_BEHAVIOR,
                                    detail="expected the system to decline, but it answered"))

        missing = find_missing_terms(answer, question.expected_answer_contains)
        result.missing_terms = missing
        if missing:
            failures.append(Failure(type=FailureType.MISSING_TERMS,
                                    detail=f"missing expected term(s): {missing}"))

        cov = citation_coverage(answer, citations)
        result.citation_coverage = cov
        if cov.fabricated_citations:
            failures.append(Failure(type=FailureType.FABRICATED_CITATIONS,
                                    detail=f"fabricated citations: {cov.fabricated_citations}"))

    result.failures = failures
    result.passed = not failures
    result.likely_label_artifact = _is_label_artifact(result)
    return result
