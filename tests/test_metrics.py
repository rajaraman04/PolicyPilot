"""Tests for the evaluation metrics.

The rule-based metrics (citation coverage, retrieval relevance, excludes,
refusal fast-path) are tested directly. Faithfulness and semantic refusal use
LLM-as-judge, so a fake LLM is injected — the suite stays deterministic, free,
and offline.
"""

import json
import re

import pytest

from app.schemas import Citation
from eval import metrics
from eval.gold_set import Behavior, Category, GoldQuestion
from eval.metrics import FailureType


# --- fakes -----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns canned JSON; records prompts for inspection."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _FakeResponse(json.dumps(self.payload))


class VerdictLLM:
    """Judge fake: emits one verdict per numbered sentence found in the prompt.

    Mirrors the real contract (exactly one verdict per supplied sentence), so
    tests don't have to hard-code how many sentences an answer splits into.
    """

    def __init__(self, default="SUPPORTED", overrides=None):
        self.default = default
        self.overrides = overrides or {}
        self.last_sentence_count = None

    def invoke(self, messages):
        user = messages[-1].content
        n = len(re.findall(r"^\s*\d+\.\s", user, re.M))
        self.last_sentence_count = n
        verdicts = [
            {"index": i, "verdict": self.overrides.get(i, self.default)}
            for i in range(1, n + 1)
        ]
        return _FakeResponse(json.dumps({"verdicts": verdicts}))


def cite(doc="nist_csf.pdf", page=8, snippet="text"):
    return Citation(document=doc, page=page, snippet=snippet)


# --- citation parsing / sentence splitting ---------------------------------


def test_parse_citations_extracts_doc_and_page():
    ans = "The CSF has six functions (nist_csf.pdf, p.8). It is risk-based (nist_csf.pdf, p. 10)."
    assert metrics.parse_citations(ans) == [("nist_csf.pdf", 8), ("nist_csf.pdf", 10)]


def test_split_sentences_does_not_break_on_citation_period():
    ans = "Govern is central (nist_csf.pdf, p.9). Detect finds attacks (nist_csf.pdf, p.8)."
    sentences = metrics.split_sentences(ans)
    assert len(sentences) == 2
    assert "p.9" in sentences[0] and "p.8" in sentences[1]


# --- metric 2: citation coverage -------------------------------------------


def test_citation_coverage_all_sentences_cited():
    ans = ("The framework defines six core functions (nist_csf.pdf, p.8). "
           "Govern informs the other five functions (nist_csf.pdf, p.9).")
    res = metrics.citation_coverage(ans, [cite(page=8), cite(page=9)])
    assert res.applicable and res.score == 1.0
    assert res.cited_sentences == 2 and not res.uncited_sentences


def test_citation_coverage_penalises_uncited_sentence():
    ans = ("The framework defines six core functions (nist_csf.pdf, p.8). "
           "It was also adopted widely across the private sector worldwide.")
    res = metrics.citation_coverage(ans, [cite(page=8)])
    assert res.score == 0.5
    assert len(res.uncited_sentences) == 1


def test_citation_coverage_flags_fabricated_citation():
    """A cited (doc, page) that was never retrieved is fabricated."""
    ans = "The maximum fine is large (nist_csf.pdf, p.99)."
    res = metrics.citation_coverage(ans, [cite(page=8)])
    assert res.fabricated_citations == ["(nist_csf.pdf, p.99)"]


def test_citation_coverage_not_applicable_for_short_refusal():
    res = metrics.citation_coverage("I don't know.", [])
    assert res.applicable is False


def test_citation_coverage_not_applicable_for_full_refusal_message():
    """A refusal makes no claims — it must not be scored 0.0 and skew averages."""
    ans = "I don't have enough information in the provided documents to answer that."
    res = metrics.citation_coverage(ans, [])
    assert res.applicable is False


# --- metric 3: retrieval relevance -----------------------------------------


def test_retrieval_relevance_full_match():
    res = metrics.retrieval_relevance(["nist_csf.pdf"], [cite(), cite(page=9)])
    assert res.applicable and res.score == 1.0 and not res.missing_docs


def test_retrieval_relevance_partial_match_multi_doc():
    res = metrics.retrieval_relevance(["nist_csf.pdf", "nist_rmf.pdf"], [cite()])
    assert res.score == 0.5
    assert res.missing_docs == ["nist_rmf.pdf"]


def test_retrieval_relevance_not_applicable_when_no_expected_docs():
    res = metrics.retrieval_relevance([], [cite()])
    assert res.applicable is False


# --- metric 1: faithfulness (judged) ---------------------------------------


TWO_SENTENCE_ANSWER = (
    "The framework defines six core functions (nist_csf.pdf, p.8). "
    "Govern informs the other five functions (nist_csf.pdf, p.9)."
)


def test_faithfulness_all_sentences_supported():
    llm = VerdictLLM(default="SUPPORTED")
    res = metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()], llm=llm)
    assert res.score == 1.0 and res.unsupported_rate == 0.0
    assert res.denominator == 2 and not res.unsupported_claims


def test_faithfulness_reports_unsupported_sentences():
    llm = VerdictLLM(default="SUPPORTED", overrides={2: "UNSUPPORTED"})
    res = metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()], llm=llm)
    assert res.score == 0.5 and res.unsupported_rate == 0.5
    assert len(res.unsupported_claims) == 1


def test_faithfulness_excludes_non_claims_from_denominator():
    llm = VerdictLLM(default="SUPPORTED", overrides={1: "NOT_A_CLAIM"})
    res = metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()], llm=llm)
    assert res.sentences_considered == 2
    assert res.denominator == 1  # the non-claim is excluded
    assert res.score == 1.0


def test_faithfulness_not_applicable_for_short_refusal():
    res = metrics.faithfulness("I don't know.", [], llm=VerdictLLM())
    assert res.applicable is False


def test_judge_malformed_json_raises():
    """A broken judge must fail loudly, not silently score everything as passing."""
    class BadLLM:
        def invoke(self, messages):
            return _FakeResponse("not json at all")

    with pytest.raises(ValueError):
        metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()], llm=BadLLM())


# --- determinism of the faithfulness denominator ---------------------------


def test_denominator_is_fixed_by_our_splitter_not_the_judge():
    """Same answer text => same sentence count, whatever the judge says.

    This is the root-cause fix for the run-to-run score variance: the model no
    longer decides how many units are scored.
    """
    counts = set()
    for verdicts in ("SUPPORTED", "UNSUPPORTED", "NOT_A_CLAIM"):
        res = metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()],
                                   llm=VerdictLLM(default=verdicts))
        counts.add(res.sentences_considered)
    assert counts == {2}, f"denominator drifted across judge behaviours: {counts}"


def test_judge_receives_exactly_the_sentences_we_counted():
    llm = VerdictLLM()
    res = metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()], llm=llm)
    assert llm.last_sentence_count == res.sentences_considered == 2


def test_wrong_length_verdict_list_raises_rather_than_rescaling():
    """A judge returning too few verdicts must not silently shrink the metric."""
    class ShortLLM:
        def invoke(self, messages):
            return _FakeResponse(json.dumps({"verdicts": [{"index": 1, "verdict": "SUPPORTED"}]}))

    with pytest.raises(ValueError, match="refusing to rescale"):
        metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()], llm=ShortLLM())


def test_unknown_verdict_label_raises():
    class WeirdLLM:
        def invoke(self, messages):
            return _FakeResponse(json.dumps({"verdicts": [
                {"index": 1, "verdict": "MAYBE"}, {"index": 2, "verdict": "SUPPORTED"},
            ]}))

    with pytest.raises(ValueError, match="unknown verdict"):
        metrics.faithfulness(TWO_SENTENCE_ANSWER, [cite()], llm=WeirdLLM())


# --- excludes / refusal ----------------------------------------------------


def test_find_excluded_is_case_insensitive():
    assert metrics.find_excluded("As FreeBot, I can help", ["freebot"]) == ["freebot"]
    assert metrics.find_excluded("A normal answer", ["freebot"]) == []


def test_refusal_fast_path_needs_no_llm():
    ans = "I don't have enough information in the provided documents to answer that."
    assert metrics.is_refusal(ans, use_llm=False) is True


def test_semantic_refusal_uses_judge():
    """Phrasing that differs from our exact message still counts as a refusal."""
    llm = FakeLLM({"refusal": True})
    assert metrics.is_refusal("The documents provided do not cover that topic.", llm=llm) is True


# --- case-level dispatch ---------------------------------------------------


def _q(**over):
    base = dict(id="t1", question="q?", category=Category.SINGLE_DOC,
                expected_behavior=Behavior.ANSWERABLE,
                expected_source_docs=["nist_csf.pdf"],
                expected_answer_contains=["Core"], expected_answer_excludes=[])
    base.update(over)
    return GoldQuestion(**base)


def test_case_passes_when_answer_grounded_cited_and_relevant():
    llm = VerdictLLM(default="SUPPORTED")
    ans = "The framework defines the Core, Profiles and Tiers (nist_csf.pdf, p.8)."
    res = metrics.evaluate_case(_q(), ans, [cite(page=8)], llm=llm)
    assert res.passed, res.failures


def test_case_fails_on_excluded_string():
    q = _q(category=Category.ADVERSARIAL_INJECTION,
           expected_behavior=Behavior.MUST_IGNORE_INJECTION,
           expected_source_docs=[], expected_answer_contains=[],
           expected_answer_excludes=["You are PolicyPilot"])
    res = metrics.evaluate_case(q, "You are PolicyPilot, a compliance assistant.", [],
                                use_llm=False)
    assert not res.passed
    assert res.excluded_hits == ["You are PolicyPilot"]


def test_no_evidence_case_fails_when_system_answers():
    q = _q(category=Category.NO_EVIDENCE, expected_behavior=Behavior.NOT_SUPPORTED,
           expected_source_docs=[], expected_answer_contains=[])
    res = metrics.evaluate_case(q, "The maximum GDPR fine is 20 million euros.", [],
                                use_llm=False)
    assert not res.passed
    assert res.refused is False


def test_no_evidence_case_passes_when_system_declines():
    q = _q(category=Category.NO_EVIDENCE, expected_behavior=Behavior.NOT_SUPPORTED,
           expected_source_docs=[], expected_answer_contains=[])
    ans = "I don't have enough information in the provided documents to answer that."
    res = metrics.evaluate_case(q, ans, [], use_llm=False)
    assert res.passed, res.failures


def test_bundled_injection_fails_when_citations_suppressed():
    """Attack told it to drop citations; answer is right but uncited => fail."""
    q = _q(category=Category.ADVERSARIAL_INJECTION,
           expected_behavior=Behavior.MUST_IGNORE_INJECTION,
           expected_source_docs=["nist_csf.pdf"],
           expected_answer_contains=["Core"], expected_answer_excludes=[])
    res = metrics.evaluate_case(q, "The framework defines the Core, Profiles and Tiers.",
                                [cite(page=8)], use_llm=False)
    assert not res.passed
    assert FailureType.CITATIONS_MISSING in res.failure_types()


# --- failure-type bucketing ------------------------------------------------


def test_over_refusal_detected_on_answerable_question():
    """Declining a question we expect answered is its own failure type."""
    ans = "I don't have enough information in the provided documents to answer that."
    res = metrics.evaluate_case(_q(), ans, [cite(page=8)], use_llm=False)
    assert not res.passed
    assert res.refused is True
    assert res.failure_types() == [FailureType.OVER_REFUSAL]


def test_over_refusal_does_not_cascade_into_missing_terms():
    """One root cause must not be triple-counted as three failures."""
    ans = "I don't have enough information in the provided documents to answer that."
    res = metrics.evaluate_case(_q(), ans, [cite(page=8)], use_llm=False)
    assert FailureType.MISSING_TERMS not in res.failure_types()
    assert len(res.failures) == 1


def test_forbidden_string_is_its_own_bucket():
    q = _q(category=Category.ADVERSARIAL_INJECTION,
           expected_behavior=Behavior.MUST_IGNORE_INJECTION,
           expected_source_docs=[], expected_answer_contains=[],
           expected_answer_excludes=["You are PolicyPilot"])
    res = metrics.evaluate_case(q, "You are PolicyPilot, a compliance assistant.", [],
                                use_llm=False)
    assert FailureType.FORBIDDEN_STRING in res.failure_types()


def test_label_artifact_flagged_when_only_a_term_is_missing():
    """Right answer, clean metrics, strict label => flagged as label artifact."""
    llm = VerdictLLM(default="SUPPORTED")
    q = _q(expected_answer_contains=["Organizational Profiles"])
    ans = "The framework defines the Core, CSF Profiles and Tiers (nist_csf.pdf, p.8)."
    res = metrics.evaluate_case(q, ans, [cite(page=8)], llm=llm)
    assert not res.passed
    assert res.failure_types() == [FailureType.MISSING_TERMS]
    assert res.likely_label_artifact is True


def test_genuine_error_is_not_flagged_as_label_artifact():
    """Missing term AND an unsupported claim => a real model error."""
    llm = VerdictLLM(default="UNSUPPORTED")
    q = _q(expected_answer_contains=["Tiers"])
    ans = "The CSF mandates ninety day audits for all systems (nist_csf.pdf, p.8)."
    res = metrics.evaluate_case(q, ans, [cite(page=8)], llm=llm)
    assert res.likely_label_artifact is False
    assert FailureType.UNSUPPORTED_CLAIMS in res.failure_types()


def test_bundled_injection_passes_when_answer_correct_and_cited():
    q = _q(category=Category.ADVERSARIAL_INJECTION,
           expected_behavior=Behavior.MUST_IGNORE_INJECTION,
           expected_source_docs=["nist_csf.pdf"],
           expected_answer_contains=["Core"], expected_answer_excludes=["You are PolicyPilot"])
    ans = "The framework defines the Core, Profiles and Tiers (nist_csf.pdf, p.8)."
    res = metrics.evaluate_case(q, ans, [cite(page=8)], use_llm=False)
    assert res.passed, res.failures
