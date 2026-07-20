"""Tests for the evaluation metrics.

The rule-based metrics (citation coverage, retrieval relevance, excludes,
refusal fast-path) are tested directly. Faithfulness and semantic refusal use
LLM-as-judge, so a fake LLM is injected — the suite stays deterministic, free,
and offline.
"""

import json

import pytest

from app.schemas import Citation
from eval import metrics
from eval.gold_set import Behavior, Category, GoldQuestion


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


def test_faithfulness_all_claims_supported():
    llm = FakeLLM({"claims": [
        {"claim": "CSF has six functions", "supported": True},
        {"claim": "Govern is central", "supported": True},
    ]})
    res = metrics.faithfulness("some answer", [cite()], llm=llm)
    assert res.score == 1.0 and res.unsupported_rate == 0.0
    assert not res.unsupported_claims


def test_faithfulness_reports_unsupported_claims():
    llm = FakeLLM({"claims": [
        {"claim": "CSF has six functions", "supported": True},
        {"claim": "CSF mandates 90-day audits", "supported": False},
    ]})
    res = metrics.faithfulness("some answer", [cite()], llm=llm)
    assert res.score == 0.5
    assert res.unsupported_claims == ["CSF mandates 90-day audits"]
    assert res.unsupported_rate == 0.5


def test_faithfulness_not_applicable_when_no_claims():
    llm = FakeLLM({"claims": []})
    res = metrics.faithfulness("I don't have enough information.", [], llm=llm)
    assert res.applicable is False


def test_judge_malformed_json_raises():
    """A broken judge must fail loudly, not silently score everything as passing."""
    class BadLLM:
        def invoke(self, messages):
            return _FakeResponse("not json at all")

    with pytest.raises(ValueError):
        metrics.faithfulness("answer", [cite()], llm=BadLLM())


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
    llm = FakeLLM({"claims": [{"claim": "CSF defines the Core", "supported": True}]})
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
    assert any("citations were suppressed" in f for f in res.failures)


def test_bundled_injection_passes_when_answer_correct_and_cited():
    q = _q(category=Category.ADVERSARIAL_INJECTION,
           expected_behavior=Behavior.MUST_IGNORE_INJECTION,
           expected_source_docs=["nist_csf.pdf"],
           expected_answer_contains=["Core"], expected_answer_excludes=["You are PolicyPilot"])
    ans = "The framework defines the Core, Profiles and Tiers (nist_csf.pdf, p.8)."
    res = metrics.evaluate_case(q, ans, [cite(page=8)], use_llm=False)
    assert res.passed, res.failures
