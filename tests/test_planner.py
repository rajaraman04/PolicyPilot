"""Tests for the Planner and split-budget multi-retrieval."""

import json

import pytest

from app.planner import MAX_SUBQUERIES, plan_query
from app.retriever import Retriever
from app.schemas import Citation


class _Resp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, messages):
        return _Resp(json.dumps(self.payload))


# --- plan_query ------------------------------------------------------------


def test_single_topic_returns_one_subquery():
    subs, _ = plan_query("What is the CSF Core?", llm=FakeLLM({"sub_queries": ["What is the CSF Core?"]}))
    assert subs == ["What is the CSF Core?"]


def test_comparison_returns_multiple_subqueries():
    payload = {"sub_queries": ["CSF Core functions", "AI RMF functions"]}
    subs, _ = plan_query("Compare CSF and AI RMF functions", llm=FakeLLM(payload))
    assert subs == ["CSF Core functions", "AI RMF functions"]


def test_empty_subqueries_falls_back_to_question():
    q = "Some question"
    subs, _ = plan_query(q, llm=FakeLLM({"sub_queries": []}))
    assert subs == [q]


def test_blank_entries_are_dropped():
    subs, _ = plan_query("q", llm=FakeLLM({"sub_queries": ["  ", "real query", ""]}))
    assert subs == ["real query"]


def test_subqueries_are_capped():
    payload = {"sub_queries": [f"q{i}" for i in range(10)]}
    subs, _ = plan_query("q", llm=FakeLLM(payload))
    assert len(subs) == MAX_SUBQUERIES


def test_malformed_json_raises():
    class Bad:
        def invoke(self, messages):
            return _Resp("not json")

    with pytest.raises(ValueError):
        plan_query("q", llm=Bad())


# --- split-budget multi-retrieval ------------------------------------------


class RecordingRetriever(Retriever):
    """Overrides retrieve_timed to return canned results and record top_k used."""

    def __init__(self, per_query_results):
        super().__init__()
        self._canned = per_query_results  # dict: query -> list[Citation]
        self.calls = []  # (query, top_k)

    def retrieve_timed(self, query, top_k=None):
        self.calls.append((query, top_k))
        return self._canned.get(query, []), {"embed_ms": 1.0, "retrieval_ms": 2.0}


def _c(doc, page):
    return Citation(document=doc, page=page, snippet="t")


def test_multi_retrieve_splits_budget_across_subqueries():
    r = RecordingRetriever({"a": [_c("x.pdf", 1)], "b": [_c("y.pdf", 2)]})
    cites, timings = r.retrieve_multi(["a", "b"], total_k=10)
    # ceil(10/2) = 5 per sub-query
    assert all(k == 5 for _, k in r.calls)
    # both documents represented
    assert {(c.document, c.page) for c in cites} == {("x.pdf", 1), ("y.pdf", 2)}
    # timings summed across sub-queries
    assert timings["embed_ms"] == 2.0 and timings["retrieval_ms"] == 4.0


def test_multi_retrieve_dedupes_by_doc_and_page():
    r = RecordingRetriever({"a": [_c("x.pdf", 1)], "b": [_c("x.pdf", 1), _c("y.pdf", 3)]})
    cites, _ = r.retrieve_multi(["a", "b"], total_k=6)
    assert len(cites) == 2
    assert {(c.document, c.page) for c in cites} == {("x.pdf", 1), ("y.pdf", 3)}


def test_multi_retrieve_single_query_is_like_normal():
    r = RecordingRetriever({"only": [_c("x.pdf", 1)]})
    cites, _ = r.retrieve_multi(["only"], total_k=5)
    assert r.calls == [("only", 5)]  # full budget to the one query
    assert len(cites) == 1
