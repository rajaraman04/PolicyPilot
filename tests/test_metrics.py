"""Tests for eval metrics.

Per project convention, every eval metric gets a test as it is built. These are
placeholders (xfail) until the corresponding metric in eval/metrics.py is
implemented — replace each with real assertions when you build the metric.
"""

import pytest

from eval import metrics


@pytest.mark.xfail(reason="faithfulness not implemented yet", raises=NotImplementedError, strict=True)
def test_faithfulness():
    assert metrics.faithfulness([], []) == 1.0


@pytest.mark.xfail(reason="citation_coverage not implemented yet", raises=NotImplementedError, strict=True)
def test_citation_coverage():
    assert 0.0 <= metrics.citation_coverage("", []) <= 1.0


@pytest.mark.xfail(reason="retrieval_relevance not implemented yet", raises=NotImplementedError, strict=True)
def test_retrieval_relevance():
    assert 0.0 <= metrics.retrieval_relevance("", []) <= 1.0
