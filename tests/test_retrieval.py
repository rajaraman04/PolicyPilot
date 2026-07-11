"""Retrieval smoke test.

Runs a sample query against the built vector store and asserts we get
non-empty results carrying citation metadata (source filename + page).

Skips cleanly if the store has not been built yet, so `pytest` never fails
just because someone hasn't run `python -m ingest.build`.
"""

import chromadb
import pytest

from app.config import settings
from app.retriever import Retriever


def _store_ready() -> bool:
    try:
        client = chromadb.PersistentClient(path=settings.chroma_dir)
        return client.get_collection(settings.chroma_collection).count() > 0
    except Exception:
        return False


@pytest.mark.skipif(
    not _store_ready(),
    reason="vector store not built — run `python -m ingest.build` first",
)
def test_retrieve_returns_citations():
    results = Retriever(top_k=3).retrieve("What is the NIST AI Risk Management Framework?")

    assert results, "expected non-empty retrieval results"
    assert len(results) <= 3

    for c in results:
        assert isinstance(c.document, str) and c.document, "citation must have a source filename"
        assert isinstance(c.page, int) and c.page >= 1, "citation must have a page number"
        assert c.snippet.strip(), "citation must carry the retrieved text"


def test_empty_query_returns_nothing():
    # Does not require the store — guards the trivial input path.
    assert Retriever().retrieve("   ") == []
