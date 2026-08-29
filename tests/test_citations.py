"""Tests for citation parsing, incl. citations grouped in one parenthesis."""

from app.citations import parse_citations, split_sentences


def test_parses_single_citation():
    assert parse_citations("The Core (nist_csf.pdf, p.8).") == [("nist_csf.pdf", 8)]


def test_parses_grouped_citations_in_one_parenthesis():
    """Regression: '(doc, p.4; doc, p.31)' must parse as TWO citations, not zero."""
    ans = "Tiers are defined (nist_csf.pdf, p.4; nist_csf.pdf, p.31)."
    assert parse_citations(ans) == [("nist_csf.pdf", 4), ("nist_csf.pdf", 31)]


def test_parses_grouped_citations_across_documents():
    ans = "Compared (nist_csf.pdf, p.8; nist_rmf.pdf, p.14)."
    assert parse_citations(ans) == [("nist_csf.pdf", 8), ("nist_rmf.pdf", 14)]


def test_parses_multiple_separate_citations():
    ans = "A (nist_csf.pdf, p.7). B (nist_csf.pdf, p.9)."
    assert parse_citations(ans) == [("nist_csf.pdf", 7), ("nist_csf.pdf", 9)]


def test_no_false_positive_on_prose():
    assert parse_citations("See page 4 of the report for details.") == []
    assert parse_citations("No citation here at all.") == []


def test_split_sentences_not_broken_by_grouped_citation_period():
    ans = "Tiers are defined (nist_csf.pdf, p.4; nist_csf.pdf, p.31). Govern is central (nist_csf.pdf, p.9)."
    sentences = split_sentences(ans)
    assert len(sentences) == 2
