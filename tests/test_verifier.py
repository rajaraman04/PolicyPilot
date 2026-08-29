"""Tests for the citation verifier."""

from app.schemas import Citation
from app.verifier import UNVERIFIED_TAG, verify_citations


def cite(doc="nist_csf.pdf", page=8):
    return Citation(document=doc, page=page, snippet="text")


def test_all_citations_present_passes_untouched():
    ans = "The Core has six functions (nist_csf.pdf, p.8). Govern is central (nist_csf.pdf, p.9)."
    result, marked = verify_citations(ans, [cite(page=8), cite(page=9)])
    assert result.verified is True
    assert marked == ans  # unchanged
    assert len(result.passed) == 2 and not result.fabricated


def test_fabricated_citation_is_detected_and_marked():
    ans = "The Core is a taxonomy of outcomes (nist_csf.pdf, p.99)."
    result, marked = verify_citations(ans, [cite(page=8)])
    assert result.verified is False
    assert [(c.document, c.page) for c in result.fabricated] == [("nist_csf.pdf", 99)]
    assert "(nist_csf.pdf, p.99)" not in marked
    assert UNVERIFIED_TAG in marked
    # The claim text itself is preserved.
    assert "The Core is a taxonomy of outcomes" in marked


def test_mixed_passed_and_fabricated():
    ans = ("The Core is a taxonomy (nist_csf.pdf, p.9). "
           "Detect finds attacks (nist_csf.pdf, p.8).")
    result, marked = verify_citations(ans, [cite(page=8)])  # only p.8 retrieved
    assert result.verified is False
    assert [(c.document, c.page) for c in result.passed] == [("nist_csf.pdf", 8)]
    assert [(c.document, c.page) for c in result.fabricated] == [("nist_csf.pdf", 9)]
    # Passed citation untouched, fabricated one replaced.
    assert "(nist_csf.pdf, p.8)" in marked
    assert "(nist_csf.pdf, p.9)" not in marked


def test_filename_match_is_case_insensitive():
    ans = "Claim (NIST_CSF.PDF, p.8)."
    result, marked = verify_citations(ans, [cite(doc="nist_csf.pdf", page=8)])
    assert result.verified is True and not result.fabricated


def test_wrong_document_is_fabricated_even_if_page_retrieved():
    """Right page number but a document that was never retrieved is fabricated."""
    ans = "Claim (nist_ai_rmf.pdf, p.8)."
    result, marked = verify_citations(ans, [cite(doc="nist_csf.pdf", page=8)])
    assert result.verified is False
    assert [(c.document, c.page) for c in result.fabricated] == [("nist_ai_rmf.pdf", 8)]


def test_answer_with_no_citations_verifies_trivially():
    ans = "I don't have enough information in the provided documents to answer that."
    result, marked = verify_citations(ans, [])
    assert result.verified is True
    assert marked == ans
    assert not result.passed and not result.fabricated


def test_accepts_tuple_pairs_as_retrieved():
    ans = "Claim (nist_csf.pdf, p.8)."
    result, _ = verify_citations(ans, [("nist_csf.pdf", 8)])
    assert result.verified is True


def test_fabricated_citation_inside_a_group_is_caught():
    """Regression: a fabricated citation grouped with a valid one — the verifier
    must flag only the fabricated page, not miss the whole group."""
    ans = "Tiers are defined (nist_csf.pdf, p.4; nist_csf.pdf, p.31)."
    result, marked = verify_citations(ans, [cite(page=4)])  # only p.4 retrieved
    assert [(c.document, c.page) for c in result.passed] == [("nist_csf.pdf", 4)]
    assert [(c.document, c.page) for c in result.fabricated] == [("nist_csf.pdf", 31)]
    assert result.verified is False
    assert "p.4" in marked and "p.31" not in marked
    assert UNVERIFIED_TAG in marked


def test_repeated_fabricated_citation_all_occurrences_marked():
    ans = "First (nist_csf.pdf, p.99). Second (nist_csf.pdf, p.99)."
    result, marked = verify_citations(ans, [cite(page=8)])
    assert marked.count(UNVERIFIED_TAG) == 2
    assert "p.99" not in marked
