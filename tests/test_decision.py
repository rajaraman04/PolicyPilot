"""Tests for the deterministic decision + confidence logic."""

from app.decision import citation_coverage, decide, verification_pass_rate
from app.schemas import Citation, CitationRef, Decision, VerificationResult


def _verified(pairs):
    return VerificationResult(
        verified=True,
        passed=[CitationRef(document=d, page=p) for d, p in pairs],
        fabricated=[],
    )


def _partly(passed, fabricated):
    return VerificationResult(
        verified=not fabricated,
        passed=[CitationRef(document=d, page=p) for d, p in passed],
        fabricated=[CitationRef(document=d, page=p) for d, p in fabricated],
    )


def _cite(doc="nist_csf.pdf", page=8):
    return Citation(document=doc, page=page, snippet="t")


# --- refusal -> NEEDS_MORE_INFO --------------------------------------------


def test_refusal_is_needs_more_info_with_zero_confidence():
    ans = "I don't have enough information in the provided documents to answer that."
    decision, conf = decide(ans, [], VerificationResult(verified=True))
    assert decision == Decision.NEEDS_MORE_INFO
    assert conf == 0.0


# --- grounded answer -> APPROVED -------------------------------------------


def test_grounded_verified_answer_is_approved_high_confidence():
    ans = ("The Core has six functions (nist_csf.pdf, p.8). "
           "Govern is central (nist_csf.pdf, p.9).")
    decision, conf = decide(ans, [_cite(page=8), _cite(page=9)],
                            _verified([("nist_csf.pdf", 8), ("nist_csf.pdf", 9)]))
    assert decision == Decision.APPROVED
    assert conf == 1.0  # full coverage, full verification


def test_partial_fabrication_lowers_confidence_but_stays_approved():
    """One good citation remains, so still APPROVED, but penalised."""
    ans = ("The Core has six functions (nist_csf.pdf, p.8). "
           "It also mandates audits [unverified — citation not in retrieved sources].")
    decision, conf = decide(ans, [_cite(page=8)],
                            _partly(passed=[("nist_csf.pdf", 8)], fabricated=[("nist_csf.pdf", 99)]))
    assert decision == Decision.APPROVED
    assert conf < 1.0


# --- ungrounded answer -> DENIED -------------------------------------------


def test_answer_with_no_valid_citation_is_denied():
    """All citations stripped by the verifier -> nothing grounded -> DENIED."""
    ans = "The maximum fine is large [unverified — citation not in retrieved sources]."
    decision, conf = decide(ans, [_cite(page=8)],
                            _partly(passed=[], fabricated=[("nist_csf.pdf", 99)]))
    assert decision == Decision.DENIED
    assert conf < 0.5


def test_uncited_answer_is_denied():
    """Model produced a substantive answer but never cited -> DENIED."""
    ans = "The CSF has six core functions that organizations should implement."
    decision, conf = decide(ans, [_cite(page=8)], VerificationResult(verified=True))
    assert decision == Decision.DENIED


# --- component helpers ------------------------------------------------------


def test_citation_coverage_counts_cited_factual_sentences():
    ans = ("The Core has six functions (nist_csf.pdf, p.8). "
           "It was adopted across the private sector worldwide over many years.")
    assert citation_coverage(ans) == 0.5


def test_verification_pass_rate():
    assert verification_pass_rate(_verified([("a.pdf", 1)])) == 1.0
    assert verification_pass_rate(_partly([("a.pdf", 1)], [("a.pdf", 2)])) == 0.5
    assert verification_pass_rate(_partly([], [("a.pdf", 2)])) == 0.0
    assert verification_pass_rate(None) == 0.0
