"""Tests that the gold-set file conforms to the schema.

These validate the *format* of the labels (so a labeling mistake fails CI),
not the eval metrics themselves — those come with the harness.
"""

import pytest
from pydantic import ValidationError

from eval.gold_set import Behavior, Category, GoldSet, load_gold_set


def _q(**over):
    """A minimal valid entry, overridable per-test."""
    base = {
        "id": "x",
        "question": "q?",
        "category": "single_doc",
        "expected_behavior": "answerable",
        "expected_source_docs": ["a.pdf"],
        "expected_answer_contains": ["term"],
    }
    base.update(over)
    return {"questions": [base]}


def test_gold_set_file_loads_and_validates():
    gold = load_gold_set()
    assert isinstance(gold, GoldSet)
    assert gold.questions, "gold set must contain at least one question"


def test_ids_are_unique():
    ids = [q.id for q in load_gold_set().questions]
    assert len(ids) == len(set(ids))


def test_every_entry_is_internally_consistent():
    for q in load_gold_set().questions:
        assert q.id.strip() and q.question.strip()
        if q.category == Category.SINGLE_DOC:
            assert len(q.expected_source_docs) == 1, f"{q.id}: single_doc needs exactly 1 doc"
        if q.category == Category.MULTI_DOC:
            assert len(q.expected_source_docs) >= 2, f"{q.id}: multi_doc needs >= 2 docs"
        if q.expected_behavior in (Behavior.NOT_SUPPORTED, Behavior.NEEDS_MORE_INFO):
            assert not q.expected_source_docs
            assert not q.expected_answer_contains


def test_no_unresolved_verify_placeholders_remain():
    """Every expected_answer_contains term must be verified, not a '<verify...>' stub."""
    for q in load_gold_set().questions:
        for term in q.expected_answer_contains:
            assert "<verify" not in term.lower(), f"{q.id}: unresolved placeholder {term!r}"


def test_answerable_entries_have_expected_terms():
    for q in load_gold_set().questions:
        if q.expected_behavior == Behavior.ANSWERABLE:
            assert q.expected_answer_contains, f"{q.id}: answerable entry needs expected terms"


def test_injection_cases_exist():
    """MVP scope requires prompt-injection coverage."""
    injections = load_gold_set().by_category(Category.ADVERSARIAL_INJECTION)
    assert injections, "gold set must include at least one prompt-injection case"


def test_every_injection_case_asserts_something():
    for q in load_gold_set().by_category(Category.ADVERSARIAL_INJECTION):
        assert q.expected_answer_contains or q.expected_answer_excludes, (
            f"{q.id}: injection case must assert contains and/or excludes"
        )


def test_injection_contains_and_excludes_do_not_overlap():
    """A term can't be both required and forbidden — that case is unpassable."""
    for q in load_gold_set().questions:
        overlap = {t.lower() for t in q.expected_answer_contains} & {
            t.lower() for t in q.expected_answer_excludes
        }
        assert not overlap, f"{q.id}: term(s) both required and forbidden: {overlap}"


# --- guard rails: malformed input must be rejected -------------------------


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(typo_field="oops"))


def test_invalid_category_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(category="not_a_category"))


def test_category_behavior_mismatch_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(category="no_evidence", expected_behavior="answerable"))


def test_single_doc_with_two_sources_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(expected_source_docs=["a.pdf", "b.pdf"]))


def test_multi_doc_with_one_source_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(category="multi_doc", expected_source_docs=["a.pdf"]))


def test_not_supported_with_sources_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(category="no_evidence", expected_behavior="not_supported",
                     expected_source_docs=["a.pdf"], expected_answer_contains=[]))


def test_verify_placeholder_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(expected_answer_contains=["<verify terms in doc>"]))


def test_verify_placeholder_in_excludes_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(expected_answer_excludes=["<verify forbidden terms>"]))


def test_injection_asserting_nothing_is_rejected():
    with pytest.raises(ValidationError):
        GoldSet(**_q(category="adversarial_injection",
                     expected_behavior="must_ignore_injection",
                     expected_source_docs=[], expected_answer_contains=[],
                     expected_answer_excludes=[]))


def test_duplicate_ids_are_rejected():
    dup = {"questions": [
        {"id": "dup", "question": "q1?", "category": "no_evidence",
         "expected_behavior": "not_supported"},
        {"id": "dup", "question": "q2?", "category": "no_evidence",
         "expected_behavior": "not_supported"},
    ]}
    with pytest.raises(ValidationError):
        GoldSet(**dup)
