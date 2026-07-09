"""Evaluation metrics — the crown jewel of the project.

Each metric here gets a matching test in tests/test_metrics.py as it is built.
Keep functions pure (inputs -> score) so they are easy to test and reuse.
"""

from app.schemas import Citation


def faithfulness(claims: list[str], citations: list[Citation]) -> float:
    """Fraction of answer claims that are grounded in retrieved text.

    Returns a score in [0, 1]. 1.0 - faithfulness is the unsupported-claim rate.
    """
    raise NotImplementedError


def unsupported_claim_rate(claims: list[str], citations: list[Citation]) -> float:
    """Fraction of claims NOT grounded in retrieved text (1 - faithfulness)."""
    raise NotImplementedError


def citation_coverage(answer: str, citations: list[Citation]) -> float:
    """Whether/how well the answer's statements are backed by citations. [0, 1]."""
    raise NotImplementedError


def retrieval_relevance(query: str, citations: list[Citation]) -> float:
    """How relevant the retrieved chunks are to the query. [0, 1]."""
    raise NotImplementedError
