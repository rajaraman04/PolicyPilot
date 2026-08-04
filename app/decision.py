"""Decision + confidence derivation for the Verifier/Decision node.

Trust/answerability semantics (deterministic — no LLM call, no eval-judge):

  NEEDS_MORE_INFO : the system declined (no/insufficient evidence, ambiguous).
  DENIED          : an answer was produced but nothing in it is grounded — it
                    carries no valid citation (all were fabricated & stripped, or
                    it never cited). We refuse to stand behind it.
  APPROVED        : a grounded, cited answer the system stands behind. Confidence
                    is reduced when some citations were fabricated or coverage is low.

Confidence is a documented heuristic composite in [0, 1], NOT a calibrated
probability: mean(citation_coverage, verification_pass_rate), or 0.0 on refusal.
"""

from __future__ import annotations

from app.citations import CITATION_RE, parse_citations, split_sentences
from app.schemas import Citation, Decision, VerificationResult

# Matches the no-evidence message app/rag.py emits (and semantic refusals we mark).
_REFUSAL_MARKER = "don't have enough information"
_MIN_SENTENCE_CHARS = 20


def _is_refusal(answer: str) -> bool:
    return _REFUSAL_MARKER in answer.lower()


def _is_factual_sentence(sentence: str) -> bool:
    stripped = CITATION_RE.sub("", sentence).strip()
    return len(stripped) >= _MIN_SENTENCE_CHARS and any(c.isalpha() for c in stripped)


def citation_coverage(answer: str) -> float:
    """Rule-based fraction of factual sentences that carry a citation (no judge)."""
    sentences = [s for s in split_sentences(answer) if _is_factual_sentence(s)]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if CITATION_RE.search(s))
    return cited / len(sentences)


def verification_pass_rate(verification: VerificationResult | None) -> float:
    """passed / (passed + fabricated); 0.0 when no citations were made."""
    if verification is None:
        return 0.0
    total = len(verification.passed) + len(verification.fabricated)
    return len(verification.passed) / total if total else 0.0


def decide(
    answer: str,
    citations: list[Citation],
    verification: VerificationResult | None,
) -> tuple[Decision, float]:
    """Map the pipeline's output signals to a (decision, confidence) pair."""
    if _is_refusal(answer):
        return Decision.NEEDS_MORE_INFO, 0.0

    confidence = round((citation_coverage(answer) + verification_pass_rate(verification)) / 2, 2)

    # An answer with no valid citation left is ungrounded — we won't stand behind it.
    if not parse_citations(answer):
        return Decision.DENIED, confidence

    return Decision.APPROVED, confidence
