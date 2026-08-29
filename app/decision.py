"""Decision + confidence derivation for the Verifier/Decision node.

Trust/answerability semantics (deterministic — no LLM call, no eval-judge):

  NEEDS_MORE_INFO : the system declined (no/insufficient evidence, ambiguous), OR
                    it answered but attributed nothing (no citations at all), so
                    grounding can't be confirmed — we don't reject it, we ask for
                    more/clearer input.
  DENIED          : the answer cited sources, but every citation was fabricated
                    (not in the retrieved set). It actively misattributes — we
                    refuse to stand behind it.
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

    # Cited sources, but every one was fabricated (and stripped) → actively
    # misattributes. This is the only "Denied" case.
    if verification is not None and verification.fabricated and not verification.passed:
        return Decision.DENIED, confidence

    # Produced an answer but cited nothing at all → can't confirm grounding.
    # Not a rejection (the content may be fine), just unverifiable → ask for more.
    if not parse_citations(answer):
        return Decision.NEEDS_MORE_INFO, confidence

    return Decision.APPROVED, confidence
