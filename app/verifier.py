"""Citation verifier.

Checks every citation in a generated answer against the (document, page) pairs
that were actually retrieved. A citation pointing at anything not retrieved is
"fabricated" — the model invented (or misremembered) an attribution.

Handling of a fabricated citation (chosen deliberately): the answer is NOT
silently passed through, nor is the claim deleted. Instead the fabricated
(doc, p.X) marker is redacted and replaced inline with an [unverified] tag, so
the reader sees the assertion is not backed by retrieved evidence while
correct-but-mis-cited content is preserved.

Pure and LLM-free — deterministic and cheap. The eventual LangGraph
Verifier/Decision node will wrap this.
"""

from __future__ import annotations

from app.citations import CITATION_RE
from app.schemas import Citation, CitationRef, VerificationResult

UNVERIFIED_TAG = "[unverified — citation not in retrieved sources]"


def _retrieved_pairs(retrieved: list[Citation] | list[tuple[str, int]]) -> set[tuple[str, int]]:
    """Normalise retrieved evidence to a set of (lowercased doc, page) pairs."""
    pairs: set[tuple[str, int]] = set()
    for item in retrieved:
        if isinstance(item, Citation):
            pairs.add((item.document.lower(), item.page))
        else:
            doc, page = item
            pairs.add((str(doc).lower(), int(page)))
    return pairs


def verify_citations(
    answer: str,
    retrieved: list[Citation] | list[tuple[str, int]],
) -> tuple[VerificationResult, str]:
    """Verify an answer's citations against the retrieved set.

    Returns (result, marked_answer). The marked answer has each fabricated
    (doc, p.X) marker replaced with an [unverified] tag; if nothing was
    fabricated it is the original answer unchanged.
    """
    retrieved_set = _retrieved_pairs(retrieved)

    passed: list[CitationRef] = []
    fabricated: list[CitationRef] = []
    # Rebuild the answer, replacing fabricated citation spans as we go.
    out: list[str] = []
    cursor = 0

    for m in CITATION_RE.finditer(answer):
        doc, page = m.group(1), int(m.group(2))
        ref = CitationRef(document=doc, page=page)
        if (doc.lower(), page) in retrieved_set:
            passed.append(ref)
            continue  # leave the marker untouched

        # Fabricated: redact this span, keep everything before it verbatim.
        fabricated.append(ref)
        out.append(answer[cursor : m.start()])
        out.append(UNVERIFIED_TAG)
        cursor = m.end()

    out.append(answer[cursor:])
    marked_answer = "".join(out) if fabricated else answer

    return VerificationResult(
        verified=not fabricated,
        passed=passed,
        fabricated=fabricated,
    ), marked_answer
