"""Node 3: Verifier / Decision.

Verifies the answer's citations against what was retrieved (marking any
fabricated citation [unverified]), then derives the trust/answerability decision
and a confidence score. Never invents citations or unsupported facts.
"""

from __future__ import annotations

from app.decision import decide
from app.verifier import verify_citations


def verifier_node(state: dict) -> dict:
    citations = state.get("citations", [])
    verification, marked = verify_citations(state["answer"], citations)
    decision, confidence = decide(marked, citations, verification)
    return {
        "answer": marked,
        "verification": verification,
        "decision": decision,
        "confidence": confidence,
    }
