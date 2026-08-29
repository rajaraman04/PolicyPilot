"""Node 2: Retriever.

Retrieves evidence for the planned sub-queries (shared budget) and generates a
grounded, cited answer. Falls back to the no-evidence message when nothing is
retrieved (so the LLM isn't called and the query stays free).
"""

from __future__ import annotations

from app.rag import NO_EVIDENCE_MSG, _retriever, generate_answer


def retriever_node(state: dict) -> dict:
    citations, _timings = _retriever.retrieve_multi(state["planned_queries"])
    if not citations:
        return {"citations": [], "answer": NO_EVIDENCE_MSG, "gen_usage": None}

    answer, usage, model, fingerprint = generate_answer(
        state["question"], citations, retry_uncited=True
    )
    return {
        "citations": citations,
        "answer": answer,
        "gen_usage": usage,
        "model": model,
        "fingerprint": fingerprint,
    }
