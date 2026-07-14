"""Single-pass RAG: retrieve -> prompt the LLM -> grounded, cited answer.

This is the baseline pipeline (no verifier). It also serves as the "single-pass
RAG" arm of the eval ablation vs. the agentic-with-verifier flow.

The prompt hard-constrains the model to use ONLY the retrieved context and to
cite every claim as (filename, page). If the context is insufficient it must say
so rather than guess — this is our no-evidence / anti-hallucination guardrail.
"""

import logging
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.retriever import Retriever
from app.schemas import AnswerResponse, Citation, LatencyBreakdown

logger = logging.getLogger("uvicorn")

NO_EVIDENCE_MSG = "I don't have enough information in the provided documents to answer that."

SYSTEM_PROMPT = (
    "You are PolicyPilot, a policy compliance assistant. Answer the user's question "
    "using ONLY the provided context passages.\n\n"
    "Rules:\n"
    "- Use only information found in the context. Never use outside knowledge.\n"
    "- Every factual statement must cite its source inline as (filename, p.PAGE), "
    "e.g. (nist_ai_rmf.pdf, p.9).\n"
    "- If the context does not contain enough information to answer, respond with "
    f'exactly: "{NO_EVIDENCE_MSG}" and cite nothing.\n'
    "- Never invent sources, page numbers, or facts.\n"
    "Be concise."
)

_SNIPPET_PREVIEW = 240

# Retriever is lazy internally, so this is cheap at import time.
_retriever = Retriever()

# The chat client is built once and reused; constructing it is surprisingly
# expensive (heavy provider imports), so we don't want that cost per query.
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def warmup() -> None:
    """Warm the retriever (embedding model) and build the LLM client before serving.

    Building the LLM client makes no API call, so it's free — but it pays the
    one-time provider-import cost here instead of on the first user query.
    """
    _retriever.warmup()
    try:
        _get_llm()
    except ValueError as exc:  # missing API key — embeddings still warmed
        logger.warning("LLM warm-up skipped (%s).", exc)


def _format_context(citations: list[Citation]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        blocks.append(f"[{i}] (Source: {c.document}, p.{c.page})\n{c.snippet}")
    return "\n\n".join(blocks)


def _dedupe_sources(citations: list[Citation]) -> list[Citation]:
    """One entry per (document, page), with a short snippet preview for the API."""
    seen: set[tuple[str, int]] = set()
    sources: list[Citation] = []
    for c in citations:
        key = (c.document, c.page)
        if key in seen:
            continue
        seen.add(key)
        preview = c.snippet.strip().replace("\n", " ")
        if len(preview) > _SNIPPET_PREVIEW:
            preview = preview[:_SNIPPET_PREVIEW] + "..."
        sources.append(Citation(document=c.document, page=c.page, snippet=preview))
    return sources


def _log_breakdown(question: str, b: LatencyBreakdown) -> None:
    logger.info(
        "query latency breakdown | embed=%.1fms retrieval=%.1fms llm=%.1fms total=%.1fms | q=%r",
        b.embed_ms,
        b.retrieval_ms,
        b.llm_ms,
        b.total_ms,
        question[:80],
    )


def answer_question(question: str, top_k: int | None = None) -> AnswerResponse:
    """Retrieve evidence and produce a grounded, cited answer.

    Times each stage (embedding, Chroma retrieval, LLM call) separately and
    returns the breakdown so callers can see which stage dominates latency.
    """
    start = time.perf_counter()

    citations, timings = _retriever.retrieve_timed(question, top_k=top_k)
    llm_ms = 0.0

    # No-evidence path: don't even call the LLM.
    if not citations:
        answer_text = NO_EVIDENCE_MSG
        sources: list[Citation] = []
    else:
        context = _format_context(citations)
        # Time the whole LLM stage (client fetch + call). After warm-up the
        # client is already built, so this is essentially the API round-trip.
        t_llm = time.perf_counter()
        llm = _get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
            ]
        )
        llm_ms = (time.perf_counter() - t_llm) * 1000
        answer_text = response.content if isinstance(response.content, str) else str(response.content)
        answer_text = answer_text.strip()
        sources = _dedupe_sources(citations)

    total_ms = (time.perf_counter() - start) * 1000
    breakdown = LatencyBreakdown(
        embed_ms=timings["embed_ms"],
        retrieval_ms=timings["retrieval_ms"],
        llm_ms=round(llm_ms, 1),
        total_ms=round(total_ms, 1),
    )
    _log_breakdown(question, breakdown)

    return AnswerResponse(
        question=question,
        answer=answer_text,
        sources=sources,
        latency_ms=breakdown.total_ms,
        timings=breakdown,
    )
