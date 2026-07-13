"""Single-pass RAG: retrieve -> prompt the LLM -> grounded, cited answer.

This is the baseline pipeline (no verifier). It also serves as the "single-pass
RAG" arm of the eval ablation vs. the agentic-with-verifier flow.

The prompt hard-constrains the model to use ONLY the retrieved context and to
cite every claim as (filename, page). If the context is insufficient it must say
so rather than guess — this is our no-evidence / anti-hallucination guardrail.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.retriever import Retriever
from app.schemas import AnswerResponse, Citation

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


def warmup() -> None:
    """Warm the shared retriever (loads the embedding model) before serving."""
    _retriever.warmup()


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


def answer_question(question: str, top_k: int | None = None) -> AnswerResponse:
    """Retrieve evidence and produce a grounded, cited answer."""
    start = time.perf_counter()

    citations = _retriever.retrieve(question, top_k=top_k)

    # No-evidence path: don't even call the LLM.
    if not citations:
        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerResponse(
            question=question,
            answer=NO_EVIDENCE_MSG,
            sources=[],
            latency_ms=round(latency_ms, 1),
        )

    context = _format_context(citations)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    latency_ms = (time.perf_counter() - start) * 1000
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    return AnswerResponse(
        question=question,
        answer=answer_text.strip(),
        sources=_dedupe_sources(citations),
        latency_ms=round(latency_ms, 1),
    )
