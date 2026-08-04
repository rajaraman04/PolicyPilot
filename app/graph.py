"""LangGraph wiring for the 3-node agentic flow.

    Planner  ->  Retriever  ->  Verifier / Decision

This is the only agentic flow in scope. Do not add more nodes/agents here
(no separate risk agent, no separate report agent) — those are v2 roadmap.
"""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.planner import planner_node
from app.agents.retriever_node import retriever_node
from app.agents.verifier import verifier_node
from app.config import settings
from app.pricing import estimate_cost_usd
from app.rag import _dedupe_sources
from app.schemas import Citation, Decision, QueryResponse, TokenUsage, VerificationResult


class GraphState(TypedDict, total=False):
    """State passed between nodes (total=False: each node fills its own keys)."""

    question: str
    planned_queries: list[str]
    citations: list[Citation]
    answer: str
    verification: VerificationResult
    decision: Decision
    confidence: float
    # Telemetry, accumulated per node.
    planner_usage: TokenUsage | None
    gen_usage: TokenUsage | None
    model: str | None
    fingerprint: str | None


def build_graph():
    """Construct and compile the Planner -> Retriever -> Verifier/Decision graph."""
    g = StateGraph(GraphState)
    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("verifier_decision", verifier_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "verifier_decision")
    g.add_edge("verifier_decision", END)
    return g.compile()


_compiled = None


def _graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def answer_with_decision(question: str) -> QueryResponse:
    """Run the agentic graph and map its final state to a QueryResponse."""
    start = time.perf_counter()
    final = _graph().invoke({"question": question})
    latency_ms = (time.perf_counter() - start) * 1000

    # Product cost = Planner + generation tokens (the judge is eval-only).
    in_tokens = out_tokens = 0
    for usage in (final.get("planner_usage"), final.get("gen_usage")):
        if usage:
            in_tokens += usage.input_tokens
            out_tokens += usage.output_tokens
    model = final.get("model") or settings.openai_llm_model
    cost = estimate_cost_usd(model, in_tokens, out_tokens) if (in_tokens or out_tokens) else 0.0

    return QueryResponse(
        decision=final["decision"],
        answer=final["answer"],
        confidence=final["confidence"],
        citations=_dedupe_sources(final.get("citations", [])),
        latency_ms=round(latency_ms, 1),
        cost_usd=cost,
    )
