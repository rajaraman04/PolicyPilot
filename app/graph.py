"""LangGraph wiring for the 3-node agentic flow.

    Planner  ->  Retriever  ->  Verifier / Decision

This is the only agentic flow in scope. Do not add more nodes/agents here
(no separate risk agent, no separate report agent) — those are v2 roadmap.
"""

from typing import TypedDict

from app.schemas import Citation, Decision


class GraphState(TypedDict, total=False):
    """State passed between nodes."""

    question: str
    plan: str
    citations: list[Citation]
    decision: Decision
    answer: str
    confidence: float


def build_graph():
    """Construct and compile the Planner -> Retriever -> Verifier graph.

    TODO: wire nodes from app.agents.{planner, retriever_node, verifier}
    into a langgraph.StateGraph and compile.
    """
    raise NotImplementedError("Graph wiring pending.")
