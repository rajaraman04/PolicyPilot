"""Node 1: Planner.

Interprets the policy question and decides what evidence to retrieve.
"""

from app.graph import GraphState


def planner_node(state: GraphState) -> GraphState:
    """Produce a retrieval plan from the question.

    TODO: call the configured LLM to turn state["question"] into state["plan"].
    """
    raise NotImplementedError("Planner node pending.")
