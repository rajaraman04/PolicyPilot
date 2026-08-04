"""Node 1: Planner.

Decomposes the question into retrieval sub-queries. Thin graph wrapper around
app.planner.plan_query. Returns a partial state update (LangGraph merges it).
"""

from __future__ import annotations

from app.planner import plan_query


def planner_node(state: dict) -> dict:
    sub_queries, usage = plan_query(state["question"])
    return {"planned_queries": sub_queries, "planner_usage": usage}
