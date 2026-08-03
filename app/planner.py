"""Node 1: Planner.

Decomposes a question into the minimal set of self-contained retrieval
sub-queries. Single-topic questions yield exactly one sub-query (so behaviour is
unchanged), while comparison / multi-part questions yield one sub-query per topic
so each can be retrieved independently — the fix for cross-document questions
where a single embedding can't pull balanced evidence for both halves.

One LLM call, temperature 0, JSON out. Kept pure and injectable (the ``llm``
argument) so it can be tested without an API call.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.schemas import TokenUsage

MAX_SUBQUERIES = 4

PLANNER_SYSTEM = (
    "You are a query planner for a document-retrieval system. Decompose the user's "
    "question into the minimal set of self-contained retrieval sub-queries needed to "
    "gather evidence.\n\n"
    "- If the question is about a single topic, return exactly ONE sub-query (the "
    "question itself, lightly cleaned).\n"
    "- If it compares, contrasts, or combines multiple topics, return one sub-query "
    "per distinct topic, each phrased to stand alone (no 'it'/'the former').\n"
    f"- Return between 1 and {MAX_SUBQUERIES} sub-queries.\n\n"
    'Return JSON only: {"sub_queries": ["...", "..."]}'
)

_planner = None


def _get_planner():
    global _planner
    if _planner is None:
        _planner = get_llm()
    return _planner


def warmup_planner() -> None:
    """Build the planner client ahead of the first request (no API call)."""
    _get_planner()


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating ``` fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"planner returned unparseable JSON: {text[:200]!r}") from exc


def plan_query(question: str, llm=None) -> tuple[list[str], TokenUsage | None]:
    """Return (sub_queries, token_usage) for a question.

    Falls back to ``[question]`` if the planner returns nothing usable, so the
    pipeline degrades to single-pass retrieval rather than failing.
    """
    response = (llm or _get_planner()).invoke(
        [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=question)]
    )
    content = response.content if isinstance(response.content, str) else str(response.content)

    data = _extract_json(content)
    raw = data.get("sub_queries", [])
    if not isinstance(raw, list):
        raise ValueError(f"planner 'sub_queries' must be a list, got {type(raw).__name__}")

    sub_queries = [s.strip() for s in raw if isinstance(s, str) and s.strip()][:MAX_SUBQUERIES]
    if not sub_queries:
        sub_queries = [question]

    usage = None
    meta = getattr(response, "usage_metadata", None)
    if meta:
        usage = TokenUsage(
            input_tokens=meta.get("input_tokens", 0),
            output_tokens=meta.get("output_tokens", 0),
        )
    return sub_queries, usage
