"""LLM-as-judge plumbing for the evaluation harness.

Kept separate from metrics.py so scoring logic stays testable without API calls:
every judge entry point takes an optional ``llm`` so tests can inject a fake.

The judge runs at temperature 0 and is asked for JSON only. Parsing is defensive
— a judge that returns malformed output should raise loudly rather than silently
score everything as passing.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.schemas import Citation

_judge = None


def get_judge():
    """Cached judge model (same provider config as the app)."""
    global _judge
    if _judge is None:
        _judge = get_llm()
    return _judge


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
        raise ValueError(f"judge returned unparseable JSON: {text[:300]!r}") from exc


def _invoke(llm, system: str, user: str) -> str:
    response = (llm or get_judge()).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    content = response.content
    return content if isinstance(content, str) else str(content)


def format_context(citations: list[Citation]) -> str:
    """Render retrieved chunks the way the judge should see them."""
    return "\n\n".join(
        f"[{i}] (Source: {c.document}, p.{c.page})\n{c.snippet}"
        for i, c in enumerate(citations, start=1)
    )


CLAIM_JUDGE_SYSTEM = (
    "You are a strict grounding evaluator. You are given CONTEXT passages and an ANSWER.\n\n"
    "1. Decompose the ANSWER into atomic factual claims. A claim is a single verifiable "
    "assertion. Ignore pure meta-commentary (e.g. 'I don't have enough information'), "
    "citation markers, and hedging language.\n"
    "2. For each claim decide whether it is FULLY supported by the CONTEXT. A claim is "
    "supported only if the CONTEXT states or directly entails it. Plausible-but-absent "
    "information is NOT supported. Your own world knowledge is irrelevant.\n\n"
    'Return JSON only: {"claims": [{"claim": "...", "supported": true, "reason": "..."}]}\n'
    "If the answer contains no factual claims, return an empty claims list."
)


def judge_claims(answer: str, citations: list[Citation], llm=None) -> list[dict]:
    """Decompose an answer into claims and label each supported/unsupported."""
    user = f"CONTEXT:\n{format_context(citations)}\n\nANSWER:\n{answer}"
    data = _extract_json(_invoke(llm, CLAIM_JUDGE_SYSTEM, user))
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError(f"judge 'claims' must be a list, got {type(claims).__name__}")
    return claims


REFUSAL_JUDGE_SYSTEM = (
    "You decide whether a response DECLINED to answer.\n\n"
    "A response counts as a refusal if it states it lacks sufficient information, cannot "
    "answer from the provided documents, or asks for more information instead of "
    "answering. A response that gives a substantive factual answer is NOT a refusal, "
    "even if it adds caveats.\n\n"
    'Return JSON only: {"refusal": true} or {"refusal": false}'
)


def judge_is_refusal(answer: str, llm=None) -> bool:
    """Semantic refusal detection (not exact-string matching)."""
    data = _extract_json(_invoke(llm, REFUSAL_JUDGE_SYSTEM, f"RESPONSE:\n{answer}"))
    return bool(data.get("refusal", False))
