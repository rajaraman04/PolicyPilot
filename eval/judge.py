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

from app.config import settings
from app.llm import get_llm
from app.schemas import Citation

_judge = None

# Judge calls are eval overhead, not product cost, so their token usage is
# accumulated separately from the RAG pipeline's.
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def get_judge():
    """Cached judge model — seeded, unlike the app's client."""
    global _judge
    if _judge is None:
        _judge = get_llm(seed=settings.llm_seed)
    return _judge


def reset_judge_usage() -> None:
    _usage.update(input_tokens=0, output_tokens=0, calls=0)


def get_judge_usage() -> dict[str, int]:
    return dict(_usage)


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
    usage = getattr(response, "usage_metadata", None)
    if usage:
        _usage["input_tokens"] += usage.get("input_tokens", 0)
        _usage["output_tokens"] += usage.get("output_tokens", 0)
        _usage["calls"] += 1
    content = response.content
    return content if isinstance(content, str) else str(content)


def format_context(citations: list[Citation]) -> str:
    """Render retrieved chunks the way the judge should see them."""
    return "\n\n".join(
        f"[{i}] (Source: {c.document}, p.{c.page})\n{c.snippet}"
        for i, c in enumerate(citations, start=1)
    )


SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
NOT_A_CLAIM = "NOT_A_CLAIM"
_VALID_VERDICTS = {SUPPORTED, UNSUPPORTED, NOT_A_CLAIM}

VERDICT_JUDGE_SYSTEM = (
    "You are a strict grounding evaluator. You are given CONTEXT passages and a "
    "NUMBERED list of sentences taken from an ANSWER.\n\n"
    "For EACH numbered sentence return exactly one verdict:\n"
    f"  {SUPPORTED}   - the CONTEXT states or directly entails the sentence.\n"
    f"  {UNSUPPORTED} - the sentence asserts something the CONTEXT does not support. "
    "Plausible-but-absent information is UNSUPPORTED.\n"
    f"  {NOT_A_CLAIM} - the sentence makes no factual assertion (a preamble, heading, "
    "or pure hedging).\n\n"
    "Judge only against the CONTEXT. Your own world knowledge is irrelevant. Ignore "
    "inline citation markers when judging.\n\n"
    'Return JSON only: {"verdicts": [{"index": 1, "verdict": "SUPPORTED"}, ...]}\n'
    "You MUST return exactly one entry per numbered sentence, using the same indices."
)


def judge_sentence_support(
    sentences: list[str], citations: list[Citation], llm=None
) -> list[str]:
    """Label each sentence SUPPORTED / UNSUPPORTED / NOT_A_CLAIM.

    The caller supplies the sentence list, so the *number* of items scored is
    fixed by deterministic code rather than by the model. Returns verdicts in
    the same order as ``sentences``; a response of the wrong length or with bad
    indices raises rather than silently rescaling the metric.
    """
    if not sentences:
        return []

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))
    user = f"CONTEXT:\n{format_context(citations)}\n\nANSWER SENTENCES:\n{numbered}"
    data = _extract_json(_invoke(llm, VERDICT_JUDGE_SYSTEM, user))

    raw = data.get("verdicts", [])
    if not isinstance(raw, list):
        raise ValueError(f"judge 'verdicts' must be a list, got {type(raw).__name__}")
    if len(raw) != len(sentences):
        raise ValueError(
            f"judge returned {len(raw)} verdicts for {len(sentences)} sentences — "
            "refusing to rescale the faithfulness denominator"
        )

    by_index: dict[int, str] = {}
    for entry in raw:
        try:
            idx = int(entry["index"])
            verdict = str(entry["verdict"]).strip().upper()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed verdict entry: {entry!r}") from exc
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r} (expected one of {_VALID_VERDICTS})")
        by_index[idx] = verdict

    missing = [i for i in range(1, len(sentences) + 1) if i not in by_index]
    if missing:
        raise ValueError(f"judge omitted verdicts for sentence index/indices {missing}")

    return [by_index[i] for i in range(1, len(sentences) + 1)]


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
