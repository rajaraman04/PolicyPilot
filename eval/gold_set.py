"""Gold-set schema + loader for the evaluation harness.

Validation is strict on purpose: unknown fields are rejected, ids must be unique,
category/behavior must agree, and unresolved "<verify..." placeholders are
refused — so a labeling mistake fails loudly rather than silently skewing eval
numbers.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_GOLD_PATH = Path(__file__).parent / "gold_set.yaml"

PLACEHOLDER_MARKER = "<verify"


class Category(str, Enum):
    SINGLE_DOC = "single_doc"
    MULTI_DOC = "multi_doc"
    NO_EVIDENCE = "no_evidence"
    AMBIGUOUS = "ambiguous"
    ADVERSARIAL_INJECTION = "adversarial_injection"


class Behavior(str, Enum):
    ANSWERABLE = "answerable"
    NOT_SUPPORTED = "not_supported"
    NEEDS_MORE_INFO = "needs_more_info"
    MUST_IGNORE_INJECTION = "must_ignore_injection"


# Which behavior each category implies.
_EXPECTED_BEHAVIOR: dict[Category, Behavior] = {
    Category.SINGLE_DOC: Behavior.ANSWERABLE,
    Category.MULTI_DOC: Behavior.ANSWERABLE,
    Category.NO_EVIDENCE: Behavior.NOT_SUPPORTED,
    Category.AMBIGUOUS: Behavior.NEEDS_MORE_INFO,
    Category.ADVERSARIAL_INJECTION: Behavior.MUST_IGNORE_INJECTION,
}

# Behaviors where the system must NOT produce a sourced answer.
_NO_ANSWER_BEHAVIORS = {Behavior.NOT_SUPPORTED, Behavior.NEEDS_MORE_INFO}


class GoldQuestion(BaseModel):
    """One labeled evaluation question."""

    model_config = ConfigDict(extra="forbid")  # reject unknown fields (catch typos)

    id: str
    question: str
    category: Category
    expected_behavior: Behavior
    expected_source_docs: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    # Terms that must NOT appear — a leaked system prompt, a fabricated
    # citation, an attacker's payload. Primary assertion for injection tests.
    expected_answer_excludes: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("id", "question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> GoldQuestion:
        want = _EXPECTED_BEHAVIOR[self.category]
        if self.expected_behavior != want:
            raise ValueError(
                f"{self.id}: category '{self.category.value}' implies expected_behavior "
                f"'{want.value}', got '{self.expected_behavior.value}'"
            )

        n_docs = len(self.expected_source_docs)
        if self.category == Category.SINGLE_DOC and n_docs != 1:
            raise ValueError(f"{self.id}: single_doc must list exactly 1 source doc, got {n_docs}")
        if self.category == Category.MULTI_DOC and n_docs < 2:
            raise ValueError(f"{self.id}: multi_doc must list at least 2 source docs, got {n_docs}")

        if self.expected_behavior in _NO_ANSWER_BEHAVIORS:
            if self.expected_source_docs:
                raise ValueError(f"{self.id}: {self.expected_behavior.value} must have no source docs")
            if self.expected_answer_contains:
                raise ValueError(
                    f"{self.id}: {self.expected_behavior.value} must have empty expected_answer_contains"
                )
        elif self.expected_behavior == Behavior.ANSWERABLE and not self.expected_source_docs:
            raise ValueError(f"{self.id}: answerable questions must list at least one source doc")

        # An injection test that asserts nothing is not a test.
        if self.category == Category.ADVERSARIAL_INJECTION and not (
            self.expected_answer_contains or self.expected_answer_excludes
        ):
            raise ValueError(
                f"{self.id}: adversarial_injection needs expected_answer_contains and/or "
                "expected_answer_excludes — otherwise the case asserts nothing"
            )

        # Refuse unresolved labeling placeholders.
        unresolved = [
            t
            for t in (*self.expected_answer_contains, *self.expected_answer_excludes)
            if PLACEHOLDER_MARKER in t.lower()
        ]
        if unresolved:
            raise ValueError(
                f"{self.id}: unresolved placeholder(s) in expected_answer_contains: {unresolved}. "
                "Verify against the source document and replace."
            )
        return self


class GoldSet(BaseModel):
    """The full labeled question set."""

    model_config = ConfigDict(extra="forbid")

    questions: list[GoldQuestion]

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldSet:
        ids = [q.id for q in self.questions]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate question id(s): {', '.join(dupes)}")
        return self

    def by_category(self, category: Category) -> list[GoldQuestion]:
        return [q for q in self.questions if q.category == category]


def load_gold_set(path: str | Path | None = None) -> GoldSet:
    """Load and validate the gold-set file. Raises on malformed content.

    Accepts either a bare top-level list of questions, or a mapping with a
    ``questions:`` key.
    """
    path = Path(path) if path else DEFAULT_GOLD_PATH
    if not path.exists():
        raise FileNotFoundError(f"gold set not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if isinstance(raw, list):
        raw = {"questions": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"gold set must be a list or mapping, got {type(raw).__name__}")

    return GoldSet(**raw)
