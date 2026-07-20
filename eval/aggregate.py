"""Aggregation across repeated evaluation runs.

Every metric is reported as mean +/- spread rather than a single value, because
a one-shot number hides run-to-run variance. Questions that pass in some runs
and fail in others are surfaced explicitly as "flaky" — that instability is
itself a finding, not noise to be averaged away.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from eval.metrics import CaseResult, FailureType


class MetricStat(BaseModel):
    """Summary of one metric across N runs."""

    n: int = 0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None

    @classmethod
    def from_values(cls, values: list[float]) -> MetricStat:
        vals = [v for v in values if v is not None]
        if not vals:
            return cls(n=0)
        return cls(
            n=len(vals),
            mean=round(statistics.fmean(vals), 4),
            std=round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            min=round(min(vals), 4),
            max=round(max(vals), 4),
        )

    def render(self, precision: int = 3) -> str:
        if self.n == 0:
            return "n/a"
        if self.n == 1:
            return f"{self.mean:.{precision}f}"
        return f"{self.mean:.{precision}f} +/- {self.std:.{precision}f}"


class QuestionSummary(BaseModel):
    id: str
    category: str
    runs: int
    passes: int
    flaky: bool = False
    likely_label_artifact: bool = False
    failure_types: list[str] = Field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.runs if self.runs else 0.0


class AggregateReport(BaseModel):
    runs: int
    questions: int

    pass_rate: MetricStat = Field(default_factory=MetricStat)
    faithfulness: MetricStat = Field(default_factory=MetricStat)
    citation_coverage: MetricStat = Field(default_factory=MetricStat)
    retrieval_relevance: MetricStat = Field(default_factory=MetricStat)

    # Failure counts split so "N failures" is readable at a glance.
    genuine_failures: dict[str, int] = Field(default_factory=dict)
    label_artifacts: int = 0

    flaky_questions: list[QuestionSummary] = Field(default_factory=list)
    per_question: list[QuestionSummary] = Field(default_factory=list)

    # Cost/latency (product only — judge cost tracked separately by the runner).
    latency_ms: MetricStat = Field(default_factory=MetricStat)
    embed_ms: MetricStat = Field(default_factory=MetricStat)
    retrieval_ms: MetricStat = Field(default_factory=MetricStat)
    llm_ms: MetricStat = Field(default_factory=MetricStat)
    cost_usd_per_query: MetricStat = Field(default_factory=MetricStat)


def aggregate(runs: list[list[CaseResult]]) -> AggregateReport:
    """Combine per-run case results into a single report."""
    if not runs:
        return AggregateReport(runs=0, questions=0)

    n_runs = len(runs)
    by_id: dict[str, list[CaseResult]] = defaultdict(list)
    for run in runs:
        for case in run:
            by_id[case.id].append(case)

    # Headline metrics are aggregated PER RUN and then across runs, so the
    # reported spread is run-to-run instability — not the (much larger, and
    # uninteresting here) spread between different questions.
    def _run_mean(run: list[CaseResult], pick) -> float | None:
        vals = [pick(c) for c in run]
        vals = [v for v in vals if v is not None]
        return statistics.fmean(vals) if vals else None

    def _faith(c):
        return c.faithfulness.score if c.faithfulness and c.faithfulness.applicable else None

    def _cov(c):
        return (c.citation_coverage.score
                if c.citation_coverage and c.citation_coverage.applicable else None)

    def _rel(c):
        return (c.retrieval_relevance.score
                if c.retrieval_relevance and c.retrieval_relevance.applicable else None)

    pass_rates = [sum(c.passed for c in run) / len(run) for run in runs if run]
    faith = [m for run in runs if run for m in [_run_mean(run, _faith)] if m is not None]
    cov = [m for run in runs if run for m in [_run_mean(run, _cov)] if m is not None]
    rel = [m for run in runs if run for m in [_run_mean(run, _rel)] if m is not None]

    genuine: Counter[str] = Counter()
    artifacts = 0
    summaries: list[QuestionSummary] = []

    for qid, cases in by_id.items():
        passes = sum(c.passed for c in cases)
        # Only excuse a question as a label artifact if EVERY failing run looks
        # like one. Using any() here hid real failures: a question flagged as an
        # artifact in two runs would suppress a genuine fabricated citation in
        # the third.
        failing = [c for c in cases if not c.passed]
        is_artifact = bool(failing) and all(c.likely_label_artifact for c in failing)
        if is_artifact:
            artifacts += 1
        else:
            for c in cases:
                for ft in c.failure_types():
                    genuine[ft.value] += 1

        summaries.append(QuestionSummary(
            id=qid,
            category=cases[0].category.value,
            runs=len(cases),
            passes=passes,
            flaky=0 < passes < len(cases),
            likely_label_artifact=is_artifact,
            failure_types=sorted({ft.value for c in cases for ft in c.failure_types()}),
        ))

    summaries.sort(key=lambda s: s.id)

    return AggregateReport(
        runs=n_runs,
        questions=len(by_id),
        pass_rate=MetricStat.from_values(pass_rates),
        faithfulness=MetricStat.from_values(faith),
        citation_coverage=MetricStat.from_values(cov),
        retrieval_relevance=MetricStat.from_values(rel),
        genuine_failures=dict(genuine.most_common()),
        label_artifacts=artifacts,
        flaky_questions=[s for s in summaries if s.flaky],
        per_question=summaries,
    )


def attach_telemetry(report: AggregateReport, responses: list) -> AggregateReport:
    """Fold per-query latency/cost telemetry into an existing report."""
    report.latency_ms = MetricStat.from_values([r.latency_ms for r in responses])
    report.embed_ms = MetricStat.from_values(
        [r.timings.embed_ms for r in responses if r.timings])
    report.retrieval_ms = MetricStat.from_values(
        [r.timings.retrieval_ms for r in responses if r.timings])
    report.llm_ms = MetricStat.from_values(
        [r.timings.llm_ms for r in responses if r.timings])
    report.cost_usd_per_query = MetricStat.from_values(
        [r.cost_usd for r in responses if r.cost_usd is not None])
    return report


def summarize_failure_counts(report: AggregateReport) -> list[tuple[str, int, int]]:
    """Rows of (failure_type, genuine_count, label_artifact_count) for display.

    Label artifacts are always missing_terms, so they merge into that row rather
    than printing a confusing duplicate line.
    """
    counts = dict(report.genuine_failures)
    key = FailureType.MISSING_TERMS.value
    rows = [(ft, count, 0) for ft, count in counts.items() if ft != key]
    if key in counts or report.label_artifacts:
        rows.append((key, counts.get(key, 0), report.label_artifacts))
    return sorted(rows, key=lambda r: -(r[1] + r[2]))
