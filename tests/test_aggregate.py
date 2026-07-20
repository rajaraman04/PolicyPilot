"""Tests for multi-run aggregation."""

from eval.aggregate import MetricStat, aggregate, summarize_failure_counts
from eval.gold_set import Behavior, Category
from eval.metrics import CaseResult, Failure, FailureType, FaithfulnessResult


def _case(qid="q1", passed=True, faith=None, failures=None, artifact=False):
    return CaseResult(
        id=qid,
        category=Category.SINGLE_DOC,
        expected_behavior=Behavior.ANSWERABLE,
        passed=passed,
        failures=failures or [],
        likely_label_artifact=artifact,
        faithfulness=FaithfulnessResult(score=faith, denominator=2) if faith is not None else None,
    )


def test_metric_stat_reports_spread_across_runs():
    stat = MetricStat.from_values([0.6, 0.8, 1.0])
    assert stat.n == 3
    assert stat.mean == 0.8
    assert stat.min == 0.6 and stat.max == 1.0
    assert stat.std > 0


def test_single_value_has_zero_spread():
    stat = MetricStat.from_values([0.75])
    assert stat.n == 1 and stat.mean == 0.75 and stat.std == 0.0
    assert stat.render() == "0.750"


def test_render_shows_plus_minus_for_multiple_runs():
    assert "+/-" in MetricStat.from_values([0.6, 1.0]).render()


def test_empty_values_are_not_applicable():
    assert MetricStat.from_values([]).render() == "n/a"
    assert MetricStat.from_values([None, None]).n == 0


def test_flaky_question_is_detected():
    """Passing in one run and failing in another is itself the finding."""
    runs = [[_case(passed=True)], [_case(passed=False)], [_case(passed=True)]]
    report = aggregate(runs)
    assert report.runs == 3
    flaky_ids = [q.id for q in report.flaky_questions]
    assert flaky_ids == ["q1"]
    assert report.per_question[0].passes == 2


def test_stable_question_is_not_flaky():
    runs = [[_case(passed=True)] for _ in range(3)]
    assert aggregate(runs).flaky_questions == []


def test_genuine_failures_and_label_artifacts_are_counted_separately():
    """The headline split: real model errors vs too-strict labels."""
    genuine = _case(
        qid="q_bad", passed=False,
        failures=[Failure(type=FailureType.UNSUPPORTED_CLAIMS, detail="x")],
    )
    artifact = _case(
        qid="q_strict", passed=False, artifact=True,
        failures=[Failure(type=FailureType.MISSING_TERMS, detail="y")],
    )
    report = aggregate([[genuine, artifact]])
    assert report.genuine_failures == {"unsupported_claims": 1}
    assert report.label_artifacts == 1
    # The artifact's missing_terms must NOT inflate the genuine bucket.
    assert "missing_terms" not in report.genuine_failures


def test_faithfulness_averaged_across_runs():
    runs = [[_case(faith=0.6)], [_case(faith=1.0)]]
    report = aggregate(runs)
    assert report.faithfulness.mean == 0.8
    assert report.faithfulness.n == 2


def test_spread_is_run_to_run_not_question_to_question():
    """Two questions differ wildly, but both runs are identical => zero spread.

    The reported +/- must mean "how much does this move between runs", not
    "how much do questions differ", or the variance story is unreadable.
    """
    run = [_case(qid="a", faith=0.2), _case(qid="b", faith=1.0)]
    report = aggregate([list(run), list(run)])
    assert report.faithfulness.mean == 0.6  # mean of the two per-run means
    assert report.faithfulness.std == 0.0  # identical runs => no run-to-run spread
    assert report.faithfulness.n == 2  # one datapoint per run, not per case


def test_inapplicable_metrics_are_excluded_from_averages():
    """A refusal has no citation coverage; it must not be averaged in as 0."""
    runs = [[_case(faith=None)]]
    assert aggregate(runs).faithfulness.n == 0


def test_empty_runs_produce_empty_report():
    report = aggregate([])
    assert report.runs == 0 and report.questions == 0


def test_missing_terms_row_is_not_duplicated():
    """Genuine and artifact missing_terms merge into one display row."""
    genuine = _case(qid="a", passed=False,
                    failures=[Failure(type=FailureType.MISSING_TERMS, detail="x")])
    artifact = _case(qid="b", passed=False, artifact=True,
                     failures=[Failure(type=FailureType.MISSING_TERMS, detail="y")])
    rows = summarize_failure_counts(aggregate([[genuine, artifact]]))
    missing_rows = [r for r in rows if r[0] == "missing_terms"]
    assert len(missing_rows) == 1
    assert missing_rows[0] == ("missing_terms", 1, 1)
