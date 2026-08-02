"""Tests for the ablation comparison helper (pure, no pipeline)."""

from eval.aggregate import MetricStat
from eval.run_eval import _ablation_rows, _fabricated_count


def _report(pass_rate, faith, cov, rel, fabricated=0):
    from eval.aggregate import AggregateReport
    return AggregateReport(
        runs=1,
        questions=10,
        pass_rate=MetricStat.from_values([pass_rate]),
        faithfulness=MetricStat.from_values([faith]),
        citation_coverage=MetricStat.from_values([cov]),
        retrieval_relevance=MetricStat.from_values([rel]),
        genuine_failures={"fabricated_citations": fabricated} if fabricated else {},
    )


def test_fabricated_count_reads_from_genuine_failures():
    assert _fabricated_count(_report(0.5, 0.8, 0.7, 0.9, fabricated=6)) == 6
    assert _fabricated_count(_report(0.5, 0.8, 0.7, 0.9)) == 0


def test_ablation_rows_compute_deltas():
    control = _report(0.56, 0.85, 0.80, 0.96, fabricated=10)
    verifier = _report(0.62, 0.85, 0.76, 0.96, fabricated=0)
    rows = {r[0]: r for r in _ablation_rows(control, verifier)}

    # pass rate improved
    assert rows["pass rate"][1] == "0.560" and rows["pass rate"][2] == "0.620"
    assert rows["pass rate"][3] == "+0.060"
    # fabricated citations eliminated (integer delta)
    assert rows["fabricated citations"][1] == "10"
    assert rows["fabricated citations"][2] == "0"
    assert rows["fabricated citations"][3] == "-10"
    # citation coverage dipped (marked claims read as uncited)
    assert rows["citation coverage"][3] == "-0.040"
    # faithfulness unchanged
    assert rows["faithfulness"][3] == "+0.000"


def test_ablation_rows_handle_missing_metric():
    """A metric with no data renders n/a and an empty delta, not a crash."""
    control = _report(0.5, 0.8, 0.7, 0.9)
    verifier = _report(0.5, 0.8, 0.7, 0.9)
    # blank out one metric
    control.retrieval_relevance = MetricStat.from_values([])
    rows = {r[0]: r for r in _ablation_rows(control, verifier)}
    assert rows["retrieval relevance"][1] == "n/a"
    assert rows["retrieval relevance"][3] == ""
