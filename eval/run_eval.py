"""Evaluation harness entrypoint.

Run:
    python eval/run_eval.py                    # one pass over the gold set
    python eval/run_eval.py --runs 3           # repeated, reports mean +/- spread
    python eval/run_eval.py --category no_evidence
    python eval/run_eval.py --ids q001 q036
    python eval/run_eval.py --dry-run          # estimate cost, call nothing

Writes a JSON result file to eval/results/ and prints a human summary.

Product cost (the RAG call) and judge cost (eval overhead) are reported
separately: folding the judge into cost-per-query would overstate what running
PolicyPilot actually costs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.pricing import estimate_cost_usd
from app.rag import answer_question, warmup
from eval.aggregate import aggregate, attach_telemetry, summarize_failure_counts
from eval.gold_set import Category, load_gold_set
from eval.judge import get_judge_usage, reset_judge_usage
from eval.metrics import evaluate_case

RESULTS_DIR = Path(__file__).parent / "results"

# Rough per-case call budget for --dry-run: 1 answer call + up to 2 judge calls.
_EST_TOKENS_PER_CASE = {"input": 3500, "output": 350}


def _select(questions, ids, category, limit):
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q.id in wanted]
        missing = wanted - {q.id for q in questions}
        if missing:
            raise SystemExit(f"unknown gold-set id(s): {sorted(missing)}")
    if category:
        questions = [q for q in questions if q.category == Category(category)]
    if limit:
        questions = questions[:limit]
    return questions


def _dry_run(questions, runs) -> None:
    cases = len(questions) * runs
    est = estimate_cost_usd(
        settings.openai_llm_model,
        _EST_TOKENS_PER_CASE["input"] * cases,
        _EST_TOKENS_PER_CASE["output"] * cases,
    )
    print(f"DRY RUN — nothing was called.")
    print(f"  questions      : {len(questions)}")
    print(f"  runs           : {runs}")
    print(f"  total cases    : {cases}")
    print(f"  rough estimate : "
          + (f"${est:.4f}" if est is not None else "unknown (model not in pricing table)"))
    print("  (order-of-magnitude only; assumes ~3.5k in / ~350 out tokens per case)")


def _print_summary(report, judge_cost, judge_usage, elapsed_s) -> None:
    print()
    print("=" * 78)
    print(f"EVALUATION SUMMARY  ({report.questions} questions x {report.runs} run(s))")
    print("=" * 78)
    print(f"  pass rate ............ {report.pass_rate.render()}")
    print(f"  faithfulness ......... {report.faithfulness.render()}")
    print(f"  citation coverage .... {report.citation_coverage.render()}")
    print(f"  retrieval relevance .. {report.retrieval_relevance.render()}")

    print()
    print("FAILURES BY TYPE" + " " * 24 + "GENUINE   LABEL-ARTIFACT")
    rows = summarize_failure_counts(report)
    if not rows:
        print("  (none)")
    for ft, genuine, artifact in rows:
        print(f"  {ft:<36}{genuine:>7}   {artifact:>14}")

    if report.flaky_questions:
        print()
        print("FLAKY (passed in some runs, failed in others)")
        for q in report.flaky_questions:
            print(f"  {q.id:<8} {q.passes}/{q.runs} passed   {', '.join(q.failure_types)}")
    elif report.runs > 1:
        print()
        print("FLAKY: none — every question was stable across all runs.")

    print()
    print("LATENCY (ms per query)")
    print(f"  total ....... {report.latency_ms.render()}")
    print(f"  embedding ... {report.embed_ms.render()}")
    print(f"  retrieval ... {report.retrieval_ms.render()}")
    print(f"  llm ......... {report.llm_ms.render()}")

    print()
    print("COST (USD, estimated)")
    print(f"  product, per query ... {report.cost_usd_per_query.render(precision=6)}")
    total_product = (report.cost_usd_per_query.mean or 0) * report.questions * report.runs
    print(f"  product, this run .... ${total_product:.4f}")
    print(f"  judge overhead ....... "
          + (f"${judge_cost:.4f}" if judge_cost is not None else "unknown")
          + f"  ({judge_usage['calls']} calls)")
    print(f"  wall clock ........... {elapsed_s:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PolicyPilot evaluation harness.")
    parser.add_argument("--runs", type=int, default=1,
                        help="repeat the whole set N times and report mean +/- spread")
    parser.add_argument("--ids", nargs="*", help="only these gold-set ids")
    parser.add_argument("--category", help="only this category (e.g. no_evidence)")
    parser.add_argument("--limit", type=int, help="cap the number of questions")
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate cost and exit without calling anything")
    parser.add_argument("--out", help="explicit output path for the JSON results file")
    args = parser.parse_args()

    questions = _select(load_gold_set().questions, args.ids, args.category, args.limit)
    if not questions:
        raise SystemExit("no questions selected")

    if args.dry_run:
        _dry_run(questions, args.runs)
        return

    import time

    started = time.perf_counter()
    warmup()
    reset_judge_usage()

    all_runs, responses = [], []
    for run_no in range(1, args.runs + 1):
        results = []
        for i, q in enumerate(questions, start=1):
            print(f"\rrun {run_no}/{args.runs}  case {i}/{len(questions)}  {q.id}   ",
                  end="", flush=True)
            resp = answer_question(q.question)
            responses.append(resp)
            results.append(evaluate_case(q, resp.answer, resp.sources))
        all_runs.append(results)
    print()

    report = attach_telemetry(aggregate(all_runs), responses)

    judge_usage = get_judge_usage()
    judge_cost = estimate_cost_usd(
        settings.openai_llm_model,
        judge_usage["input_tokens"],
        judge_usage["output_tokens"],
    )
    elapsed = time.perf_counter() - started
    _print_summary(report, judge_cost, judge_usage, elapsed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"eval_{stamp}.json"
    payload = {
        "timestamp_utc": stamp,
        "runs": args.runs,
        "questions": len(questions),
        "model": responses[0].model if responses else None,
        "system_fingerprint": responses[0].system_fingerprint if responses else None,
        "llm_seed": settings.llm_seed,
        "top_k": settings.top_k,
        "embed_model": settings.embed_model,
        "judge_usage": judge_usage,
        "judge_cost_usd": judge_cost,
        "elapsed_s": round(elapsed, 2),
        "report": report.model_dump(),
        "cases": [[c.model_dump() for c in run] for run in all_runs],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
