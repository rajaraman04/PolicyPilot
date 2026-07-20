"""Run the metrics against selected gold-set entries, with real LLM calls.

A scratch/inspection tool for eyeballing individual cases — `run_eval.py` is the
real harness that scores the whole set and writes results.

Usage:
    python -m eval.demo_metrics                 # default sample: one per behavior type
    python -m eval.demo_metrics q001 q036       # specific ids

Costs a few cents per case (one RAG call + judge calls). Requires an LLM key in .env.
"""

import sys

from app.rag import answer_question, warmup
from eval.gold_set import load_gold_set
from eval.metrics import evaluate_case

DEFAULT_IDS = ["q001", "q022", "q031", "q036"]


def main() -> None:
    ids = sys.argv[1:] or DEFAULT_IDS

    warmup()
    gold = {q.id: q for q in load_gold_set().questions}

    unknown = [i for i in ids if i not in gold]
    if unknown:
        raise SystemExit(f"unknown gold-set id(s): {unknown}")

    for qid in ids:
        q = gold[qid]
        print("=" * 100)
        print(f"{q.id}  [{q.category.value} / {q.expected_behavior.value}]")
        print(f"Q: {q.question[:150]}")

        resp = answer_question(q.question)
        print(f"\nANSWER: {resp.answer[:400]}")
        print(f"SOURCES: {[(c.document, c.page) for c in resp.sources]}")

        res = evaluate_case(q, resp.answer, resp.sources)

        print(f"\n  PASSED: {res.passed}")
        for f in res.failures:
            print(f"    FAIL: {f}")
        if res.refused is not None:
            print(f"  refused............: {res.refused}")
        if res.faithfulness and res.faithfulness.applicable:
            print(f"  faithfulness.......: {res.faithfulness.score}  "
                  f"(unsupported: {res.faithfulness.unsupported_claims or 'none'})")
        if res.citation_coverage and res.citation_coverage.applicable:
            cc = res.citation_coverage
            print(f"  citation_coverage..: {cc.score}  "
                  f"({cc.cited_sentences}/{cc.total_sentences} sentences cited, "
                  f"fabricated: {cc.fabricated_citations or 'none'})")
        if res.retrieval_relevance and res.retrieval_relevance.applicable:
            rr = res.retrieval_relevance
            print(f"  retrieval_relevance: {rr.score}  "
                  f"(expected {rr.expected_docs}, missing {rr.missing_docs or 'none'})")
        print(f"  latency: {resp.latency_ms} ms")


if __name__ == "__main__":
    main()
