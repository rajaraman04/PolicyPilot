"""Evaluation harness entrypoint.

Run: python eval/run_eval.py

Runs the eval dataset (plus adversarial cases: prompt-injection, no-evidence)
through the pipeline, computes metrics from eval/metrics.py, records cost +
latency, and writes results to eval/results/. Also supports the single-pass RAG
vs. agentic-with-verifier ablation.
"""


def main() -> None:
    """Load eval/datasets, run each case, compute metrics, write eval/results.

    TODO:
      - Load eval questions + adversarial cases from eval/datasets/.
      - For each: run the pipeline, capture answer/citations/cost/latency.
      - Score with eval.metrics; aggregate.
      - Ablation: run both single-pass and agentic-with-verifier modes.
      - Write a results file to eval/results/ and print a summary.
    """
    raise NotImplementedError("Eval harness pending.")


if __name__ == "__main__":
    main()
