# PolicyPilot AI

**Answer policy and compliance questions with cited, grounded evidence — not guesses.**
Ask a question in plain English ("What are the core functions of the cybersecurity
framework?"); PolicyPilot decomposes it, retrieves the relevant passages from your
policy documents, answers *only* from that retrieved text, verifies every citation,
and returns an **Approved / Denied / Needs-More-Info** decision with a confidence
score. If the documents don't support an answer, it says so instead of making one up.

The point: teams drowning in dense policy PDFs (NIST, internal compliance, regulatory
guidance) can get fast, **traceable** answers where every statement is auditable back
to a page number — the opposite of a chatbot that sounds confident and invents facts.

> **Evaluation is the headline deliverable, not an afterthought.** A gold set of 39
> labeled questions and a multi-run harness measure faithfulness, citation coverage,
> retrieval relevance, cost, and latency — and quantify what the agentic flow actually
> buys over plain RAG. See [Results / Evaluation](#results--evaluation).

## Architecture

A 3-node agentic flow, wired with LangGraph:

```
              ┌───────────┐     ┌──────────────┐     ┌────────────────────┐
 question ───▶│  Planner  │────▶│  Retriever   │────▶│ Verifier / Decision │──▶ decision
              │ decompose │     │  (ChromaDB)  │     │  verify + decide    │    + confidence
              └───────────┘     └──────────────┘     └────────────────────┘    + citations
```

- **Planner** — decomposes the question into sub-queries. Single-topic → one query;
  comparison / multi-part → one per topic, so cross-document questions get *balanced*
  evidence instead of a top-k dominated by whichever document matches more strongly.
- **Retriever** — embeds each sub-query with the *same* local model used at ingest
  time and pulls the top-k nearest chunks from ChromaDB under a shared budget, merged
  and deduped by `(source, page)`. Answer generation is constrained by a prompt that
  forbids outside knowledge and requires an inline `(filename, p.N)` citation per claim.
- **Verifier / Decision** — checks every citation against what was actually retrieved;
  any citation that wasn't retrieved is **fabricated** and gets marked `[unverified]`
  in place. It then derives the decision:
  - **Approved** — a grounded, cited answer the system stands behind.
  - **Needs-More-Info** — insufficient evidence or an ambiguous question (it declined).
  - **Denied** — an answer was produced but nothing in it is grounded (no valid citation).
  - **Confidence** = `mean(citation_coverage, verification_pass_rate)` — a documented
    heuristic signal, *not* a calibrated probability.

**Embeddings run locally** (sentence-transformers) so ingestion needs no API key.
Only the Planner and answer generation call a hosted LLM; the provider (OpenAI
default, Anthropic optional) is configurable in `.env`.

## Tech stack

Python 3.11+ · FastAPI · ChromaDB · sentence-transformers · LangChain / LangGraph ·
Streamlit · SQLite · pytest

## Install

```bash
pip install -r requirements.txt      # first run pulls PyTorch — sizeable download
cp .env.example .env                  # then edit .env (see below)
```

In `.env`:
- To **ingest / retrieve**: nothing required — embeddings default to a local model.
- To **generate answers**: set your LLM key, e.g.
  ```
  LLM_PROVIDER=openai
  OPENAI_API_KEY=sk-...
  ```
  (or `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`).

## Ingest documents

Drop policy PDFs into `data/`, then build the vector store:

```bash
python -m ingest.build            # add --reset to rebuild from scratch
```

It prints a per-document chunk count and a summary (PDFs, pages, chunks stored, path).

## Run

**API:**
```bash
uvicorn app.main:app --reload
```
- `GET  /`        — health check
- `POST /query`   — `{"question": "..."}` → decision + confidence + answer + citations + telemetry
- Interactive docs at http://localhost:8000/docs

**UI (second terminal):**
```bash
streamlit run ui/app.py           # opens http://localhost:8501
```

**Tests:**
```bash
pytest
```

**Evaluation harness:**
```bash
python eval/run_eval.py --runs 3              # single-pass baseline, mean ± spread
python eval/run_eval.py --plan --runs 3       # full agentic flow (Planner + Verifier)
python eval/run_eval.py --ablation --runs 3   # control vs +verifier, side by side
python eval/run_eval.py --plan --dry-run      # estimate cost, call nothing
```

## Results / Evaluation

Measured on a **gold set of 39 labeled questions × 3 runs**, `gpt-4o-mini`,
`top_k=10`, local `all-MiniLM-L6-v2` embeddings, temperature 0. All figures are
**real API calls**, reported as **mean ± run-to-run spread** (LLM output isn't fully
deterministic even at temperature 0, so single-run numbers would overstate precision).

### Ablation — what the agentic flow buys over plain RAG

| Configuration | Pass rate | Faithfulness | Citation coverage | Fabricated citations |
|---|---|---|---|---|
| Single-pass RAG (baseline) | 0.564 | 0.845 | 0.767 | 9 |
| + Verifier | 0.607 | 0.847 | 0.661 | **0** |
| **+ Planner + Verifier** | **~0.62** | **~0.93** | ~0.65 | **0** |

- **The Verifier eliminates fabricated citations (9 → 0)** — it's the guarantee that
  the system never shows a citation it can't back. The trade is lower citation
  *coverage*: a marked `[unverified]` claim no longer counts as cited. That
  coverage-for-integrity trade is measured, not hidden.
- **The Planner lifts faithfulness (~0.85 → ~0.93)** and reduces over-refusals by
  giving cross-document questions balanced evidence. Retrieval relevance holds at
  **0.958** throughout.

### Metrics

- **Faithfulness** — LLM-as-judge over a *deterministic* sentence split, so the
  scoring denominator can't drift between runs. Each sentence is `SUPPORTED`
  (directly stated), `DERIVED` (entailed by combining stated facts — valid synthesis,
  which comparison questions require), or `UNSUPPORTED` (a hallucination). Faithfulness
  counts supported + derived; `derived` (~8%) is reported separately.
- **Citation coverage** — rule-based: fraction of factual sentences carrying a
  citation whose `(document, page)` was actually retrieved.
- **Retrieval relevance** — did retrieval return the gold-labeled source document(s)?
- **Cost + latency** — per query, broken down by stage. ~**$0.0003 / query**;
  latency is dominated (~95%) by the LLM call, not embedding or retrieval.

### Adversarial robustness

10 prompt-injection cases (instruction override, fabricated-citation requests,
context-delimiter spoofing, role-play jailbreaks, base64-obfuscated and multilingual
payloads) plus 6 no-evidence questions. The system **leaked no system prompt, adopted
no injected persona, and minted no fabricated citation** across these; no-evidence and
adversarial requests are correctly declined as **Needs-More-Info**.

### Honest limitations

- **Cross-document synthesis is still hard.** The Planner improves answer *quality* on
  multi-document comparison questions (fewer refusals, higher faithfulness) but most
  still don't clear the strict all-terms pass bar — the remaining failures are
  retrieval misses and genuine content gaps, not judge artifacts. Reported as-is.
- **Confidence is a heuristic**, not a calibrated probability.
- **Figure-embedded content isn't retrievable** — text extraction (pypdf) can't read
  content that lives inside a PDF's figures; the gold set is scoped accordingly.

## Roadmap (v2 — out of scope for v1)

More than 3 agents · human-in-the-loop approval + decision-history DB · React
frontend · analytics dashboard · Postgres · OCR ingestion for figure-embedded content.
