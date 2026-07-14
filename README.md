# PolicyPilot AI

**Answer policy and compliance questions with cited, grounded evidence — not guesses.**
Ask a question in plain English ("What are the core functions of the cybersecurity
framework?"); PolicyPilot retrieves the relevant passages from your policy documents,
answers *only* from that retrieved text, and cites every claim back to its source
document and page. If the documents don't support an answer, it says so instead of
making one up.

The point: teams drowning in dense policy PDFs (NIST, internal compliance, regulatory
guidance) can get fast, **traceable** answers where every statement is auditable back
to a page number — the opposite of a chatbot that sounds confident and invents facts.

> **Evaluation is a first-class part of this project, not an afterthought.** The
> harness that measures faithfulness, citation coverage, retrieval relevance, cost,
> and latency is the headline deliverable — see [Results / Evaluation](#results--evaluation).

## Architecture

```
                 ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  question ─────▶│   Planner   │────▶│   Retriever  │────▶│ Verifier /       │────▶ decision
                 │  (Week 2)   │     │  (ChromaDB)  │     │ Decision (Wk 2)  │      + citations
                 └─────────────┘     └──────────────┘     └─────────────────┘
```

- **Ingestion** — PDFs are parsed page-by-page, chunked, embedded, and stored in a
  local ChromaDB collection with `{source, page}` metadata on every chunk.
- **Retrieval** — the question is embedded with the *same* model used at ingest time;
  the top-k nearest chunks come back carrying their citations.
- **Answering** — retrieved chunks are passed to the LLM under a prompt that forbids
  outside knowledge and requires an inline `(filename, p.N)` citation for every claim.
- **Agentic flow** (Planner → Retriever → Verifier/Decision) and the **evaluation
  harness** are the Week 2 build; today's `/query` runs the single-pass RAG baseline.

**Embeddings run locally** (sentence-transformers) so ingestion needs no API key.
Only answer generation calls a hosted LLM; the provider (OpenAI default, Anthropic
optional) is configurable in `.env`.

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
- `GET  /health`  — health check
- `POST /query`   — `{"question": "..."}` → grounded answer + cited sources + per-stage timing
- Interactive docs at http://localhost:8000/docs

**UI (in a second terminal):**
```bash
streamlit run ui/app.py           # opens http://localhost:8501
```

**Tests:**
```bash
pytest
```

## Results / Evaluation

> **Coming in Week 2 (v2).** This section will hold real, measured numbers from the
> evaluation harness:
>
> - **Faithfulness / unsupported-claim rate** — are answers grounded in retrieved text?
> - **Citation coverage** — does every claim cite a source?
> - **Retrieval relevance** — did we retrieve the right chunks?
> - **Cost + latency** — per-query, broken down by stage (embedding / retrieval / LLM).
> - **Adversarial tests** — prompt-injection attempt and a no-evidence query.
> - **Ablation** — single-pass RAG vs. agentic-with-verifier.
>
> The numbers reported here will come from real API calls, not estimates.

## Roadmap (v2 — out of scope for v1)

More than 3 agents · human-in-the-loop approval + decision-history DB · React
frontend · analytics dashboard · Postgres.
