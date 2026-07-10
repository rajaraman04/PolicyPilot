# PolicyPilot AI

An agentic RAG compliance assistant. Ask a policy question; PolicyPilot retrieves
evidence from public policy documents, verifies that every claim is grounded in
the retrieved text, and returns an **Approved / Denied / Needs-More-Info** decision
with citations and a confidence score.

> The evaluation harness is the point of this project. See [`eval/`](eval/).

## Architecture

A 3-node agentic flow (LangGraph):

```
Planner  ->  Retriever  ->  Verifier / Decision
```

- **Planner** — interprets the question and plans what evidence to retrieve.
- **Retriever** — pulls the top-k relevant chunks from ChromaDB (with citations).
- **Verifier / Decision** — checks each claim against retrieved text and emits the
  Approved / Denied / Needs-More-Info decision + confidence.

## Tech stack

Python 3.11+ · FastAPI · ChromaDB · LangGraph · Streamlit · SQLite · Docker · pytest

Default LLM/embeddings provider is **OpenAI**; set `LLM_PROVIDER=anthropic` in `.env`
to switch the LLM to Anthropic (embeddings remain OpenAI).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your API key(s)
```

Add source PDFs to `data/`, then build the vector store:

```bash
python -m ingest.build            # add --reset to rebuild from scratch
```

## Running

```bash
uvicorn app.main:app --reload     # API
streamlit run ui/app.py           # UI
python eval/run_eval.py           # evaluation harness
pytest                            # tests
```

## Evaluation

The harness measures, per query:

- **Faithfulness / unsupported-claim rate** — are claims grounded in retrieved text?
- **Citation coverage** — does every answer cite its sources?
- **Retrieval relevance** — did we retrieve the right chunks?
- **Cost + latency** — production sense.

Plus adversarial tests (prompt-injection, no-evidence query) and one ablation:
single-pass RAG vs. agentic-with-verifier.

> _Measured numbers will be filled in here once the harness runs against real data._

## Roadmap (v2 — out of scope for v1)

More than 3 agents · human-in-the-loop approval + decision-history DB · React
frontend · analytics dashboard · Postgres.
