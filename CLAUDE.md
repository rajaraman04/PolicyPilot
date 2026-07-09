# PolicyPilot AI

## Why
A portfolio project targeting AI / GenAI / Agentic AI Engineer roles. It exists
to demonstrate three hireable signals: (1) working RAG, (2) rigorous evaluation
of that RAG, and (3) production sense — cost, latency, and failure handling.
The evaluation harness is the differentiator: treat it as the crown jewel, not
a side feature.

## What
An agentic RAG compliance assistant. A user asks a policy question; the system
retrieves evidence from public policy documents, checks that every claim is
grounded in retrieved text, and returns an Approved / Denied / Needs-More-Info
decision with citations and a confidence score.

## Scope (MVP — do not exceed)
- Multi-document RAG over public policy PDFs, with citations
- A 3-node agentic flow ONLY: Planner -> Retriever -> Verifier/Decision
- Evaluation harness: faithfulness / unsupported-claim rate, citation coverage,
  retrieval relevance, plus cost + latency per query
- Adversarial tests: a prompt-injection attempt and a no-evidence query
- One ablation: single-pass RAG vs. agentic-with-verifier
- Live deployment + a README with real measured numbers

Explicitly OUT of scope for v1 (park as "v2 roadmap"; do NOT build these):
- More than 3 agents (no separate risk agent, no separate report agent)
- Human-in-the-loop approval flow or decision-history database
- React frontend
- Analytics dashboard
- Postgres (use SQLite)

## Tech stack
- Python 3.11+
- FastAPI (backend / API)
- ChromaDB (vector store)
- LangGraph (agent orchestration; fall back to LangChain only if it fights)
- Streamlit (minimal UI)
- SQLite (storage)
- Docker (deployment)
- pytest (tests)

## Commands (intended layout — create as the project is built)
- Install deps:  `pip install -r requirements.txt`
- Run API:       `uvicorn app.main:app --reload`
- Run UI:        `streamlit run ui/app.py`
- Run evals:     `python eval/run_eval.py`
- Run tests:     `pytest`

## Conventions
- Keep everything scoped to the MVP above. If a request would exceed scope,
  flag it and suggest deferring it to v2 instead of building it.
- Every answer MUST cite the source chunks it used (document name + page).
- Never invent citations or state facts that are not in the retrieved text.
- Write tests for each eval metric as it is built.
- Prefer small, reviewable changes. Show a plan before editing files on any
  non-trivial task and wait for approval.
- Keep secrets (API keys) in a `.env` file that is gitignored. Never commit them.

## Working style
Follow Explore -> Plan -> Implement -> Commit. On non-trivial tasks, describe the
approach and wait for my approval before creating or modifying files.
