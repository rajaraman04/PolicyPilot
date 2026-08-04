"""FastAPI entrypoint.

Run: uvicorn app.main:app --reload
Interactive docs: http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI, HTTPException

from app.db import init_db
from app.graph import answer_with_decision
from app.rag import warmup
from app.schemas import QueryRequest, QueryResponse

app = FastAPI(title="PolicyPilot AI", version="0.1.0")
logger = logging.getLogger("uvicorn")


@app.on_event("startup")
def _startup() -> None:
    init_db()
    # Load the embedding model now so the first /query isn't charged the
    # one-time model-load latency. Never let warm-up block the server booting.
    try:
        warmup()
        logger.info("Warm-up complete: embedding model loaded.")
    except Exception as exc:  # noqa: BLE001 - boot even if the store isn't built
        logger.warning("Warm-up skipped (%s). First query may be slower.", exc)


@app.get("/")
def root() -> dict[str, str]:
    """Root health check."""
    return {"status": "ok", "service": "PolicyPilot AI", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a policy question via the agentic flow (Planner -> Retriever ->
    Verifier/Decision).

    Returns an Approved / Denied / Needs-More-Info decision with a confidence
    score and citations. The answer is constrained to the retrieved context;
    fabricated citations are marked [unverified].
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        return answer_with_decision(req.question)
    except ValueError as exc:
        # e.g. missing API key for the configured provider
        raise HTTPException(status_code=503, detail=str(exc)) from exc
