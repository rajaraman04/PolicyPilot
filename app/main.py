"""FastAPI entrypoint.

Run: uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.db import init_db
from app.schemas import QueryRequest, QueryResponse

app = FastAPI(title="PolicyPilot AI", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a policy question via the agentic RAG flow.

    TODO: invoke app.graph.build_graph() and map its state to QueryResponse.
    """
    raise NotImplementedError("Query flow wiring pending (see app/graph.py).")
