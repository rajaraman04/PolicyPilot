"""Pydantic request/response models for the API."""

from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    APPROVED = "Approved"
    DENIED = "Denied"
    NEEDS_MORE_INFO = "Needs-More-Info"


class Citation(BaseModel):
    """A pointer to the source chunk an answer relied on."""

    document: str = Field(..., description="Source document name")
    page: int = Field(..., description="Page number within the document")
    snippet: str = Field("", description="The retrieved text used")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    decision: Decision
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[Citation] = []
    # Production-sense telemetry
    latency_ms: float | None = None
    cost_usd: float | None = None


class AnswerResponse(BaseModel):
    """Response for the single-pass RAG /query endpoint (no verifier yet)."""

    question: str
    answer: str
    sources: list[Citation] = []
    latency_ms: float | None = None
