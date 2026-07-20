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


class LatencyBreakdown(BaseModel):
    """Per-stage latency (ms) so we can see which stage dominates a query."""

    embed_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


class TokenUsage(BaseModel):
    """Tokens consumed by the answer-generation call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AnswerResponse(BaseModel):
    """Response for the single-pass RAG /query endpoint (no verifier yet)."""

    question: str
    answer: str
    sources: list[Citation] = []
    latency_ms: float | None = None
    timings: LatencyBreakdown | None = None

    # Product cost/telemetry (excludes any eval-judge calls).
    usage: TokenUsage | None = None
    cost_usd: float | None = None
    model: str | None = None
    # Provider backend identifier; when this changes, seeded runs may drift.
    system_fingerprint: str | None = None
