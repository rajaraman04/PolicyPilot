"""ChromaDB retrieval wrapper.

Thin layer over the persisted Chroma collection so the agent nodes and the
eval harness share one retrieval path.
"""

from app.config import settings
from app.schemas import Citation


class Retriever:
    """Retrieves top-k relevant chunks and returns them as Citations."""

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k or settings.top_k
        # TODO: open persisted Chroma collection at settings.chroma_dir

    def retrieve(self, query: str) -> list[Citation]:
        """Return the top-k relevant chunks for a query."""
        raise NotImplementedError("Retrieval wiring pending (see ingest/ingest.py).")
