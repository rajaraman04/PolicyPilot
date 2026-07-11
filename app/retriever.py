"""ChromaDB retrieval wrapper.

Thin layer over the persisted Chroma collection so the agent nodes and the
eval harness share one retrieval path. Queries are embedded with the same
factory used at ingest time, so query and chunk vectors are comparable.
"""

import chromadb

from app.config import settings
from app.embeddings import get_embeddings
from app.schemas import Citation


class Retriever:
    """Retrieves top-k relevant chunks and returns them as Citations.

    Client, collection, and embedding model are created lazily on first use so
    importing this module stays cheap and does not require a built store.
    """

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k or settings.top_k
        self._collection = None
        self._embeddings = None

    def _ensure_ready(self) -> None:
        if self._collection is None:
            client = chromadb.PersistentClient(path=settings.chroma_dir)
            # Raises if the collection was never built — surfaces a clear error.
            self._collection = client.get_collection(settings.chroma_collection)
        if self._embeddings is None:
            self._embeddings = get_embeddings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[Citation]:
        """Return the top-k relevant chunks for a query, with citation metadata."""
        if not query or not query.strip():
            return []

        self._ensure_ready()
        k = top_k or self.top_k

        vector = self._embeddings.embed_query(query)
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma nests results one level per query; we sent a single query.
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        citations: list[Citation] = []
        for text, meta in zip(documents, metadatas):
            meta = meta or {}
            citations.append(
                Citation(
                    document=meta.get("source", "unknown"),
                    page=int(meta.get("page", 0)),
                    snippet=text or "",
                )
            )
        return citations


def retrieve(query: str, top_k: int | None = None) -> list[Citation]:
    """Convenience wrapper for one-off retrieval."""
    return Retriever(top_k=top_k).retrieve(query)
