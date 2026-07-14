"""ChromaDB retrieval wrapper.

Thin layer over the persisted Chroma collection so the agent nodes and the
eval harness share one retrieval path. Queries are embedded with the same
factory used at ingest time, so query and chunk vectors are comparable.
"""

import time

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
        # Load the embedding model first so warm-up can heat it even if the
        # collection hasn't been built yet (get_collection would raise).
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        if self._collection is None:
            client = chromadb.PersistentClient(path=settings.chroma_dir)
            # Raises if the collection was never built — surfaces a clear error.
            self._collection = client.get_collection(settings.chroma_collection)

    def warmup(self) -> None:
        """Load the embedding model into memory ahead of the first request.

        Runs one real forward pass so weights are fully initialized. Called from
        the API startup hook so the first /query isn't charged the model-load time.
        """
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        self._embeddings.embed_query("warmup")
        self._ensure_ready()

    def retrieve(self, query: str, top_k: int | None = None) -> list[Citation]:
        """Return the top-k relevant chunks for a query, with citation metadata."""
        citations, _ = self.retrieve_timed(query, top_k=top_k)
        return citations

    def retrieve_timed(
        self, query: str, top_k: int | None = None
    ) -> tuple[list[Citation], dict[str, float]]:
        """Like ``retrieve`` but also returns a timing breakdown (ms) for the
        embedding step and the Chroma query step, so callers can see where
        retrieval latency goes.
        """
        if not query or not query.strip():
            return [], {"embed_ms": 0.0, "retrieval_ms": 0.0}

        self._ensure_ready()
        k = top_k or self.top_k

        t0 = time.perf_counter()
        vector = self._embeddings.embed_query(query)
        t1 = time.perf_counter()
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        t2 = time.perf_counter()

        timings = {
            "embed_ms": round((t1 - t0) * 1000, 1),
            "retrieval_ms": round((t2 - t1) * 1000, 1),
        }
        return self._to_citations(result), timings

    @staticmethod
    def _to_citations(result: dict) -> list[Citation]:
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
