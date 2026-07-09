"""Ingest pipeline: PDFs in data/ -> chunks -> embeddings -> ChromaDB.

Run: python -m ingest.ingest

Each chunk keeps its source metadata (document name + page) so retrieval can
return citations.
"""

from app.config import settings


def ingest() -> None:
    """Load PDFs from settings.data_dir, chunk, embed, and persist to Chroma.

    TODO:
      1. Load each PDF in settings.data_dir (pypdf), tracking page numbers.
      2. Chunk text, attaching {document, page} metadata to each chunk.
      3. Embed with the configured embedding model.
      4. Upsert into the Chroma collection at settings.chroma_dir.
    """
    raise NotImplementedError("Ingest pipeline pending.")


if __name__ == "__main__":
    ingest()
