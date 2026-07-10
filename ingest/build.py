"""Build the vector store from source PDFs.

Load every PDF in the data directory, extract text page by page, split it into
chunks (respecting the configured chunk size / overlap), embed each chunk, and
store it in a local persistent ChromaDB collection. Every chunk carries
{source, page} metadata so retrieval can return citations.

Run:
    python -m ingest.build
    python -m ingest.build --data-dir data --chunk-size 800 --reset
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.embeddings import get_embeddings

# Chroma rejects batches above a backend-dependent limit; stay well under it.
_ADD_BATCH_SIZE = 100


def _iter_pdf_pages(pdf_path: Path):
    """Yield (page_number, text) for each page of a PDF (1-indexed)."""
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield page_number, text


def _chunk_id(source: str, page: int, index: int, text: str) -> str:
    """Stable, unique id so re-ingesting identical content upserts rather than duplicates."""
    digest = hashlib.sha1(f"{source}|{page}|{index}|{text}".encode()).hexdigest()[:16]
    return f"{source}:p{page}:c{index}:{digest}"


def build(
    data_dir: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reset: bool = False,
) -> int:
    """Ingest all PDFs under ``data_dir`` into the Chroma collection.

    Returns the number of chunks stored.
    """
    data_path = Path(data_dir or settings.data_dir)
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    pdf_paths = sorted(data_path.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {data_path.resolve()} — add source documents and re-run.")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Collect chunks with metadata across all documents.
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    pages_seen = 0

    for pdf_path in pdf_paths:
        source = pdf_path.name
        doc_chunks = 0
        for page_number, page_text in _iter_pdf_pages(pdf_path):
            pages_seen += 1
            for index, chunk in enumerate(splitter.split_text(page_text)):
                ids.append(_chunk_id(source, page_number, index, chunk))
                documents.append(chunk)
                metadatas.append({"source": source, "page": page_number})
                doc_chunks += 1
        print(f"  {source}: {doc_chunks} chunks")

    if not documents:
        print("PDFs found but no extractable text (scanned images?). Nothing stored.")
        return 0

    # Embed.
    print(f"Embedding {len(documents)} chunks with {settings.embed_provider} "
          f"({settings.embed_model})...")
    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(documents)

    # Store in a persistent Chroma collection.
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
            print(f"Reset: dropped existing collection '{settings.chroma_collection}'.")
        except Exception:
            pass  # collection did not exist yet
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(ids), _ADD_BATCH_SIZE):
        end = start + _ADD_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    stored = collection.count()
    print(
        "\nDone.\n"
        f"  PDFs ingested : {len(pdf_paths)}\n"
        f"  Pages parsed  : {pages_seen}\n"
        f"  Chunks stored : {len(ids)}\n"
        f"  Collection    : '{settings.chroma_collection}' (total items: {stored})\n"
        f"  Persisted to  : {Path(settings.chroma_dir).resolve()}"
    )
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the PolicyPilot vector store from PDFs.")
    parser.add_argument("--data-dir", default=None, help="Folder of source PDFs (default: config DATA_DIR)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk overlap")
    parser.add_argument("--reset", action="store_true", help="Drop the collection before ingesting")
    args = parser.parse_args()

    build(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
