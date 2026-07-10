"""Embedding-model factory.

Returns a LangChain embeddings object for the provider selected in config.
Shared by the ingest pipeline and the retriever so both embed the same way.
"""

from app.config import settings


def get_embeddings():
    """Build the embeddings client for the configured EMBED_PROVIDER.

    * local  -> sentence-transformers (no API key required)
    * openai -> OpenAI embeddings API (reuses OPENAI_API_KEY)
    """
    provider = settings.embed_provider.lower()

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.local_embed_model)

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("EMBED_PROVIDER=openai requires OPENAI_API_KEY in .env")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.openai_embed_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"Unknown EMBED_PROVIDER: {settings.embed_provider!r} (use 'local' or 'openai')")
