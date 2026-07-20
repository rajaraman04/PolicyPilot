"""Application settings, loaded from environment / .env.

Two independent provider choices:

  * EMBEDDINGS default to a LOCAL sentence-transformers model, so no API key is
    needed just to embed/ingest. Set EMBED_PROVIDER=openai to use OpenAI instead.
  * The answer-generation LLM provider is configurable via LLM_PROVIDER
    (openai | anthropic) and needs the matching API key.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Answer-generation LLM ---
    llm_provider: str = "openai"  # "openai" | "anthropic"
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_llm_model: str = "claude-sonnet-5"

    # --- Embeddings ---
    # "local" uses sentence-transformers (no key needed). "openai" uses the
    # OpenAI embeddings API (reuses openai_api_key).
    embed_provider: str = "local"  # "local" | "openai"
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embed_model: str = "text-embedding-3-small"

    # --- Chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval ---
    top_k: int = 5

    # --- Reproducibility ---
    # Seed passed to providers that support it (OpenAI). Best-effort: results
    # can still shift when the provider changes backend infra, which is why the
    # runner records system_fingerprint alongside results.
    llm_seed: int | None = 20240720

    # --- Pricing overrides (USD per 1M tokens) ---
    # Set these when the built-in table in app/pricing.py goes stale.
    price_input_per_1m: float | None = None
    price_output_per_1m: float | None = None

    # --- Storage / paths ---
    chroma_dir: str = "chroma_store"
    chroma_collection: str = "policypilot"
    sqlite_path: str = "policypilot.db"
    data_dir: str = "data"

    @property
    def embed_model(self) -> str:
        """The active embedding model name for the selected provider."""
        return self.local_embed_model if self.embed_provider == "local" else self.openai_embed_model


settings = Settings()
