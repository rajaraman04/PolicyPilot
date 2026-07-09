"""Application settings, loaded from environment / .env.

Default provider is OpenAI. Set LLM_PROVIDER=anthropic to switch the LLM
(embeddings stay on OpenAI — Anthropic has no embeddings API).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider selection
    llm_provider: str = "openai"  # "openai" | "anthropic"

    # OpenAI
    openai_api_key: str = ""
    openai_llm_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_llm_model: str = "claude-sonnet-5"

    # Storage / paths
    chroma_dir: str = "chroma_store"
    chroma_collection: str = "policypilot"
    sqlite_path: str = "policypilot.db"
    data_dir: str = "data"

    # Retrieval
    top_k: int = 5


settings = Settings()
