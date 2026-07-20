"""Chat-LLM factory.

Returns a LangChain chat model for the configured LLM_PROVIDER. Temperature is
pinned to 0 for deterministic, faithful answers (important for eval).
"""

from app.config import settings


def get_llm(seed: int | None = None):
    """Build the chat model for the configured provider (openai | anthropic).

    ``seed`` is opt-in so the app keeps its default behaviour while the eval
    harness can request a reproducible client. Only providers that support a
    seed receive it.
    """
    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY in .env")
        from langchain_openai import ChatOpenAI

        extra = {"seed": seed} if seed is not None else {}
        return ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
            **extra,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY in .env")
        from langchain_anthropic import ChatAnthropic

        # Anthropic exposes no seed parameter; reproducibility rests on
        # temperature=0 alone. Ignored rather than erroring so provider swaps
        # don't break the harness.
        return ChatAnthropic(
            model=settings.anthropic_llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r} (use 'openai' or 'anthropic')")
