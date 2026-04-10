"""
llm/factory.py — singleton factory for the configured LLM provider.

Reads BUBBLE_LLM_PROVIDER (default: "anthropic") and returns the matching client.
The client is instantiated once and reused for the lifetime of the process.
"""

from .. import config
from .base import LLMClient

_instance: LLMClient | None = None


def get_llm() -> LLMClient:
    """Return the process-wide LLM client, creating it on first call."""
    global _instance
    if _instance is not None:
        return _instance

    provider = config.LLM_PROVIDER

    if provider == "anthropic":
        from .anthropic import AnthropicLLMClient
        _instance = AnthropicLLMClient()

    elif provider == "openai":
        try:
            from .openai import OpenAILLMClient
        except ImportError as e:
            raise ImportError(
                "openai package is required for provider='openai'. "
                "Install it with: pip install 'bubble-memory[openai]'"
            ) from e
        _instance = OpenAILLMClient()

    elif provider == "gemini":
        try:
            from .gemini import GeminiLLMClient
        except ImportError as e:
            raise ImportError(
                "google-generativeai package is required for provider='gemini'. "
                "Install it with: pip install 'bubble-memory[gemini]'"
            ) from e
        _instance = GeminiLLMClient()

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Set BUBBLE_LLM_PROVIDER to 'anthropic', 'openai', or 'gemini'."
        )

    return _instance
