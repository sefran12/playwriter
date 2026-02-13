from __future__ import annotations

import logging

from playwriter.config import settings
from playwriter.llm.base import LLMProvider
from playwriter.llm.openai import OpenAIProvider
from playwriter.llm.anthropic import AnthropicProvider
from playwriter.llm.groq import GroqProvider

log = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "openai": (OpenAIProvider, "openai_api_key"),
    "anthropic": (AnthropicProvider, "anthropic_api_key"),
    "groq": (GroqProvider, "groq_api_key"),
}


def get_provider(
    name: str | None = None,
    tier: str = "strong",
    model: str | None = None,
    temperature: float | None = None,
) -> LLMProvider:
    """Instantiate an LLM provider by name, tier, and optional explicit model.

    Parameters
    ----------
    name:
        ``"openai"`` | ``"anthropic"`` | ``"groq"``.
        Defaults to ``settings.default_provider``.
    tier:
        ``"strong"`` for complex generation, ``"fast"`` for parsing / cheap ops.
        Ignored when *model* is given explicitly.
    model:
        Explicit model id (e.g. ``"gpt-4.1-mini"``).  Overrides the tier default.
    temperature:
        Override; falls back to the settings default for the chosen tier.
    """
    name = name or settings.default_provider
    if name not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{name}'. Choose from: {list(_PROVIDER_MAP)}"
        )

    cls, key_attr = _PROVIDER_MAP[name]
    api_key = getattr(settings, key_attr)
    if not api_key:
        raise ValueError(
            f"API key for provider '{name}' is not configured "
            f"(set {key_attr.upper()} in .env)."
        )

    if model:
        chosen_model = model
    else:
        chosen_model = cls.STRONG_MODEL if tier == "strong" else cls.FAST_MODEL

    temp = temperature or (
        settings.default_strong_temperature
        if tier == "strong"
        else settings.default_fast_temperature
    )

    log.info("Creating %s provider: model=%s, temperature=%.2f", name, chosen_model, temp)
    return cls(api_key=api_key, model=chosen_model, temperature=temp)


def list_providers() -> dict:
    """Return info about all configured providers including all available models."""
    result = {}
    for name, (cls, key_attr) in _PROVIDER_MAP.items():
        api_key = getattr(settings, key_attr)
        result[name] = {
            "configured": bool(api_key),
            "strong_model": cls.STRONG_MODEL,
            "fast_model": cls.FAST_MODEL,
            "models": cls.MODELS,
            "is_default": name == settings.default_provider,
        }
    return result
