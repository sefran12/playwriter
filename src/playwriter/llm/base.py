from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base for all LLM providers (OpenAI, Anthropic, Groq)."""

    def __init__(self, model: str, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Return a plain-text completion."""

    @abstractmethod
    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Return a validated Pydantic model parsed from the LLM output."""
