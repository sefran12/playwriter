from __future__ import annotations

import json
import logging
from typing import TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from playwriter.llm.base import LLMProvider
from playwriter.parsing.output_parser import OutputParser

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic API."""

    STRONG_MODEL = "claude-opus-4-6"
    FAST_MODEL = "claude-haiku-4-5-20251001"

    MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        temperature: float = 0.2,
    ):
        super().__init__(model=model or self.STRONG_MODEL, temperature=temperature)
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        temp = min(temp, 1.0)
        log.info("Anthropic complete: model=%s, prompt_len=%d", self.model, len(user_prompt))
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            result = response.content[0].text
            log.info("Anthropic response: %d chars", len(result))
            return result
        except Exception as exc:
            log.error("Anthropic API error: %s", exc)
            raise

    async def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Use Anthropic tool-use to extract structured output."""
        log.info("Anthropic structured: model=%s, target=%s",
                 self.model, response_model.__name__)
        schema = response_model.model_json_schema()
        temp = temperature if temperature is not None else self.temperature
        temp = min(temp, 1.0)

        tool_name = "structured_output"
        try:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temp,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": tool_name,
                        "description": f"Return the result as a {response_model.__name__} object.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except Exception as exc:
            log.error("Anthropic API error: %s", exc)
            raise

        # Extract the tool-use block
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                log.info("Anthropic structured via tool_use OK")
                return response_model.model_validate(block.input)

        # Fallback: try to parse from text blocks
        log.warning("Anthropic: no tool_use block found, falling back to text parse")
        text_parts = [b.text for b in response.content if hasattr(b, "text")]
        raw = "\n".join(text_parts)
        return OutputParser.parse(raw, response_model)
