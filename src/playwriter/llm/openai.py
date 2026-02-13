from __future__ import annotations

import json
import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from playwriter.llm.base import LLMProvider
from playwriter.parsing.output_parser import OutputParser

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """LLM provider backed by the OpenAI API."""

    STRONG_MODEL = "gpt-4.1"
    FAST_MODEL = "gpt-4.1-mini"

    MODELS = [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.2",
    ]

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        temperature: float = 0.2,
    ):
        super().__init__(model=model or self.STRONG_MODEL, temperature=temperature)
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        log.info("OpenAI complete: model=%s, json_mode=%s, prompt_len=%d",
                 self.model, json_mode, len(user_prompt))
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content or ""
            log.info("OpenAI response: %d chars", len(result))
            return result
        except Exception as exc:
            log.error("OpenAI API error: %s", exc)
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
        log.info("OpenAI structured: model=%s, target=%s",
                 self.model, response_model.__name__)
        schema = response_model.model_json_schema()
        augmented_system = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )
        raw = await self.complete(
            augmented_system,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            result = OutputParser.parse(raw, response_model)
            log.info("OpenAI structured parse OK: %s", response_model.__name__)
            return result
        except Exception as exc:
            log.error("OpenAI structured parse FAILED: %s — raw[:300]=%s",
                      exc, raw[:300])
            raise
