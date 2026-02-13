from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class OutputParser:
    """Parse LLM text output into validated Pydantic models.

    Replaces LangChain's ``PydanticOutputParser``.
    """

    @staticmethod
    def parse(text: str, model: type[T]) -> T:
        """Extract JSON from *text* and validate against *model*.

        Handles common LLM patterns:
        - Raw JSON objects
        - JSON wrapped in ```json ... ``` fences
        - JSON embedded in surrounding prose
        """
        # 1) Try fenced code blocks first
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fenced:
            return model.model_validate_json(fenced.group(1).strip())

        # 2) Try to find a raw JSON object
        # Find the first { and the last matching }
        start = text.find("{")
        if start != -1:
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = text[start:end]
            try:
                return model.model_validate_json(candidate)
            except Exception:
                pass

        # 3) Try the entire text as JSON
        try:
            return model.model_validate_json(text.strip())
        except Exception:
            pass

        # 4) Last resort — try to parse with json.loads and then validate
        try:
            data = json.loads(text)
            return model.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Could not parse LLM output into {model.__name__}.\n"
                f"Raw text (first 500 chars): {text[:500]}"
            ) from exc

    @staticmethod
    def format_instructions(model: type[BaseModel]) -> str:
        """Generate format instructions from a Pydantic model's JSON schema.

        Equivalent to LangChain's ``PydanticOutputParser.get_format_instructions()``.
        """
        schema = model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        return (
            "The output should be formatted as a JSON instance that conforms "
            "to the JSON schema below.\n\n"
            f"```json\n{schema_str}\n```\n\n"
            "Return ONLY the JSON object, no additional text."
        )
