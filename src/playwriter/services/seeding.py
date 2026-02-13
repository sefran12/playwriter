from __future__ import annotations

from playwriter.llm.base import LLMProvider
from playwriter.models.story import TCCN, CharacterSummary, NarrativeThread
from playwriter.parsing.output_parser import OutputParser
from playwriter.prompts.loader import PromptLoader


class SeedingService:
    """Generate a TCCN story-seed from a brief description.

    Pipeline:
    1. seed_description  →  INITIAL_HISTORY_TCC_GENERATOR  →  raw TCCN text
    2. raw text  →  structured parse  →  TCCN model
    3. (optional) CHARACTER_LIST_PARSER  →  list[CharacterSummary]
    """

    def __init__(self, llm: LLMProvider, prompts: PromptLoader | None = None):
        self._llm = llm
        self._prompts = prompts or PromptLoader()

    async def generate_seed(self, seed_description: str) -> TCCN:
        """Generate a full TCCN from a seed description."""
        prompt = self._prompts.render(
            "generators",
            "INITIAL_HISTORY_TCC_GENERATOR",
            seed_description=seed_description,
        )
        # Try structured output first
        try:
            return await self._llm.complete_structured(
                system_prompt="You are an expert playwright and narrative designer.",
                user_prompt=prompt,
                response_model=TCCN,
            )
        except Exception:
            # Fallback: get raw text and parse manually
            raw = await self._llm.complete(
                system_prompt="You are an expert playwright and narrative designer.",
                user_prompt=prompt,
            )
            return self._parse_tccn_from_text(raw)

    async def parse_characters(self, tccn: TCCN) -> list[CharacterSummary]:
        """Extract structured character list from a TCCN using the parser prompt."""
        prompt = self._prompts.render(
            "parsers",
            "CHARACTER_LIST_PARSER",
            tcc_context=tccn.to_prompt_text(),
        )
        raw = await self._llm.complete(
            system_prompt="You parse structured data from narrative text.",
            user_prompt=prompt,
        )
        # Parse the JSON list of character dicts
        import json
        import re

        # Extract JSON from response
        fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        text = fenced.group(1).strip() if fenced else raw.strip()

        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            text = text[start:end]

        data = json.loads(text)
        return [
            CharacterSummary(
                name=item.get("Character", item.get("name", "")),
                description=item.get("Description", item.get("description", "")),
            )
            for item in data
        ]

    def _parse_tccn_from_text(self, raw: str) -> TCCN:
        """Best-effort parse of free-text TCCN output."""
        sections: dict[str, str] = {}
        current_key = ""
        lines = raw.split("\n")

        for line in lines:
            upper = line.strip().upper()
            if upper.startswith("TELEOLOGY"):
                current_key = "teleology"
                # Grab text after the colon if present
                rest = line.split(":", 1)[1].strip() if ":" in line else ""
                sections[current_key] = rest
            elif upper.startswith("CONTEXT"):
                current_key = "context"
                rest = line.split(":", 1)[1].strip() if ":" in line else ""
                sections[current_key] = rest
            elif upper.startswith("CHARACTERS"):
                current_key = "characters"
                sections[current_key] = ""
            elif upper.startswith("NARRATIVE THREADS"):
                current_key = "narrative_threads"
                sections[current_key] = ""
            elif current_key:
                sections[current_key] = sections.get(current_key, "") + "\n" + line

        # Parse characters from numbered list
        chars_text = sections.get("characters", "")
        characters: list[CharacterSummary] = []
        for line in chars_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number/bullet
            import re

            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if ":" in line:
                name, desc = line.split(":", 1)
                characters.append(
                    CharacterSummary(name=name.strip(), description=desc.strip())
                )
            elif " - " in line:
                name, desc = line.split(" - ", 1)
                characters.append(
                    CharacterSummary(name=name.strip(), description=desc.strip())
                )

        # Parse narrative threads
        threads_text = sections.get("narrative_threads", "")
        threads: list[NarrativeThread] = []
        for line in threads_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            import re

            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            if line:
                threads.append(NarrativeThread(thread=line))

        return TCCN(
            teleology=sections.get("teleology", "").strip(),
            context=sections.get("context", "").strip(),
            characters=characters or [CharacterSummary(name="Unknown", description="")],
            narrative_threads=threads or [NarrativeThread(thread="")],
        )
