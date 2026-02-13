from __future__ import annotations

import uuid

from playwriter.llm.base import LLMProvider
from playwriter.memory.conversation import ConversationMemory
from playwriter.models.character import Character
from playwriter.models.story import TCCN, CharacterSummary
from playwriter.parsing.output_parser import OutputParser
from playwriter.prompts.loader import PromptLoader


class CharacterService:
    """Character lifecycle: generate → refine → enrich → embody → chat.

    Mirrors the three-stage pattern from the original notebooks but without
    LangChain dependencies.
    """

    def __init__(
        self,
        strong_llm: LLMProvider,
        fast_llm: LLMProvider | None = None,
        prompts: PromptLoader | None = None,
    ):
        self._strong = strong_llm
        self._fast = fast_llm or strong_llm
        self._prompts = prompts or PromptLoader()
        # Active embodiment sessions: session_id → (system_prompt, memory)
        self._sessions: dict[str, _EmbodimentSession] = {}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        tccn: TCCN,
        character_summary: CharacterSummary,
    ) -> Character:
        """First-pass character generation using FIRST_PASS_CHARACTER_DESIGNER."""
        format_instructions = OutputParser.format_instructions(Character)
        prompt = self._prompts.render(
            "generators",
            "FIRST_PASS_CHARACTER_DESIGNER",
            tcc_context=tccn.to_prompt_text(),
            character_description=f"{character_summary.name}: {character_summary.description}",
            format_instructions=format_instructions,
        )
        character = await self._strong.complete_structured(
            system_prompt="You are an expert character designer for theatrical plays.",
            user_prompt=prompt,
            response_model=Character,
        )
        character.name = character.name or character_summary.name
        return character

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------

    async def refine(
        self,
        character: Character,
        tccn: TCCN,
        rounds: int = 1,
    ) -> Character:
        """Iteratively refine a character using FULL_DESCRIPTION_CHARACTER_REFINER."""
        current = character
        for _ in range(rounds):
            format_instructions = OutputParser.format_instructions(Character)
            prompt = self._prompts.render(
                "refiners",
                "FULL_DESCRIPTION_CHARACTER_REFINER",
                tcc_context=tccn.to_prompt_text(),
                character_profile=current.to_prompt_text(),
                format_instructions=format_instructions,
            )
            current = await self._strong.complete_structured(
                system_prompt="You are a master character developer. Reimagine and deepen this character.",
                user_prompt=prompt,
                response_model=Character,
            )
            current.name = current.name or character.name
        return current

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    async def enrich(self, character: Character) -> Character:
        """Enrich a character with inspiration from historical/fictional sources."""
        prompt = self._prompts.render(
            "generators",
            "FIRST_PASS_CHARACTER_ENRICHMENT",
            hppti_context=character.to_prompt_text(),
        )
        raw = await self._strong.complete(
            system_prompt=(
                "You enrich character designs by drawing from historical and "
                "fictional inspiration sources."
            ),
            user_prompt=prompt,
        )
        # Enrichment returns free-text; try to re-parse into Character
        try:
            enriched = OutputParser.parse(raw, Character)
            enriched.name = enriched.name or character.name
            return enriched
        except ValueError:
            # If not parseable, update the internal_state with the enrichment text
            character.internal_state += f"\n\n[Enrichment]\n{raw[:2000]}"
            return character

    # ------------------------------------------------------------------
    # Embodiment & Chat
    # ------------------------------------------------------------------

    def embody(
        self,
        character: Character,
        tccn: TCCN,
        scene_description: str,
        *,
        use_strong: bool = True,
    ) -> str:
        """Start an embodiment session. Returns a session_id for chat."""
        session_id = uuid.uuid4().hex[:12]
        system_prompt = self._prompts.render(
            "embodiers",
            "CHARACTER_EMBODIER",
            tcc_context=tccn.to_prompt_text(),
            character_profile=character.to_prompt_text(),
            scene_description=scene_description,
        )
        self._sessions[session_id] = _EmbodimentSession(
            system_prompt=system_prompt,
            memory=ConversationMemory(window_size=50),
            character_name=character.name,
            use_strong=use_strong,
        )
        return session_id

    async def chat(self, session_id: str, message: str) -> str:
        """Send a message to an embodied character and get their response."""
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"No embodiment session with id '{session_id}'")

        session.memory.add_message("user", message)
        llm = self._strong if session.use_strong else self._fast

        # Build the user prompt with conversation history
        history = session.memory.to_prompt_text()
        user_prompt = (
            f"Conversation so far:\n{history}\n\n"
            f"Respond as the character would naturally speak in this specific situation. "
            f"Follow their Voice & Speech Style. Use the play format: (action description) Dialogue. "
            f"Keep it real — not every line needs to be profound."
        )

        response = await llm.complete(
            system_prompt=session.system_prompt,
            user_prompt=user_prompt,
        )
        session.memory.add_message("assistant", response)
        return response

    def forget(self, session_id: str) -> None:
        """Clear the memory of an embodiment session."""
        session = self._sessions.get(session_id)
        if session:
            session.memory.clear()

    def end_session(self, session_id: str) -> None:
        """Remove an embodiment session entirely."""
        self._sessions.pop(session_id, None)


class _EmbodimentSession:
    """Internal state for an active character embodiment."""

    __slots__ = ("system_prompt", "memory", "character_name", "use_strong")

    def __init__(
        self,
        system_prompt: str,
        memory: ConversationMemory,
        character_name: str,
        use_strong: bool,
    ):
        self.system_prompt = system_prompt
        self.memory = memory
        self.character_name = character_name
        self.use_strong = use_strong
