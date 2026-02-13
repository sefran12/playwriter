from __future__ import annotations

import uuid
from datetime import datetime, timezone

from playwriter.llm.base import LLMProvider
from playwriter.memory.conversation import ConversationMemory
from playwriter.models.character import Character
from playwriter.models.game import GameMessage, GameSession
from playwriter.models.scene import Scene
from playwriter.models.story import TCCN
from playwriter.prompts.loader import PromptLoader


class GameService:
    """Game master orchestration for interactive play sessions.

    Manages:
    - Game session state (TCCN + characters + scene + history)
    - Turn-taking between player, NPCs, and game master
    - Shared conversation memory across all participants
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
        self._sessions: dict[str, _GameState] = {}

    def create_session(
        self,
        tccn: TCCN,
        characters: dict[str, Character],
        scene: Scene | None = None,
    ) -> GameSession:
        """Create a new game session and return its state."""
        session_id = uuid.uuid4().hex[:12]
        state = _GameState(
            session=GameSession(
                id=session_id,
                tccn=tccn,
                characters=characters,
                scene=scene,
            ),
            shared_memory=ConversationMemory(window_size=50),
        )
        self._sessions[session_id] = state
        return state.session

    def get_session(self, session_id: str) -> GameSession | None:
        state = self._sessions.get(session_id)
        return state.session if state else None

    async def player_action(self, session_id: str, message: str) -> GameMessage:
        """Process a player's input and record it."""
        state = self._get_state(session_id)
        msg = GameMessage(
            role="player",
            speaker="Player",
            content=message,
            timestamp=datetime.now(timezone.utc),
        )
        state.session.history.append(msg)
        state.shared_memory.add_message("user", f"[Player]: {message}")
        return msg

    async def gm_action(self, session_id: str) -> GameMessage:
        """Have the game master narrate / advance the story."""
        state = self._get_state(session_id)
        scene_desc = ""
        if state.session.scene:
            s = state.session.scene
            scene_desc = f"Scene {s.number}: {s.setting}. Actors: {', '.join(s.actors)}"

        system_prompt = self._prompts.render(
            "embodiers",
            "GAME_MASTER_EMBODIER",
            tcc_context=state.session.tccn.to_prompt_text(),
            gm_profile="You are the Game Master — narrator and world arbiter.",
            scene_description=scene_desc,
        )
        history = state.shared_memory.to_prompt_text()
        user_prompt = (
            f"Conversation so far:\n{history}\n\n"
            f"As the Game Master, advance the story. React to what just happened, "
            f"describe the environment, manage NPC reactions, and prompt the player."
        )
        response = await self._strong.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        msg = GameMessage(
            role="game_master",
            speaker="Game Master",
            content=response,
            timestamp=datetime.now(timezone.utc),
        )
        state.session.history.append(msg)
        state.shared_memory.add_message("assistant", f"[Game Master]: {response}")
        return msg

    async def npc_respond(
        self,
        session_id: str,
        character_name: str,
    ) -> GameMessage:
        """Have a specific NPC character respond in the game."""
        state = self._get_state(session_id)
        character = state.session.characters.get(character_name)
        if character is None:
            raise ValueError(
                f"Character '{character_name}' not in session. "
                f"Available: {list(state.session.characters)}"
            )

        scene_desc = ""
        if state.session.scene:
            s = state.session.scene
            scene_desc = f"Scene {s.number}: {s.setting}"

        system_prompt = self._prompts.render(
            "embodiers",
            "CHARACTER_EMBODIER",
            tcc_context=state.session.tccn.to_prompt_text(),
            character_profile=character.to_prompt_text(),
            scene_description=scene_desc,
        )
        history = state.shared_memory.to_prompt_text()
        user_prompt = (
            f"Conversation so far:\n{history}\n\n"
            f"Respond as {character_name}."
        )
        response = await self._fast.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        msg = GameMessage(
            role="npc",
            speaker=character_name,
            content=response,
            timestamp=datetime.now(timezone.utc),
        )
        state.session.history.append(msg)
        state.shared_memory.add_message(
            "assistant", f"[{character_name}]: {response}"
        )
        return msg

    def _get_state(self, session_id: str) -> _GameState:
        state = self._sessions.get(session_id)
        if state is None:
            raise ValueError(f"No game session with id '{session_id}'")
        return state


class _GameState:
    """Internal mutable state for an active game session."""

    __slots__ = ("session", "shared_memory")

    def __init__(self, session: GameSession, shared_memory: ConversationMemory):
        self.session = session
        self.shared_memory = shared_memory
