from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from playwriter.models.character import Character
from playwriter.models.scene import Scene
from playwriter.models.story import TCCN


class GameMessage(BaseModel):
    """A single message in a game session."""

    role: Literal["player", "npc", "game_master"]
    speaker: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GameSession(BaseModel):
    """The full state of an active game session."""

    id: str
    tccn: TCCN
    characters: Dict[str, Character] = Field(default_factory=dict)
    scene: Optional[Scene] = None
    history: List[GameMessage] = Field(default_factory=list)
