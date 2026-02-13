from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends, HTTPException

from playwriter.api.dependencies import get_game_service
from playwriter.models.character import Character
from playwriter.models.scene import Scene
from playwriter.models.story import TCCN
from playwriter.services.game import GameService

router = APIRouter(prefix="/api/games", tags=["games"])


class CreateGameRequest(BaseModel):
    tccn: TCCN
    characters: Dict[str, Character]
    scene: Optional[Scene] = None


class PlayerActionRequest(BaseModel):
    message: str


class NPCRespondRequest(BaseModel):
    character_name: str


@router.post("")
async def create_game(
    body: CreateGameRequest,
    svc: GameService = Depends(get_game_service),
):
    """Create a new game session."""
    session = svc.create_session(body.tccn, body.characters, body.scene)
    return session.model_dump()


@router.get("/{session_id}")
async def get_game(
    session_id: str,
    svc: GameService = Depends(get_game_service),
):
    """Get the current state of a game session."""
    session = svc.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Game session not found")
    return session.model_dump()


@router.post("/{session_id}/action")
async def player_action(
    session_id: str,
    body: PlayerActionRequest,
    svc: GameService = Depends(get_game_service),
):
    """Submit a player action."""
    msg = await svc.player_action(session_id, body.message)
    return msg.model_dump()


@router.post("/{session_id}/advance")
async def advance_game(
    session_id: str,
    svc: GameService = Depends(get_game_service),
):
    """Have the Game Master advance the story."""
    msg = await svc.gm_action(session_id)
    return msg.model_dump()


@router.post("/{session_id}/npc")
async def npc_respond(
    session_id: str,
    body: NPCRespondRequest,
    svc: GameService = Depends(get_game_service),
):
    """Have a specific NPC character respond."""
    msg = await svc.npc_respond(session_id, body.character_name)
    return msg.model_dump()
