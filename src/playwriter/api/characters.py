from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends

from playwriter.api.dependencies import get_character_service
from playwriter.models.character import Character
from playwriter.models.story import TCCN, CharacterSummary
from playwriter.services.character import CharacterService

router = APIRouter(prefix="/api/characters", tags=["characters"])


# --- Request models ---

class GenerateRequest(BaseModel):
    tccn: TCCN
    character_name: str
    character_description: str


class RefineRequest(BaseModel):
    character: Character
    tccn: TCCN
    rounds: int = 1


class EnrichRequest(BaseModel):
    character: Character


class EmbodyRequest(BaseModel):
    character: Character
    tccn: TCCN
    scene_description: str
    use_strong: bool = True


class ChatRequest(BaseModel):
    session_id: str
    message: str


# --- Endpoints ---

@router.post("/generate")
async def generate_character(
    body: GenerateRequest,
    svc: CharacterService = Depends(get_character_service),
):
    """Generate a full character profile from TCCN context and description."""
    char = await svc.generate(
        body.tccn,
        CharacterSummary(name=body.character_name, description=body.character_description),
    )
    return char.model_dump()


@router.post("/refine")
async def refine_character(
    body: RefineRequest,
    svc: CharacterService = Depends(get_character_service),
):
    """Iteratively refine a character profile."""
    refined = await svc.refine(body.character, body.tccn, rounds=body.rounds)
    return refined.model_dump()


@router.post("/enrich")
async def enrich_character(
    body: EnrichRequest,
    svc: CharacterService = Depends(get_character_service),
):
    """Enrich a character with external inspiration sources."""
    enriched = await svc.enrich(body.character)
    return enriched.model_dump()


@router.post("/embody")
async def embody_character(
    body: EmbodyRequest,
    svc: CharacterService = Depends(get_character_service),
):
    """Start an embodiment session for interactive character chat."""
    session_id = svc.embody(
        body.character,
        body.tccn,
        body.scene_description,
        use_strong=body.use_strong,
    )
    return {"session_id": session_id}


@router.post("/chat")
async def chat_with_character(
    body: ChatRequest,
    svc: CharacterService = Depends(get_character_service),
):
    """Send a message to an embodied character."""
    response = await svc.chat(body.session_id, body.message)
    return {"response": response}
