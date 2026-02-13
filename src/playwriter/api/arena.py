"""Character Arena — the simplified game-play flow.

Recovers the original tools/character_arena functionality:
  1. Load/set a TCC context
  2. Generate a character from a description
  3. Set a scene
  4. Chat back-and-forth with the embodied character

All state is held server-side per session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from playwriter.llm.registry import get_provider
from playwriter.memory.conversation import ConversationMemory
from playwriter.models.character import Character
from playwriter.models.story import TCCN, CharacterSummary, NarrativeThread
from playwriter.parsing.output_parser import OutputParser
from playwriter.prompts.loader import PromptLoader
from playwriter.api.dependencies import get_active_provider, get_active_model

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/arena", tags=["arena"])

# ── In-memory session store ──────────────────────────────────────────────

_sessions: Dict[str, "_ArenaSession"] = {}
_prompts = PromptLoader()


class _ArenaSession:
    __slots__ = (
        "tcc_context", "character", "scene_description",
        "system_prompt", "memory", "character_name",
    )

    def __init__(self):
        self.tcc_context: str = ""
        self.character: Optional[Character] = None
        self.scene_description: str = ""
        self.system_prompt: str = ""
        self.memory: ConversationMemory = ConversationMemory(window_size=50)
        self.character_name: str = ""


# ── Request / response models ───────────────────────────────────────────

class StartRequest(BaseModel):
    tcc_context: str
    character_description: str
    scene_description: str


class StartResponse(BaseModel):
    session_id: str
    character: Character
    character_name: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    history: List[Dict[str, str]]


class SessionInfo(BaseModel):
    session_id: str
    character_name: str
    character: Optional[Character] = None
    scene_description: str
    history_length: int


class WorldDetails(BaseModel):
    """Full snapshot of everything generated for a session."""
    session_id: str
    tcc_context: str
    character_name: str
    character: Optional[Character] = None
    scene_description: str
    system_prompt: str
    history: List[Dict[str, str]]


# ── Endpoints ────────────────────────────────────────────────────────────

@router.post("/start", response_model=StartResponse)
async def start_arena(body: StartRequest):
    """Generate a character and start an arena session in one call."""
    provider_name = get_active_provider()
    active_model = get_active_model()
    log.info("Arena start: provider=%s, model=%s", provider_name, active_model or "(default)")
    strong = get_provider(provider_name, tier="strong", model=active_model)

    # Generate character
    log.info("Generating character from description (%d chars)", len(body.character_description))
    format_instructions = OutputParser.format_instructions(Character)
    prompt = _prompts.render(
        "generators",
        "FIRST_PASS_CHARACTER_DESIGNER",
        tcc_context=body.tcc_context,
        character_description=body.character_description,
        format_instructions=format_instructions,
    )
    try:
        character = await strong.complete_structured(
            system_prompt="You are an expert character designer for theatrical plays.",
            user_prompt=prompt,
            response_model=Character,
        )
        log.info("Character generated: name=%s", character.name)
    except Exception as exc:
        log.error("Character generation failed: %s", exc)
        raise HTTPException(500, f"Character generation failed: {exc}")

    # Refine once
    log.info("Refining character: %s", character.name)
    refine_prompt = _prompts.render(
        "refiners",
        "FULL_DESCRIPTION_CHARACTER_REFINER",
        tcc_context=body.tcc_context,
        character_profile=character.to_prompt_text(),
        format_instructions=format_instructions,
    )
    try:
        character = await strong.complete_structured(
            system_prompt="You are a master character developer. Reimagine and deepen this character.",
            user_prompt=refine_prompt,
            response_model=Character,
        )
        log.info("Character refined: name=%s, fields_populated=%d",
                 character.name,
                 sum(1 for f in character.model_fields if getattr(character, f)))
    except Exception as exc:
        log.warning("Character refinement failed, using initial version: %s", exc)
        # Keep the original generated character

    # Build embodiment system prompt
    system_prompt = _prompts.render(
        "embodiers",
        "CHARACTER_EMBODIER",
        tcc_context=body.tcc_context,
        character_profile=character.to_prompt_text(),
        scene_description=body.scene_description,
    )

    # Create session
    session_id = uuid.uuid4().hex[:12]
    session = _ArenaSession()
    session.tcc_context = body.tcc_context
    session.character = character
    session.scene_description = body.scene_description
    session.system_prompt = system_prompt
    session.character_name = character.name or "Character"
    _sessions[session_id] = session

    log.info("Arena session created: id=%s, character=%s", session_id, session.character_name)
    return StartResponse(
        session_id=session_id,
        character=character,
        character_name=session.character_name,
    )


@router.post("/start/stream")
async def start_arena_stream(body: StartRequest):
    """Generate a character and start an arena session, streaming progress via SSE."""
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def _progress(step: str, detail: str, **extra):
        event = {"step": step, "detail": detail, **extra}
        await progress_queue.put(event)

    async def _run():
        provider_name = get_active_provider()
        active_model = get_active_model()
        strong = get_provider(provider_name, tier="strong", model=active_model)

        await _progress("generating", "Designing character from description...")

        format_instructions = OutputParser.format_instructions(Character)
        prompt = _prompts.render(
            "generators",
            "FIRST_PASS_CHARACTER_DESIGNER",
            tcc_context=body.tcc_context,
            character_description=body.character_description,
            format_instructions=format_instructions,
        )
        character = await strong.complete_structured(
            system_prompt="You are an expert character designer for theatrical plays.",
            user_prompt=prompt,
            response_model=Character,
        )

        snippets = []
        if character.name:
            snippets.append(f"Name: {character.name}")
        if character.philosophy:
            snippets.append(f"Philosophy: {character.philosophy[:120]}...")
        if character.voice_style:
            snippets.append(f"Voice: {character.voice_style[:120]}...")
        if character.ambitions:
            snippets.append(f"Ambitions: {character.ambitions[:120]}...")
        await _progress("generated", "\n".join(snippets) or f"Character designed: {character.name}")

        await _progress("refining", f"Refining and deepening {character.name}...")

        refine_prompt = _prompts.render(
            "refiners",
            "FULL_DESCRIPTION_CHARACTER_REFINER",
            tcc_context=body.tcc_context,
            character_profile=character.to_prompt_text(),
            format_instructions=format_instructions,
        )
        try:
            character = await strong.complete_structured(
                system_prompt="You are a master character developer. Reimagine and deepen this character.",
                user_prompt=refine_prompt,
                response_model=Character,
            )
        except Exception as exc:
            await _progress("refine_warning", f"Refinement failed, using initial version: {exc}")

        refined_snippets = []
        if character.philosophy:
            refined_snippets.append(f"Philosophy: {character.philosophy[:150]}...")
        if character.voice_style:
            refined_snippets.append(f"Voice: {character.voice_style[:150]}...")
        if character.internal_contradictions:
            refined_snippets.append(f"Contradictions: {'; '.join(character.internal_contradictions)[:150]}...")
        await _progress("refined", "\n".join(refined_snippets) or f"Character refined: {character.name}")

        await _progress("embodying", "Building embodiment prompt for scene...")

        system_prompt = _prompts.render(
            "embodiers",
            "CHARACTER_EMBODIER",
            tcc_context=body.tcc_context,
            character_profile=character.to_prompt_text(),
            scene_description=body.scene_description,
        )

        session_id = uuid.uuid4().hex[:12]
        session = _ArenaSession()
        session.tcc_context = body.tcc_context
        session.character = character
        session.scene_description = body.scene_description
        session.system_prompt = system_prompt
        session.character_name = character.name or "Character"
        _sessions[session_id] = session

        return session_id, character, session.character_name

    async def generator():
        task = asyncio.create_task(_run())

        while not task.done():
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                continue

        # Drain remaining
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield f"data: {json.dumps(msg)}\n\n"

        try:
            session_id, character, character_name = task.result()
            yield f"data: {json.dumps({'step': 'done', 'session_id': session_id, 'character': character.model_dump(), 'character_name': character_name})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'step': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/chat", response_model=ChatResponse)
async def arena_chat(body: ChatRequest):
    """Chat with the embodied character in an arena session."""
    session = _sessions.get(body.session_id)
    if session is None:
        log.warning("Chat: session not found: %s", body.session_id)
        raise HTTPException(404, "Arena session not found")

    log.info("Arena chat: session=%s, user_msg_len=%d", body.session_id, len(body.message))
    session.memory.add_message("user", body.message)

    provider_name = get_active_provider()
    active_model = get_active_model()
    llm = get_provider(provider_name, tier="strong", model=active_model)

    history = session.memory.to_prompt_text()
    user_prompt = (
        f"Conversation so far:\n{history}\n\n"
        f"Respond as the character would naturally speak in this specific situation. "
        f"Follow their Voice & Speech Style. Use the play format: (action description) Dialogue. "
        f"Keep it real — not every line needs to be profound."
    )
    try:
        response = await llm.complete(
            system_prompt=session.system_prompt,
            user_prompt=user_prompt,
        )
        log.info("Arena chat response: session=%s, response_len=%d",
                 body.session_id, len(response))
    except Exception as exc:
        log.error("Arena chat LLM error: session=%s, error=%s", body.session_id, exc)
        raise HTTPException(500, f"LLM error during chat: {exc}")

    session.memory.add_message("assistant", response)

    return ChatResponse(
        response=response,
        history=session.memory.get_all(),
    )


@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all active arena sessions."""
    log.info("Listing %d arena sessions", len(_sessions))
    return [
        SessionInfo(
            session_id=sid,
            character_name=s.character_name,
            character=s.character,
            scene_description=s.scene_description,
            history_length=len(s.memory),
        )
        for sid, s in _sessions.items()
    ]


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get info about a specific arena session."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Arena session not found")
    return SessionInfo(
        session_id=session_id,
        character_name=session.character_name,
        character=session.character,
        scene_description=session.scene_description,
        history_length=len(session.memory),
    )


@router.get("/sessions/{session_id}/world", response_model=WorldDetails)
async def get_world_details(session_id: str):
    """Return the full generated world details for a session (for the inspector)."""
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Arena session not found")
    log.info("World details requested: session=%s", session_id)
    return WorldDetails(
        session_id=session_id,
        tcc_context=session.tcc_context,
        character_name=session.character_name,
        character=session.character,
        scene_description=session.scene_description,
        system_prompt=session.system_prompt,
        history=session.memory.get_all(),
    )


@router.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """End an arena session."""
    if session_id not in _sessions:
        raise HTTPException(404, "Arena session not found")
    del _sessions[session_id]
    log.info("Arena session ended: %s", session_id)
    return {"status": "ended"}
