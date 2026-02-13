"""Narrative Engine API — temporal narrative generation with dice and feedback loops.

Endpoints for world creation, beat/scene/act progression, director controls,
and read access to accumulated prose, dice history, and thread states.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from playwriter.api.dependencies import get_narrative_engine
from playwriter.models.character import Character
from playwriter.models.narrative import (
    Act,
    Beat,
    EngineScene,
    NarrativeThreadState,
    NarrativeWorld,
    WorldEvent,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/narrative", tags=["narrative"])

# ── Request / response models ───────────────────────────────────────────


class CreateWorldRequest(BaseModel):
    seed_description: str
    mode: str = "autonomous"
    trope_pool_size: int = 30
    num_characters: Optional[int] = None


class CreateWorldResponse(BaseModel):
    world_id: str
    status: str
    characters: List[str]
    thread_count: int
    trope_pool_size: int


class AdvanceRequest(BaseModel):
    steps: int = 1


class AdvanceResponse(BaseModel):
    events: List[dict]


class DirectorDiceRequest(BaseModel):
    actor: str
    action: str
    forced_roll: int = Field(ge=1, le=100)


class DirectorEventRequest(BaseModel):
    event_description: str


class DirectorRedirectRequest(BaseModel):
    character_name: str
    new_direction: str


class DirectorTropeRequest(BaseModel):
    trope_query: str


class DirectorThreadRequest(BaseModel):
    thread_index: int
    new_status: str = "advancing"


class ModeRequest(BaseModel):
    mode: str  # "autonomous" or "director"


class WorldSummary(BaseModel):
    id: str
    seed_description: str
    status: str
    mode: str
    acts: int
    characters: List[str]
    threads: int


# ── World management ────────────────────────────────────────────────────


@router.post("/worlds", response_model=CreateWorldResponse)
async def create_world(body: CreateWorldRequest):
    """Initialize a new narrative world from a seed description."""
    engine = get_narrative_engine()
    log.info("Creating world: seed=%s, mode=%s", body.seed_description[:60], body.mode)
    try:
        world = await engine.initialize_world(
            seed_description=body.seed_description,
            mode=body.mode,
            trope_pool_size=body.trope_pool_size,
            num_characters=body.num_characters,
        )
    except Exception as exc:
        log.error("World creation failed: %s", exc)
        raise HTTPException(500, f"World creation failed: {exc}")

    return CreateWorldResponse(
        world_id=world.id,
        status=world.status,
        characters=list(world.characters.keys()),
        thread_count=len(world.thread_states),
        trope_pool_size=len(world.global_trope_pool),
    )


@router.post("/worlds/stream")
async def create_world_stream(body: CreateWorldRequest):
    """Initialize a world with SSE progress streaming.

    Yields progress events as each generation step completes, then a
    final ``complete`` event with the full world summary.
    """
    engine = get_narrative_engine()
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(step: str, detail: str):
        await progress_queue.put({"step": step, "detail": detail})

    async def generator():
        # Start world creation as a background task
        task = asyncio.create_task(engine.initialize_world(
            seed_description=body.seed_description,
            mode=body.mode,
            trope_pool_size=body.trope_pool_size,
            num_characters=body.num_characters,
            on_progress=on_progress,
        ))

        # Stream progress events until the task completes
        while not task.done():
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield f"data: {json.dumps(msg)}\n\n"
            except asyncio.TimeoutError:
                continue

        # Drain any remaining messages
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield f"data: {json.dumps(msg)}\n\n"

        # Final result
        try:
            world = task.result()
            yield f"data: {json.dumps({'step': 'done', 'world_id': world.id, 'characters': list(world.characters.keys()), 'thread_count': len(world.thread_states), 'trope_pool_size': len(world.global_trope_pool)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'step': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/worlds", response_model=List[WorldSummary])
async def list_worlds():
    """List all narrative worlds."""
    engine = get_narrative_engine()
    return engine.list_worlds()


@router.get("/worlds/{world_id}")
async def get_world(world_id: str):
    """Get full world state."""
    engine = get_narrative_engine()
    try:
        world = engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")
    return world.model_dump()


@router.get("/worlds/{world_id}/summary")
async def get_world_summary(world_id: str):
    """Condensed world summary for the timeline."""
    engine = get_narrative_engine()
    try:
        world = engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")

    acts_summary = []
    for act in world.acts:
        scenes_summary = []
        for scene in act.scenes:
            beats_summary = [
                {
                    "sequence": b.sequence,
                    "actor": b.actor,
                    "outcome": b.dice_roll.outcome.value if b.dice_roll else None,
                    "prose_preview": b.prose[:120] if b.prose else "",
                }
                for b in scene.beats
            ]
            scenes_summary.append({
                "id": scene.id,
                "number": scene.number,
                "actors": scene.actors,
                "setting": scene.setting,
                "status": scene.status,
                "beats": beats_summary,
            })
        acts_summary.append({
            "id": act.id,
            "number": act.number,
            "title": act.title,
            "status": act.status,
            "scenes": scenes_summary,
            "world_events": [we.description for we in act.world_events],
        })

    return {
        "id": world.id,
        "status": world.status,
        "mode": world.mode.value,
        "teleology": world.tccn.teleology,
        "context": world.tccn.context,
        "characters": list(world.characters.keys()),
        "threads": [
            {"thread": ts.thread, "status": ts.status, "tension": ts.tension_level}
            for ts in world.thread_states
        ],
        "acts": acts_summary,
    }


@router.delete("/worlds/{world_id}")
async def delete_world(world_id: str):
    """Delete a narrative world."""
    engine = get_narrative_engine()
    try:
        engine.delete_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")
    return {"status": "deleted"}


# ── Progression ─────────────────────────────────────────────────────────


@router.post("/worlds/{world_id}/advance", response_model=AdvanceResponse)
async def advance(world_id: str, body: AdvanceRequest):
    """Advance the narrative by N beats (auto-manages scene/act boundaries)."""
    engine = get_narrative_engine()
    try:
        events = await engine.advance(world_id, steps=body.steps)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        log.error("Advance failed: %s", exc)
        raise HTTPException(500, f"Advance failed: {exc}")
    return AdvanceResponse(events=events)


@router.post("/worlds/{world_id}/advance/scene")
async def advance_scene(world_id: str):
    """Advance until the current scene completes."""
    engine = get_narrative_engine()
    all_events = []
    try:
        # Keep advancing beats until a scene_completed event appears
        for _ in range(20):  # safety limit
            events = await engine.advance(world_id, steps=1)
            all_events.extend(events)
            if any(e.get("type") == "scene_completed" for e in events):
                break
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        log.error("Scene advance failed: %s", exc)
        raise HTTPException(500, f"Scene advance failed: {exc}")
    return {"events": all_events}


@router.post("/worlds/{world_id}/advance/act")
async def advance_act(world_id: str):
    """Advance until the current act completes."""
    engine = get_narrative_engine()
    all_events = []
    try:
        for _ in range(100):  # safety limit
            events = await engine.advance(world_id, steps=1)
            all_events.extend(events)
            if any(e.get("type") == "act_completed" for e in events):
                break
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        log.error("Act advance failed: %s", exc)
        raise HTTPException(500, f"Act advance failed: {exc}")
    return {"events": all_events}


# ── Director controls ───────────────────────────────────────────────────


@router.put("/worlds/{world_id}/mode")
async def set_mode(world_id: str, body: ModeRequest):
    """Switch between autonomous and director mode."""
    engine = get_narrative_engine()
    try:
        world = engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")

    from playwriter.models.narrative import EngineMode
    world.mode = EngineMode.DIRECTOR if body.mode == "director" else EngineMode.AUTONOMOUS
    return {"mode": world.mode.value}


@router.post("/worlds/{world_id}/director/override-dice")
async def director_override_dice(world_id: str, body: DirectorDiceRequest):
    """Director forces a specific dice roll."""
    engine = get_narrative_engine()
    try:
        beat = await engine.director_override_dice(
            world_id, body.actor, body.action, body.forced_roll,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        log.error("Director dice override failed: %s", exc)
        raise HTTPException(500, str(exc))
    return beat.model_dump()


@router.post("/worlds/{world_id}/director/inject-event")
async def director_inject_event(world_id: str, body: DirectorEventRequest):
    """Director injects a world event."""
    engine = get_narrative_engine()
    try:
        event = await engine.director_inject_event(world_id, body.event_description)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return event.model_dump()


@router.post("/worlds/{world_id}/director/redirect-character")
async def director_redirect_character(world_id: str, body: DirectorRedirectRequest):
    """Director alters a character's direction."""
    engine = get_narrative_engine()
    try:
        char = await engine.director_redirect_character(
            world_id, body.character_name, body.new_direction,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return char.model_dump()


@router.post("/worlds/{world_id}/director/force-trope")
async def director_force_trope(world_id: str, body: DirectorTropeRequest):
    """Director searches for and injects specific tropes."""
    engine = get_narrative_engine()
    try:
        tropes = await engine.director_force_trope(world_id, body.trope_query)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return [t.model_dump() for t in tropes]


@router.post("/worlds/{world_id}/director/choose-thread")
async def director_choose_thread(world_id: str, body: DirectorThreadRequest):
    """Director manually sets a thread's status."""
    engine = get_narrative_engine()
    try:
        ts = await engine.director_choose_thread(
            world_id, body.thread_index, body.new_status,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return ts.model_dump()


# ── Read endpoints ──────────────────────────────────────────────────────


@router.get("/worlds/{world_id}/acts")
async def get_acts(world_id: str):
    """List all acts in a world."""
    engine = get_narrative_engine()
    try:
        world = engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")
    return [act.model_dump() for act in world.acts]


@router.get("/worlds/{world_id}/characters")
async def get_characters(world_id: str):
    """Get all character profiles."""
    engine = get_narrative_engine()
    try:
        world = engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")
    return {name: char.model_dump() for name, char in world.characters.items()}


@router.get("/worlds/{world_id}/threads")
async def get_threads(world_id: str):
    """Get current thread states."""
    engine = get_narrative_engine()
    try:
        world = engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")
    return [ts.model_dump() for ts in world.thread_states]


@router.get("/worlds/{world_id}/prose")
async def get_prose(world_id: str):
    """Get all accumulated theatrical prose."""
    engine = get_narrative_engine()
    try:
        prose = engine.get_accumulated_prose(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")
    return {"prose": prose}


@router.get("/worlds/{world_id}/dice-history")
async def get_dice_history(world_id: str):
    """Get all dice rolls with fate modifiers."""
    engine = get_narrative_engine()
    try:
        history = engine.get_dice_history(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")
    return {"rolls": history}


@router.get("/worlds/{world_id}/scenes/{scene_id}/beats")
async def get_scene_beats(world_id: str, scene_id: str):
    """Get all beats for a specific scene."""
    engine = get_narrative_engine()
    try:
        world = engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")

    for act in world.acts:
        for scene in act.scenes:
            if scene.id == scene_id:
                return [beat.model_dump() for beat in scene.beats]
    raise HTTPException(404, "Scene not found")


# ── SSE Streaming ───────────────────────────────────────────────────────


@router.get("/worlds/{world_id}/stream")
async def stream_narrative(world_id: str, request: Request, steps: int = 10):
    """SSE endpoint that streams narrative events in real-time as beats resolve."""
    engine = get_narrative_engine()
    try:
        engine.get_world(world_id)
    except ValueError:
        raise HTTPException(404, "World not found")

    async def event_generator():
        for step in range(steps):
            if await request.is_disconnected():
                break
            try:
                events = await engine.advance(world_id, steps=1)
                for event in events:
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                break
            await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
