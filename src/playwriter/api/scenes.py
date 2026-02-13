from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends

from playwriter.api.dependencies import get_scene_service, get_trope_service
from playwriter.models.character import Character
from playwriter.models.scene import Scene
from playwriter.models.story import TCCN
from playwriter.models.trope import TropeSample
from playwriter.services.scene import SceneService
from playwriter.services.trope import TropeService

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


class ComposeRequest(BaseModel):
    tccn: TCCN
    trope_count: int = Field(default=5, description="Number of random tropes to inject")
    trope_sample: Optional[TropeSample] = Field(
        default=None, description="Provide your own tropes instead of random sampling"
    )


class EvaluateRequest(BaseModel):
    tccn: TCCN
    scenes: List[Scene]


class WriteRequest(BaseModel):
    tccn: TCCN
    scenes: List[Scene]
    characters: Optional[List[Character]] = None


@router.post("/compose")
async def compose_scenes(
    body: ComposeRequest,
    scene_svc: SceneService = Depends(get_scene_service),
    trope_svc: TropeService = Depends(get_trope_service),
):
    """Compose scenes from TCCN, auto-injecting random tropes as literary fate."""
    trope_sample = body.trope_sample
    if trope_sample is None:
        trope_sample = trope_svc.sample_random(n=body.trope_count)

    scenes = await scene_svc.compose_scenes(body.tccn, trope_sample)
    return {
        "scenes": [s.model_dump() for s in scenes],
        "tropes_used": trope_sample.model_dump(),
    }


@router.post("/evaluate")
async def evaluate_scenes(
    body: EvaluateRequest,
    svc: SceneService = Depends(get_scene_service),
):
    """Evaluate scene coherence and quality."""
    critique = await svc.evaluate_scenes(body.tccn, body.scenes)
    return {"evaluation": critique}


@router.post("/write")
async def write_scenes(
    body: WriteRequest,
    svc: SceneService = Depends(get_scene_service),
):
    """Write theatrical prose from scene outlines."""
    prose = await svc.write_scene(body.tccn, body.scenes, body.characters)
    return {"prose": prose}
