from __future__ import annotations

from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends

from playwriter.api.dependencies import get_pipeline_service
from playwriter.services.pipeline import PipelineService

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class PipelineRequest(BaseModel):
    seed_description: str
    trope_count: int = Field(default=5, ge=1, le=20)
    refine_rounds: int = Field(default=1, ge=0, le=5)


@router.post("/full")
async def run_full_pipeline(
    body: PipelineRequest,
    svc: PipelineService = Depends(get_pipeline_service),
):
    """Run the complete Playwriter pipeline: seed → TCCN → characters → scenes → prose.

    This is the main entry point for generating a complete game data package
    from a single seed description.
    """
    result = await svc.full_pipeline(
        seed_description=body.seed_description,
        trope_count=body.trope_count,
        refine_rounds=body.refine_rounds,
    )
    return result
