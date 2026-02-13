from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from playwriter.api.dependencies import get_trope_service
from playwriter.services.trope import TropeService

router = APIRouter(prefix="/api/tropes", tags=["tropes"])


@router.get("/random")
def random_tropes(
    n: int = Query(default=5, ge=1, le=50),
    svc: TropeService = Depends(get_trope_service),
):
    """Sample random tropes from the master list (~216K tropes)."""
    sample = svc.sample_random(n=n)
    return sample.model_dump()


@router.get("/search")
def search_tropes(
    q: str = Query(..., min_length=1),
    n: int = Query(default=10, ge=1, le=100),
    svc: TropeService = Depends(get_trope_service),
):
    """Search tropes by name or description."""
    sample = svc.search(q, n=n)
    return sample.model_dump()


@router.get("/by-media")
def tropes_by_media(
    media: str = Query(..., description="tv, film, or lit"),
    title: Optional[str] = Query(default=None),
    n: int = Query(default=5, ge=1, le=50),
    svc: TropeService = Depends(get_trope_service),
):
    """Sample tropes from a specific media type, optionally for a title."""
    sample = svc.sample_by_media(media, title=title, n=n)
    return sample.model_dump()
