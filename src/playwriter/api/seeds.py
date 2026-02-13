from __future__ import annotations

from pydantic import BaseModel

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from playwriter.api.dependencies import get_seeding_service
from playwriter.db.database import get_session
from playwriter.db.tables import DBSeed, DBSeedCharacter, DBSeedThread
from playwriter.services.seeding import SeedingService

router = APIRouter(prefix="/api/seeds", tags=["seeds"])


class SeedRequest(BaseModel):
    description: str


@router.post("")
async def create_seed(
    body: SeedRequest,
    svc: SeedingService = Depends(get_seeding_service),
    db: AsyncSession = Depends(get_session),
):
    """Generate a TCCN from a seed description and persist it."""
    tccn = await svc.generate_seed(body.description)

    # Persist
    row = DBSeed(
        description=body.description,
        teleology=tccn.teleology,
        context=tccn.context,
    )
    db.add(row)
    await db.flush()

    for c in tccn.characters:
        db.add(DBSeedCharacter(seed_id=row.id, name=c.name, description=c.description))
    for t in tccn.narrative_threads:
        db.add(DBSeedThread(seed_id=row.id, thread=t.thread))

    await db.commit()

    return {"id": row.id, "tccn": tccn.model_dump()}


@router.get("")
async def list_seeds(db: AsyncSession = Depends(get_session)):
    """List all persisted seeds."""
    result = await db.execute(select(DBSeed).order_by(DBSeed.created_at.desc()))
    seeds = result.scalars().all()
    return [
        {
            "id": s.id,
            "description": s.description,
            "teleology": s.teleology[:200],
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in seeds
    ]


@router.get("/{seed_id}")
async def get_seed(seed_id: int, db: AsyncSession = Depends(get_session)):
    """Get a seed by ID with full TCCN data."""
    seed = await db.get(DBSeed, seed_id)
    if not seed:
        from fastapi import HTTPException
        raise HTTPException(404, "Seed not found")

    # Eagerly load relationships
    chars_result = await db.execute(
        select(DBSeedCharacter).where(DBSeedCharacter.seed_id == seed_id)
    )
    threads_result = await db.execute(
        select(DBSeedThread).where(DBSeedThread.seed_id == seed_id)
    )

    return {
        "id": seed.id,
        "description": seed.description,
        "teleology": seed.teleology,
        "context": seed.context,
        "characters": [
            {"name": c.name, "description": c.description}
            for c in chars_result.scalars().all()
        ],
        "narrative_threads": [
            {"thread": t.thread} for t in threads_result.scalars().all()
        ],
        "created_at": seed.created_at.isoformat() if seed.created_at else None,
    }
