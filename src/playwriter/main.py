"""Playwriter — AI-driven narrative generation API.

Run with:  uvicorn playwriter.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Configure logging for all playwriter modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from playwriter.api.arena import router as arena_router
from playwriter.api.characters import router as characters_router
from playwriter.api.games import router as games_router
from playwriter.api.pipeline import router as pipeline_router
from playwriter.api.providers import router as providers_router
from playwriter.api.scenes import router as scenes_router
from playwriter.api.seeds import router as seeds_router
from playwriter.api.tropes import router as tropes_router
from playwriter.db.database import init_db

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    await init_db()
    yield


app = FastAPI(
    title="Playwriter",
    description=(
        "Complex storyteller with agent federation and multiple world "
        "representation threads, stochastic events and state management."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── API routers ──────────────────────────────────────────────────────────
app.include_router(seeds_router)
app.include_router(characters_router)
app.include_router(scenes_router)
app.include_router(tropes_router)
app.include_router(games_router)
app.include_router(pipeline_router)
app.include_router(providers_router)
app.include_router(arena_router)

# ── Static files (css, js) ──────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── HTML entry point ────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the Character Arena frontend."""
    return FileResponse(str(_STATIC_DIR / "html" / "index.html"))
