"""Shared FastAPI dependencies — service singletons and provider access."""

from __future__ import annotations

import logging
from typing import Optional

from playwriter.config import settings
from playwriter.llm.registry import get_provider
from playwriter.prompts.loader import PromptLoader
from playwriter.services.character import CharacterService
from playwriter.services.game import GameService
from playwriter.services.pipeline import PipelineService
from playwriter.services.scene import SceneService
from playwriter.services.seeding import SeedingService
from playwriter.services.trope import TropeService

log = logging.getLogger(__name__)

# --- Singletons ---

_prompts = PromptLoader()
_trope_service = TropeService()

# Provider, model & tier can be switched at runtime via the /providers endpoint
_active_provider: str = settings.default_provider
_active_model: Optional[str] = None  # None = use tier default


def set_active_provider(name: str) -> None:
    global _active_provider
    _active_provider = name
    log.info("Active provider set to: %s", name)


def get_active_provider() -> str:
    return _active_provider


def set_active_model(model: Optional[str]) -> None:
    global _active_model
    _active_model = model
    log.info("Active model set to: %s", model or "(tier default)")


def get_active_model() -> Optional[str]:
    return _active_model


def _strong():
    return get_provider(_active_provider, tier="strong", model=_active_model)


def _fast():
    # Only use active_model for fast if it's explicitly a fast model;
    # otherwise fall back to provider's fast default
    return get_provider(_active_provider, tier="fast")


def get_seeding_service() -> SeedingService:
    return SeedingService(_strong(), _prompts)


def get_character_service() -> CharacterService:
    return CharacterService(_strong(), _fast(), _prompts)


def get_scene_service() -> SceneService:
    return SceneService(_strong(), _fast(), _prompts)


def get_game_service() -> GameService:
    return GameService(_strong(), _fast(), _prompts)


def get_trope_service() -> TropeService:
    return _trope_service


def get_pipeline_service() -> PipelineService:
    return PipelineService(_strong(), _fast(), _prompts, _trope_service)
