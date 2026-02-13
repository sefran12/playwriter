from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel
from fastapi import APIRouter, HTTPException

from playwriter.api.dependencies import (
    get_active_provider, set_active_provider,
    get_active_model, set_active_model,
)
from playwriter.llm.registry import list_providers

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/providers", tags=["providers"])


class SetProviderRequest(BaseModel):
    name: str  # "openai" | "anthropic" | "groq"
    model: Optional[str] = None  # explicit model override


@router.get("")
def get_providers():
    """List all configured LLM providers and their models."""
    return {
        "active": get_active_provider(),
        "active_model": get_active_model(),
        "providers": list_providers(),
    }


@router.put("/active")
def switch_provider(body: SetProviderRequest):
    """Switch the active LLM provider (and optionally model) at runtime."""
    info = list_providers()
    if body.name not in info:
        raise HTTPException(400, f"Unknown provider '{body.name}'. Choose from: {list(info)}")
    if not info[body.name]["configured"]:
        raise HTTPException(400, f"Provider '{body.name}' has no API key configured.")
    if body.model and body.model not in info[body.name]["models"]:
        raise HTTPException(
            400,
            f"Model '{body.model}' not available for '{body.name}'. "
            f"Choose from: {info[body.name]['models']}",
        )

    set_active_provider(body.name)
    set_active_model(body.model)  # None means use tier default
    log.info("Switched provider to %s, model=%s", body.name, body.model or "(tier default)")
    return {
        "active": body.name,
        "active_model": body.model,
        "providers": list_providers(),
    }
