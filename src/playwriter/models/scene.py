from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SceneAction(BaseModel):
    """A single action within a scene."""

    actor: str
    action: str


class Scene(BaseModel):
    """A scene in the play with actors, setting, threads, and actions."""

    number: int
    actors: List[str] = Field(default_factory=list)
    setting: str = ""
    narrative_threads: List[str] = Field(default_factory=list)
    actions: List[SceneAction] = Field(default_factory=list)
    tropes_used: List[str] = Field(
        default_factory=list,
        description="Tropes injected as 'literary fate' that shaped this scene.",
    )


class Place(BaseModel):
    """A location / setting parsed from scene design."""

    name: str
    description: str
