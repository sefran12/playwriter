from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Trope(BaseModel):
    """A single TV-trope entry."""

    trope_id: str
    name: str
    description: str


class TropeSample(BaseModel):
    """A collection of sampled tropes with provenance."""

    tropes: List[Trope] = Field(default_factory=list)
    source: str = "random"  # "random" | "filtered" | "thematic" | "by_media"

    def to_prompt_text(self) -> str:
        """Format tropes as the 'literary fate' injection for scene prompts."""
        lines = [
            f"- {t.name}: {t.description}" for t in self.tropes
        ]
        return "\n".join(lines)
