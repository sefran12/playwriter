from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class NarrativeThread(BaseModel):
    """A single narrative thread expressed as a trope.

    Format: ACTION between ACTORS in CONTEXT help attain TELEOLOGY: REASON
    """

    thread: str


class CharacterSummary(BaseModel):
    """Lightweight character reference used inside a TCCN seed."""

    name: str
    description: str


class TCCN(BaseModel):
    """Teleology - Context - Characters - Narrative-threads.

    The fundamental story-seed structure that drives the entire Playwriter
    pipeline.
    """

    teleology: str = Field(
        ...,
        description=(
            "The ultimate finality of the play - fate, moral, or ethical teaching."
        ),
    )
    context: str = Field(
        ...,
        description="World-building background where the play develops.",
    )
    characters: List[CharacterSummary] = Field(
        default_factory=list,
        min_length=1,
        description="Actors that populate the world (aim for 10+).",
    )
    narrative_threads: List[NarrativeThread] = Field(
        default_factory=list,
        min_length=1,
        description="Trope-based narrative threads serving the teleology (aim for 10+).",
    )

    def to_prompt_text(self) -> str:
        """Render the TCCN as a plain-text block suitable for prompt injection."""
        chars = "\n".join(
            f"  {i + 1}. {c.name}: {c.description}"
            for i, c in enumerate(self.characters)
        )
        threads = "\n".join(
            f"  {i + 1}. {t.thread}" for i, t in enumerate(self.narrative_threads)
        )
        return (
            f"TELEOLOGY:\n{self.teleology}\n\n"
            f"CONTEXT:\n{self.context}\n\n"
            f"CHARACTERS:\n{chars}\n\n"
            f"NARRATIVE THREADS:\n{threads}"
        )
