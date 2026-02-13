from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Character(BaseModel):
    """Full character profile following the HPPTI framework.

    H = History (captured in internal_state + long_term_memory)
    P = Physical (physical_state)
    P = Philosophy (philosophy)
    T = Teleology (teleology / ambitions)
    I = Internal contradictions
    """

    name: str = ""
    internal_state: str = ""
    ambitions: str = ""
    teleology: str = ""
    philosophy: str = ""
    physical_state: str = ""
    long_term_memory: List[str] = Field(default_factory=list)
    short_term_memory: List[str] = Field(default_factory=list)
    internal_contradictions: List[str] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        """Render as plain text for prompt injection."""
        parts = [
            f"Name: {self.name}",
            f"Internal State: {self.internal_state}",
            f"Ambitions: {self.ambitions}",
            f"Teleology: {self.teleology}",
            f"Philosophy: {self.philosophy}",
            f"Physical State: {self.physical_state}",
            f"Long-Term Memory: {'; '.join(self.long_term_memory)}",
            f"Short-Term Memory: {'; '.join(self.short_term_memory)}",
            f"Internal Contradictions: {'; '.join(self.internal_contradictions)}",
        ]
        return "\n".join(parts)
