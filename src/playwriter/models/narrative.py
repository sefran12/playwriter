"""Narrative Engine models — multi-scale temporal narrative generation.

Scales:
  - Beat (small):  individual character actions with stochastic resolution
  - Scene (meso):  composed scenes with thread tracking and character updates
  - Act (large):   world events, teleology shifts, act planning
  - World (top):   the complete evolving narrative state
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from playwriter.models.character import Character
from playwriter.models.story import TCCN, NarrativeThread
from playwriter.models.trope import Trope, TropeSample


# ─── Dice / Stochastic System ───────────────────────────────────────────


class DiceOutcome(str, Enum):
    """Five-tier outcome ladder for action resolution."""

    CATASTROPHIC_FAILURE = "catastrophic_failure"
    FAILURE = "failure"
    MIXED = "mixed"
    SUCCESS = "success"
    CRITICAL_SUCCESS = "critical_success"


class FateModifier(BaseModel):
    """A trope applied as a fate modifier to a dice roll."""

    trope: Trope
    modifier: int = Field(
        default=0,
        ge=-30,
        le=30,
        description="Signed integer modifier to the d100 roll",
    )
    rationale: str = ""


class DiceRoll(BaseModel):
    """Complete record of a stochastic action resolution."""

    raw_roll: int = Field(ge=1, le=100, description="The raw d100 roll")
    fate_modifiers: List[FateModifier] = Field(default_factory=list)
    final_value: int = Field(
        description="raw_roll + sum of modifiers, clamped 1-100"
    )
    outcome: DiceOutcome
    action_description: str = ""
    actor: str = ""


# ─── Beat (Small Scale) ─────────────────────────────────────────────────


class CharacterDelta(BaseModel):
    """Changes to apply to a character after a beat resolves."""

    character_name: str = ""
    new_short_term_memories: List[str] = Field(default_factory=list)
    new_long_term_memories: List[str] = Field(default_factory=list)
    internal_state_shift: str = ""
    ambition_shift: str = ""
    contradiction_shifts: List[str] = Field(default_factory=list)
    physical_state_change: str = ""


class Beat(BaseModel):
    """The smallest narrative unit: a single character action with
    stochastic resolution and prose output."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    scene_id: str = ""
    sequence: int = Field(default=0, description="Order within the scene")
    actor: str = ""
    intended_action: str = ""
    dice_roll: Optional[DiceRoll] = None
    actual_outcome: str = Field(
        default="",
        description="What actually happened after dice and trope modifiers",
    )
    prose: str = Field(default="", description="Theatrical prose for this beat")
    character_deltas: List[CharacterDelta] = Field(default_factory=list)
    tropes_active: List[Trope] = Field(
        default_factory=list,
        description="Tropes active as fate during this beat",
    )


# ─── Narrative Thread Tracking ───────────────────────────────────────────


class NarrativeThreadState(BaseModel):
    """Tracks the lifecycle of a single narrative thread."""

    thread: NarrativeThread
    status: Literal[
        "active", "advancing", "stalled", "resolved", "spawned"
    ] = "active"
    tension_level: int = Field(default=5, ge=1, le=10)
    notes: str = ""


# ─── Scene (Meso Scale) ─────────────────────────────────────────────────


class EngineScene(BaseModel):
    """An extended Scene for the Narrative Engine with beat-level tracking."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    act_id: str = ""
    number: int = 0
    actors: List[str] = Field(default_factory=list)
    setting: str = ""
    place_description: str = ""
    narrative_threads: List[NarrativeThreadState] = Field(default_factory=list)
    tropes_injected: TropeSample = Field(
        default_factory=lambda: TropeSample()
    )
    beats: List[Beat] = Field(default_factory=list)
    scene_evaluation: str = ""
    full_prose: str = ""
    status: Literal["planned", "composing", "in_progress", "completed"] = "planned"


# ─── Act (Large Scale) ──────────────────────────────────────────────────


class WorldEvent(BaseModel):
    """A large-scale event that shifts the world context."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    impact_on_context: str = ""
    affected_characters: List[str] = Field(default_factory=list)
    affected_threads: List[str] = Field(default_factory=list)
    spawned_threads: List[NarrativeThread] = Field(default_factory=list)


class TeleologyShift(BaseModel):
    """Records how the teleology evolves across acts."""

    original: str = ""
    shifted: str = ""
    reason: str = ""


class ActPlan(BaseModel):
    """Plan for an act: which scenes to run and with what goals."""

    planned_scenes: List[str] = Field(
        default_factory=list,
        description="Brief descriptions of intended scenes",
    )
    thread_goals: Dict[str, str] = Field(
        default_factory=dict,
        description="thread_text -> what should happen in this act",
    )
    character_arcs: Dict[str, str] = Field(
        default_factory=dict,
        description="character_name -> intended development this act",
    )
    world_events_planned: List[str] = Field(default_factory=list)


class Act(BaseModel):
    """A full act of the narrative — the large-scale container."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    number: int = 0
    title: str = ""
    plan: Optional[ActPlan] = None
    scenes: List[EngineScene] = Field(default_factory=list)
    world_events: List[WorldEvent] = Field(default_factory=list)
    teleology_shift: Optional[TeleologyShift] = None
    context_evolution: str = Field(
        default="",
        description="How the world context changed during this act",
    )
    status: Literal["planned", "in_progress", "completed"] = "planned"


# ─── Director Mode ──────────────────────────────────────────────────────


class EngineMode(str, Enum):
    AUTONOMOUS = "autonomous"
    DIRECTOR = "director"


class DirectorIntervention(BaseModel):
    """A record of a director override or choice."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    intervention_type: Literal[
        "override_dice",
        "choose_thread",
        "redirect_character",
        "inject_event",
        "skip_scene",
        "modify_plan",
        "force_trope",
    ] = "override_dice"
    description: str = ""
    data: Dict = Field(default_factory=dict)


# ─── NarrativeWorld (Top-Level State) ────────────────────────────────────


class NarrativeWorld(BaseModel):
    """The complete state of a running Narrative Engine session."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    seed_description: str = ""
    tccn: Optional[TCCN] = None
    characters: Dict[str, Character] = Field(default_factory=dict)
    acts: List[Act] = Field(default_factory=list)
    current_act_index: int = 0
    current_scene_index: int = 0
    current_beat_index: int = 0
    thread_states: List[NarrativeThreadState] = Field(default_factory=list)
    global_trope_pool: List[Trope] = Field(
        default_factory=list,
        description="Tropes pre-sampled for the full run",
    )
    mode: EngineMode = EngineMode.AUTONOMOUS
    director_interventions: List[DirectorIntervention] = Field(
        default_factory=list
    )
    accumulated_prose: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: str = "initializing"
