"""Multi-scale temporal narrative engine.

The NarrativeEngine is the conductor that composes all existing services
into a temporal narrative loop operating at three scales:

    LARGE  (Act)   — world events, teleology shifts, context evolution
    MESO   (Scene) — scene composition, thread advancement, character updates
    SMALL  (Beat)  — character actions resolved by d100 dice + trope fate

The LLM never decides whether actions succeed — random dice rolls do.
The LLM narrates the outcome it is told, within the constraints of the
stochastic result.  Tropes act as "fate modifiers" that shift the roll
±30, creating thematic texture without overriding randomness.

Feedback flows upward (beat → scene → act) and guidance flows downward
(act plan → scene composition → beat generation).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple

from playwriter.llm.base import LLMProvider
from playwriter.models.character import Character
from playwriter.models.narrative import (
    Act,
    ActPlan,
    Beat,
    CharacterDelta,
    DirectorIntervention,
    EngineMode,
    EngineScene,
    NarrativeThreadState,
    NarrativeWorld,
    TeleologyShift,
    WorldEvent,
)
from playwriter.models.story import TCCN, CharacterSummary, NarrativeThread
from playwriter.models.trope import Trope, TropeSample
from playwriter.parsing.output_parser import OutputParser
from playwriter.prompts.loader import PromptLoader
from playwriter.services.character import CharacterService
from playwriter.services.dice import DiceService
from playwriter.services.scene import SceneService
from playwriter.services.seeding import SeedingService
from playwriter.services.trope import TropeService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_loads(text: str) -> dict | list:
    """Extract JSON from LLM output, tolerating markdown fences and prose."""
    # Try fenced code block first
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())
    # Try raw JSON
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    return json.loads(text.strip())


class NarrativeEngine:
    """Conductor that composes existing services into temporal narrative generation.

    Never touches LLMs directly for operations that existing services handle.
    Uses SeedingService, CharacterService, SceneService, TropeService, and
    DiceService as building blocks.
    """

    def __init__(
        self,
        strong_llm: LLMProvider,
        fast_llm: LLMProvider | None = None,
        prompts: PromptLoader | None = None,
        trope_service: TropeService | None = None,
    ):
        self._strong = strong_llm
        self._fast = fast_llm or strong_llm
        self._prompts = prompts or PromptLoader()
        self._tropes = trope_service or TropeService()

        # Compose sub-services
        self._seeding = SeedingService(llm=strong_llm, prompts=self._prompts)
        self._characters = CharacterService(
            strong_llm=strong_llm, fast_llm=self._fast, prompts=self._prompts,
        )
        self._scenes = SceneService(
            strong_llm=strong_llm, fast_llm=self._fast, prompts=self._prompts,
        )
        self._dice = DiceService(
            fast_llm=self._fast, trope_service=self._tropes, prompts=self._prompts,
        )

        # In-memory world store: world_id → NarrativeWorld
        self._worlds: Dict[str, NarrativeWorld] = {}

    # ── helpers ─────────────────────────────────────────────────────────

    def _get_world(self, world_id: str) -> NarrativeWorld:
        world = self._worlds.get(world_id)
        if world is None:
            raise ValueError(f"No narrative world with id '{world_id}'")
        return world

    def _current_act(self, world: NarrativeWorld) -> Act:
        return world.acts[world.current_act_index]

    def _current_scene(self, world: NarrativeWorld) -> EngineScene:
        act = self._current_act(world)
        return act.scenes[world.current_scene_index]

    def _thread_states_text(self, world: NarrativeWorld) -> str:
        lines = []
        for ts in world.thread_states:
            lines.append(
                f"- [{ts.status.upper()}] (tension {ts.tension_level}/10) "
                f"{ts.thread}"
            )
        return "\n".join(lines) or "(no threads yet)"

    def _characters_text(self, world: NarrativeWorld) -> str:
        parts = []
        for name, char in world.characters.items():
            parts.append(char.to_prompt_text())
        return "\n\n---\n\n".join(parts) or "(no characters)"

    def _accumulated_events_text(self, world: NarrativeWorld) -> str:
        events = []
        for act in world.acts:
            for we in act.world_events:
                events.append(f"- {we.description}")
            for scene in act.scenes:
                for beat in scene.beats:
                    events.append(f"- [Beat] {beat.actual_outcome}")
        return "\n".join(events[-30:]) or "(no events yet)"

    def _act_summaries_text(self, world: NarrativeWorld) -> str:
        parts = []
        for act in world.acts:
            if act.status == "completed":
                summary = act.context_evolution or f"Act {act.number}: {act.title}"
                parts.append(f"Act {act.number} — {summary[:300]}")
        return "\n".join(parts) or "(no completed acts)"

    # =====================================================================
    #  INITIALIZATION
    # =====================================================================

    async def initialize_world(
        self,
        seed_description: str,
        mode: str = "autonomous",
        trope_pool_size: int = 30,
        num_characters: Optional[int] = None,
        on_progress: Optional[callable] = None,
    ) -> NarrativeWorld:
        """Create a new narrative world from a seed description.

        Pipeline:
        1. SeedingService → TCCN seed
        2. CharacterService → generate + refine for each character
        3. TropeService → sample global trope pool
        4. Initialize thread states from TCCN narrative threads

        Parameters
        ----------
        on_progress:
            Optional async callback ``(step, detail) -> None`` called at each
            major step so callers (e.g. SSE endpoints) can stream progress.
        """
        async def _progress(step: str, detail: str = ""):
            log.info("World init: %s — %s", step, detail)
            if on_progress:
                await on_progress(step, detail)

        world_id = uuid.uuid4().hex[:12]
        await _progress("starting", f"World {world_id} from: {seed_description[:60]}...")

        # 1. Generate TCCN seed
        await _progress("generating_seed", "Generating world seed (TCCN)...")
        tccn = await self._seeding.generate_seed(seed_description)
        char_names = ", ".join(cs.name for cs in tccn.characters)
        first_thread = tccn.narrative_threads[0].thread[:100] if tccn.narrative_threads else "(none)"
        await _progress("seed_ready",
                        f"Teleology: {tccn.teleology[:120]}...\n"
                        f"Characters: {char_names}\n"
                        f"Threads ({len(tccn.narrative_threads)}): {first_thread}...")

        # 2. Generate characters
        chars_to_gen = tccn.characters
        if num_characters:
            chars_to_gen = chars_to_gen[:num_characters]

        characters: Dict[str, Character] = {}
        total = len(chars_to_gen)
        for i, cs in enumerate(chars_to_gen):
            await _progress("generating_character",
                            f"Generating character {i+1}/{total}: {cs.name}...")
            char = await self._characters.generate(tccn, cs)
            await _progress("refining_character",
                            f"Refining character {i+1}/{total}: {cs.name}...")
            char = await self._characters.refine(char, tccn, rounds=1)
            characters[char.name] = char
            voice_snip = (char.voice_style[:120] + "...") if char.voice_style else "(no voice style)"
            philo_snip = (char.philosophy[:120] + "...") if char.philosophy else "(no philosophy)"
            await _progress("character_ready",
                            f"Character ready: {char.name} ({i+1}/{total})\n"
                            f"Voice: {voice_snip}\n"
                            f"Philosophy: {philo_snip}")

        # 3. Sample global trope pool
        await _progress("sampling_tropes", f"Sampling {trope_pool_size} tropes...")
        trope_sample = self._tropes.sample_random(n=trope_pool_size)
        await _progress("tropes_ready", f"{len(trope_sample.tropes)} tropes in pool")

        # 4. Initialize thread states from TCCN
        thread_states = [
            NarrativeThreadState(
                thread=nt,
                status="active",
                tension_level=3,
            )
            for nt in tccn.narrative_threads
        ]

        # 5. Build world
        engine_mode = EngineMode.DIRECTOR if mode == "director" else EngineMode.AUTONOMOUS
        world = NarrativeWorld(
            id=world_id,
            seed_description=seed_description,
            tccn=tccn,
            characters=characters,
            acts=[],
            current_act_index=0,
            current_scene_index=0,
            current_beat_index=0,
            thread_states=thread_states,
            global_trope_pool=trope_sample.tropes,
            mode=engine_mode,
            director_interventions=[],
            accumulated_prose="",
            status="initialized",
        )
        self._worlds[world_id] = world
        await _progress("complete",
                        f"World ready: {len(characters)} characters, "
                        f"{len(thread_states)} threads, mode={mode}")
        return world

    # =====================================================================
    #  LARGE SCALE — Act
    # =====================================================================

    async def plan_act(self, world_id: str) -> Act:
        """Plan the next act from the current world state."""
        world = self._get_world(world_id)
        act_number = len(world.acts) + 1
        log.info("Planning act %d for world %s", act_number, world_id)

        # Gather previous act summary
        prev_summary = ""
        if world.acts:
            prev_act = world.acts[-1]
            prev_summary = prev_act.context_evolution or f"Act {prev_act.number} completed."

        prompt = self._prompts.render(
            "generators", "ACT_PLANNER",
            teleology=world.tccn.teleology,
            context=world.tccn.context,
            thread_states=self._thread_states_text(world),
            previous_act_summary=prev_summary or "(This is the first act)",
            characters_summary=", ".join(world.characters.keys()),
            act_number=str(act_number),
        )

        raw = await self._strong.complete(
            system_prompt="You are a master dramaturg planning the next act of a play.",
            user_prompt=prompt,
            json_mode=True,
            max_tokens=2048,
        )

        try:
            data = _safe_json_loads(raw)
            plan = ActPlan(
                planned_scenes=data.get("planned_scenes", []),
                thread_goals={str(k): str(v) for k, v in data.get("thread_goals", {}).items()},
                character_arcs={str(k): str(v) for k, v in data.get("character_arcs", {}).items()},
                world_events_planned=data.get("world_events_planned", []),
            )
        except Exception as exc:
            log.warning("Act plan parse failed, using minimal plan: %s", exc)
            plan = ActPlan(
                planned_scenes=[f"Scene {i+1}" for i in range(3)],
                thread_goals={},
                character_arcs={},
                world_events_planned=[],
            )

        act = Act(
            id=uuid.uuid4().hex[:12],
            number=act_number,
            title=data.get("title", f"Act {act_number}") if isinstance(data, dict) else f"Act {act_number}",
            plan=plan,
            scenes=[],
            world_events=[],
            teleology_shift=None,
            context_evolution="",
            status="planned",
        )
        world.acts.append(act)
        world.current_act_index = len(world.acts) - 1
        world.current_scene_index = 0
        world.current_beat_index = 0
        world.status = "act_planned"
        log.info("Act %d planned: %d scenes, title=%s",
                 act_number, len(plan.planned_scenes), act.title)
        return act

    async def generate_world_events(self, world_id: str) -> List[WorldEvent]:
        """Generate disruptive world events after an act completes."""
        world = self._get_world(world_id)
        act = self._current_act(world)

        # Build act summary from scene beats
        beat_summaries = []
        for scene in act.scenes:
            for beat in scene.beats:
                beat_summaries.append(f"- {beat.actor}: {beat.actual_outcome}")
        act_summary = "\n".join(beat_summaries[-20:]) or "(no beats yet)"

        # Sample a trope for the world event
        trope_sample = self._tropes.sample_random(n=2)
        trope_text = trope_sample.to_prompt_text()

        prompt = self._prompts.render(
            "generators", "WORLD_EVENT_GENERATOR",
            context=world.tccn.context,
            teleology=world.tccn.teleology,
            trope_injection=trope_text,
            act_summary=act_summary,
            thread_states=self._thread_states_text(world),
        )

        raw = await self._strong.complete(
            system_prompt="You generate world-shaping events for a narrative.",
            user_prompt=prompt,
            json_mode=True,
            max_tokens=2048,
        )

        events: List[WorldEvent] = []
        try:
            data = _safe_json_loads(raw)
            items = data if isinstance(data, list) else data.get("events", [data])
            for item in items:
                events.append(WorldEvent(
                    description=item.get("description", ""),
                    impact_on_context=item.get("impact_on_context", ""),
                    affected_characters=item.get("affected_characters", []),
                    affected_threads=item.get("affected_threads", []),
                    spawned_threads=item.get("spawned_threads", []),
                ))
        except Exception as exc:
            log.warning("World event parse failed: %s", exc)

        act.world_events.extend(events)
        log.info("Generated %d world events for act %d", len(events), act.number)
        return events

    async def evaluate_teleology_shift(self, world_id: str) -> Optional[TeleologyShift]:
        """Assess whether accumulated events have fundamentally shifted the teleology."""
        world = self._get_world(world_id)

        # Gather thread resolutions
        resolved = [ts for ts in world.thread_states if ts.status == "resolved"]
        thread_resolutions = "\n".join(
            f"- {ts.thread}" for ts in resolved
        ) or "(no threads resolved yet)"

        prompt = self._prompts.render(
            "generators", "TELEOLOGY_SHIFT_EVALUATOR",
            original_teleology=world.tccn.teleology,
            accumulated_events=self._accumulated_events_text(world),
            thread_resolutions=thread_resolutions,
            act_summaries=self._act_summaries_text(world),
        )

        raw = await self._strong.complete(
            system_prompt="You are a dramaturgical evaluator assessing teleological shifts.",
            user_prompt=prompt,
            json_mode=True,
            max_tokens=1024,
        )

        try:
            data = _safe_json_loads(raw)
            shifted = bool(data.get("shifted", False))
            if shifted:
                shift = TeleologyShift(
                    original=world.tccn.teleology,
                    shifted=data.get("new_teleology", ""),
                    reason=data.get("reason", ""),
                )
                # Apply the shift
                world.tccn.teleology = shift.shifted
                act = self._current_act(world)
                act.teleology_shift = shift
                log.info("Teleology shifted: %s → %s", shift.original[:50], shift.shifted[:50])
                return shift
            else:
                log.info("Teleology remains unchanged: %s", data.get("reason", "")[:80])
                return None
        except Exception as exc:
            log.warning("Teleology shift evaluation failed: %s", exc)
            return None

    async def update_context(self, world_id: str) -> str:
        """Evolve the world context after an act completes."""
        world = self._get_world(world_id)
        act = self._current_act(world)

        # Build summaries
        beat_summaries = []
        for scene in act.scenes:
            for beat in scene.beats:
                beat_summaries.append(f"- {beat.actor}: {beat.actual_outcome}")
        act_summary = "\n".join(beat_summaries[-20:]) or "(no events)"

        world_events_text = "\n".join(
            f"- {we.description}: {we.impact_on_context}" for we in act.world_events
        ) or "(no world events)"

        thread_changes = self._thread_states_text(world)

        prompt = self._prompts.render(
            "updaters", "CONTEXT_UPDATER",
            current_context=world.tccn.context,
            act_summary=act_summary,
            world_events=world_events_text,
            thread_changes=thread_changes,
            teleology=world.tccn.teleology,
        )

        new_context = await self._strong.complete(
            system_prompt="You evolve a play's world context after an act.",
            user_prompt=prompt,
            max_tokens=1024,
        )

        # Apply update
        world.tccn.context = new_context.strip()
        act.context_evolution = new_context.strip()
        log.info("Context evolved for act %d", act.number)
        return new_context.strip()

    async def complete_act(self, world_id: str) -> Act:
        """Finalize the current act: world events → teleology check → context evolution."""
        world = self._get_world(world_id)
        act = self._current_act(world)
        log.info("Completing act %d for world %s", act.number, world_id)

        # 1. Generate world events
        await self.generate_world_events(world_id)

        # 2. Evaluate teleology shift
        await self.evaluate_teleology_shift(world_id)

        # 3. Evolve context
        await self.update_context(world_id)

        act.status = "completed"
        world.status = "act_completed"
        log.info("Act %d completed: %d scenes, %d world events",
                 act.number, len(act.scenes), len(act.world_events))
        return act

    # =====================================================================
    #  MESO SCALE — Scene
    # =====================================================================

    async def compose_next_scene(self, world_id: str) -> EngineScene:
        """Compose the next scene within the current act."""
        world = self._get_world(world_id)
        act = self._current_act(world)
        scene_number = len(act.scenes) + 1
        log.info("Composing scene %d in act %d", scene_number, act.number)

        # Sample tropes for this scene
        trope_sample = self._tropes.sample_random(n=3)
        trope_text = trope_sample.to_prompt_text()

        # Act plan context
        act_plan_text = ""
        if act.plan:
            idx = scene_number - 1
            if idx < len(act.plan.planned_scenes):
                act_plan_text = act.plan.planned_scenes[idx]
            act_plan_text += "\nThread goals: " + json.dumps(act.plan.thread_goals)
            act_plan_text += "\nCharacter arcs: " + json.dumps(act.plan.character_arcs)

        prompt = self._prompts.render(
            "generators", "ENGINE_SCENE_COMPOSER",
            tcc_context=world.tccn.to_prompt_text(),
            act_plan=act_plan_text or "(no specific plan for this scene)",
            thread_states=self._thread_states_text(world),
            trope_injection=trope_text,
            characters_summary=self._characters_text(world),
            scene_number=str(scene_number),
            act_number=str(act.number),
        )

        raw = await self._strong.complete(
            system_prompt="You are a master scene architect composing a single scene.",
            user_prompt=prompt,
            json_mode=True,
            max_tokens=2048,
        )

        try:
            data = _safe_json_loads(raw)
            actors = data.get("actors", list(world.characters.keys())[:3])
            setting = data.get("setting", "An unspecified location")
            place_description = data.get("place_description", setting)
            scene_threads = data.get("narrative_threads", [])
        except Exception as exc:
            log.warning("Scene compose parse failed, using defaults: %s", exc)
            actors = list(world.characters.keys())[:3]
            setting = "A place in the world"
            place_description = setting
            scene_threads = []

        # Map thread texts to NarrativeThreadState
        thread_states_for_scene: List[NarrativeThreadState] = []
        for ts in world.thread_states:
            if ts.status != "resolved":
                thread_states_for_scene.append(ts.model_copy())

        scene = EngineScene(
            id=uuid.uuid4().hex[:12],
            act_id=act.id,
            number=scene_number,
            actors=actors,
            setting=setting,
            place_description=place_description,
            narrative_threads=thread_states_for_scene,
            tropes_injected=trope_sample,
            beats=[],
            scene_evaluation="",
            full_prose="",
            status="composing",
        )
        act.scenes.append(scene)
        world.current_scene_index = len(act.scenes) - 1
        world.current_beat_index = 0
        world.status = "scene_composing"
        log.info("Scene %d composed: actors=%s, setting=%s",
                 scene_number, actors, setting[:60])
        return scene

    async def update_characters_after_scene(
        self, world_id: str, scene_id: str,
    ) -> Dict[str, Character]:
        """Apply accumulated character deltas from a scene's beats."""
        world = self._get_world(world_id)

        # Find the scene
        scene = self._find_scene(world, scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")

        # Collect deltas per character
        deltas_by_char: Dict[str, List[CharacterDelta]] = {}
        beats_by_char: Dict[str, List[str]] = {}
        for beat in scene.beats:
            for delta in beat.character_deltas:
                deltas_by_char.setdefault(delta.character_name, []).append(delta)
            beats_by_char.setdefault(beat.actor, []).append(beat.actual_outcome)

        updated: Dict[str, Character] = {}
        for char_name, deltas in deltas_by_char.items():
            if char_name not in world.characters:
                continue

            char = world.characters[char_name]
            beats_text = "\n".join(
                f"- {b}" for b in beats_by_char.get(char_name, [])
            )
            deltas_text = json.dumps(
                [d.model_dump() for d in deltas], indent=2
            )

            prompt = self._prompts.render(
                "updaters", "CHARACTER_STATE_UPDATER",
                character_profile=char.to_prompt_text(),
                scene_beats_summary=beats_text or "(no direct beats)",
                character_deltas=deltas_text,
                tcc_context=world.tccn.to_prompt_text(),
            )

            raw = await self._strong.complete(
                system_prompt="You integrate character changes into a living profile.",
                user_prompt=prompt,
                json_mode=True,
                max_tokens=2048,
            )

            try:
                updated_char = OutputParser.parse(raw, Character)
                updated_char.name = updated_char.name or char_name
                world.characters[char_name] = updated_char
                updated[char_name] = updated_char
                log.info("Character updated: %s", char_name)
            except Exception as exc:
                log.warning("Character update failed for %s: %s", char_name, exc)
                updated[char_name] = char

        return updated

    async def advance_thread_states(
        self, world_id: str, scene_id: str,
    ) -> List[NarrativeThreadState]:
        """Update narrative thread states after a scene completes."""
        world = self._get_world(world_id)
        scene = self._find_scene(world, scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")

        # Build scene summary and beat outcomes
        beat_outcomes = "\n".join(
            f"- {b.actor}: {b.actual_outcome} [{b.dice_roll.outcome.value if b.dice_roll else 'N/A'}]"
            for b in scene.beats
        )
        scene_summary = f"Scene {scene.number}: {scene.setting}. Actors: {', '.join(scene.actors)}"

        # Character changes summary
        char_changes = []
        for beat in scene.beats:
            for delta in beat.character_deltas:
                if delta.internal_state_shift:
                    char_changes.append(f"- {delta.character_name}: {delta.internal_state_shift}")
        char_changes_text = "\n".join(char_changes) or "(no significant character changes)"

        prompt = self._prompts.render(
            "generators", "THREAD_STATE_ADVANCER",
            thread_states=self._thread_states_text(world),
            scene_summary=scene_summary,
            beat_outcomes=beat_outcomes or "(no beats)",
            character_changes=char_changes_text,
        )

        raw = await self._fast.complete(
            system_prompt="You track narrative thread evolution across scenes.",
            user_prompt=prompt,
            json_mode=True,
            max_tokens=2048,
        )

        try:
            data = _safe_json_loads(raw)
            items = data if isinstance(data, list) else data.get("threads", [])

            new_states: List[NarrativeThreadState] = []
            for item in items:
                status = item.get("status", "active")
                if status not in ("active", "advancing", "stalled", "resolved", "spawned"):
                    status = "active"
                new_states.append(NarrativeThreadState(
                    thread=item.get("thread", ""),
                    status=status,
                    tension_level=max(0, min(10, int(item.get("tension_level", 5)))),
                ))

            world.thread_states = new_states
            log.info("Thread states updated: %d threads (%d resolved)",
                     len(new_states), sum(1 for ts in new_states if ts.status == "resolved"))
            return new_states
        except Exception as exc:
            log.warning("Thread state update failed: %s", exc)
            return world.thread_states

    async def complete_scene(self, world_id: str, scene_id: str) -> EngineScene:
        """Finalize a scene: update characters, advance threads, compile prose."""
        world = self._get_world(world_id)
        scene = self._find_scene(world, scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")

        log.info("Completing scene %d (id=%s)", scene.number, scene_id)

        # 1. Update character states from accumulated deltas
        await self.update_characters_after_scene(world_id, scene_id)

        # 2. Advance thread states
        await self.advance_thread_states(world_id, scene_id)

        # 3. Compile full scene prose
        prose_parts = []
        for beat in scene.beats:
            if beat.prose:
                prose_parts.append(beat.prose)
        scene.full_prose = "\n\n".join(prose_parts)

        # Accumulate into world prose
        if scene.full_prose:
            world.accumulated_prose += f"\n\n--- Scene {scene.number} ---\n\n{scene.full_prose}"

        scene.status = "completed"
        world.status = "scene_completed"
        log.info("Scene %d completed: %d beats, %d chars prose",
                 scene.number, len(scene.beats), len(scene.full_prose))
        return scene

    def _find_scene(self, world: NarrativeWorld, scene_id: str) -> Optional[EngineScene]:
        """Find a scene by ID across all acts."""
        for act in world.acts:
            for scene in act.scenes:
                if scene.id == scene_id:
                    return scene
        return None

    # =====================================================================
    #  SMALL SCALE — Beat
    # =====================================================================

    async def generate_beat_actions(
        self, world_id: str, scene_id: str,
    ) -> List[Tuple[str, str]]:
        """Generate 5-10 character actions for a scene.

        Returns a list of (actor_name, intended_action) tuples.
        """
        world = self._get_world(world_id)
        scene = self._find_scene(world, scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")

        # Build actor profiles
        actor_profiles = "\n\n".join(
            world.characters[a].to_prompt_text()
            for a in scene.actors if a in world.characters
        )

        # Act goals
        act = self._current_act(world)
        act_goals = ""
        if act.plan:
            act_goals = json.dumps(act.plan.thread_goals, indent=2)

        prompt = self._prompts.render(
            "generators", "BEAT_ACTION_GENERATOR",
            scene_context=f"Setting: {scene.setting}\nPlace: {scene.place_description}",
            actors_profiles=actor_profiles or "(no actor profiles)",
            act_goals=act_goals or "(no specific act goals)",
            thread_states=self._thread_states_text(world),
            scene_number=str(scene.number),
        )

        raw = await self._strong.complete(
            system_prompt="You generate character actions for theatrical scenes.",
            user_prompt=prompt,
            json_mode=True,
            max_tokens=2048,
        )

        actions: List[Tuple[str, str]] = []
        try:
            data = _safe_json_loads(raw)
            items = data if isinstance(data, list) else data.get("actions", [])
            for item in items:
                actor = item.get("actor", "")
                action = item.get("action", "")
                if actor and action:
                    actions.append((actor, action))
        except Exception as exc:
            log.warning("Beat action generation failed: %s", exc)

        # Fallback: at least one action per actor
        if not actions:
            for actor in scene.actors:
                actions.append((actor, f"{actor} observes the scene cautiously."))

        log.info("Generated %d beat actions for scene %d", len(actions), scene.number)
        return actions

    async def resolve_beat(
        self,
        world_id: str,
        scene_id: str,
        actor: str,
        action: str,
        override_roll: Optional[int] = None,
    ) -> Beat:
        """Resolve a single beat: dice roll → narration → prose → character deltas.

        This is the core small-scale loop:
        1. DiceService resolves the action (roll + trope fate modifiers)
        2. BEAT_RESOLVER narrates what actually happened given the dice outcome
        3. BEAT_PROSE_WRITER writes theatrical prose
        4. BEAT_DELTA_CALCULATOR computes character state changes
        """
        world = self._get_world(world_id)
        scene = self._find_scene(world, scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found")

        beat_sequence = len(scene.beats) + 1
        log.info("Resolving beat %d: %s attempts '%s'", beat_sequence, actor, action[:60])

        # Scene context for the LLM
        scene_context = (
            f"Setting: {scene.setting}\n"
            f"Place: {scene.place_description}\n"
            f"Actors present: {', '.join(scene.actors)}\n"
        )
        # Add previous beats for continuity
        if scene.beats:
            recent = scene.beats[-3:]
            scene_context += "\nRecent events:\n"
            for b in recent:
                scene_context += f"- {b.actor}: {b.actual_outcome}\n"

        # 1. Dice resolution
        dice_roll = await self._dice.resolve_action(
            action=action,
            actor=actor,
            scene_context=scene_context,
            trope_pool=world.global_trope_pool,
            n_tropes=2,
            override_roll=override_roll,
        )

        # Fate modifiers as readable text
        fate_text = "\n".join(
            f"- {fm.trope.name} ({fm.modifier:+d}): {fm.rationale}"
            for fm in dice_roll.fate_modifiers
        ) or "(no fate modifiers)"

        # 2. BEAT_RESOLVER — narrate what actually happened
        actor_profile = ""
        if actor in world.characters:
            actor_profile = world.characters[actor].to_prompt_text()

        # Other characters present
        others = [a for a in scene.actors if a != actor and a in world.characters]
        others_text = "\n".join(
            world.characters[o].to_prompt_text() for o in others[:2]
        ) or "(none)"

        resolver_prompt = self._prompts.render(
            "generators", "BEAT_RESOLVER",
            intended_action=action,
            dice_outcome=dice_roll.outcome.value,
            fate_modifiers_text=fate_text,
            actor=actor,
            actor_profile=actor_profile or "(unknown character)",
            scene_context=scene_context,
            other_characters_present=others_text,
            raw_roll=str(dice_roll.raw_roll),
            final_value=str(dice_roll.final_value),
        )

        actual_outcome = await self._strong.complete(
            system_prompt=(
                "You narrate what ACTUALLY happened given a dice outcome. "
                "You CANNOT override the dice result — only describe HOW it manifests."
            ),
            user_prompt=resolver_prompt,
            max_tokens=1024,
        )
        actual_outcome = actual_outcome.strip()
        log.info("Beat resolved: %s [%s] → %s",
                 actor, dice_roll.outcome.value, actual_outcome[:80])

        # 3. BEAT_PROSE_WRITER — theatrical prose
        previous_prose = ""
        if scene.beats:
            previous_prose = scene.beats[-1].prose[:500] if scene.beats[-1].prose else ""

        prose_prompt = self._prompts.render(
            "generators", "BEAT_PROSE_WRITER",
            actual_outcome=actual_outcome,
            dice_outcome=dice_roll.outcome.value,
            scene_setting=scene.setting,
            previous_prose=previous_prose or "(opening of the scene)",
            actor=actor,
            fate_modifiers_text=fate_text,
        )

        prose = await self._strong.complete(
            system_prompt="You are a master playwright writing theatrical prose.",
            user_prompt=prose_prompt,
            max_tokens=1024,
        )
        prose = prose.strip()

        # 4. BEAT_DELTA_CALCULATOR — character state changes
        deltas = await self._calculate_beat_deltas(
            actor=actor,
            actual_outcome=actual_outcome,
            dice_roll=dice_roll,
            world=world,
            scene_context=scene_context,
            others=others,
        )

        # Active tropes for this beat
        active_tropes = [fm.trope for fm in dice_roll.fate_modifiers]

        beat = Beat(
            id=uuid.uuid4().hex[:12],
            scene_id=scene_id,
            sequence=beat_sequence,
            actor=actor,
            intended_action=action,
            dice_roll=dice_roll,
            actual_outcome=actual_outcome,
            prose=prose,
            character_deltas=deltas,
            tropes_active=active_tropes,
        )
        scene.beats.append(beat)
        world.current_beat_index = len(scene.beats) - 1
        world.status = "beat_resolved"
        return beat

    async def _calculate_beat_deltas(
        self,
        actor: str,
        actual_outcome: str,
        dice_roll,
        world: NarrativeWorld,
        scene_context: str,
        others: List[str],
    ) -> List[CharacterDelta]:
        """Calculate character state changes from a beat outcome."""
        deltas: List[CharacterDelta] = []

        actor_profile = ""
        if actor in world.characters:
            actor_profile = world.characters[actor].to_prompt_text()

        others_text = "\n".join(
            f"- {o}: {world.characters[o].internal_state[:100]}"
            for o in others if o in world.characters
        ) or "(none)"

        prompt = self._prompts.render(
            "generators", "BEAT_DELTA_CALCULATOR",
            actor=actor,
            actor_profile=actor_profile or "(unknown)",
            actual_outcome=actual_outcome,
            dice_outcome=dice_roll.outcome.value,
            other_characters_present=others_text,
            scene_context=scene_context,
        )

        raw = await self._fast.complete(
            system_prompt="You calculate character state changes from narrative events.",
            user_prompt=prompt,
            json_mode=True,
            max_tokens=1024,
        )

        try:
            data = _safe_json_loads(raw)
            deltas.append(CharacterDelta(
                character_name=data.get("character_name", actor),
                new_short_term_memories=data.get("new_short_term_memories", []),
                new_long_term_memories=data.get("new_long_term_memories", []),
                internal_state_shift=data.get("internal_state_shift", ""),
                ambition_shift=data.get("ambition_shift", ""),
                contradiction_shifts=data.get("contradiction_shifts", []),
                physical_state_change=data.get("physical_state_change", ""),
            ))
        except Exception as exc:
            log.warning("Delta calculation failed for %s: %s", actor, exc)
            deltas.append(CharacterDelta(
                character_name=actor,
                new_short_term_memories=[actual_outcome[:200]],
                new_long_term_memories=[],
                internal_state_shift="",
                ambition_shift="",
                contradiction_shifts=[],
                physical_state_change="",
            ))

        return deltas

    # =====================================================================
    #  MAIN LOOP
    # =====================================================================

    async def advance(
        self, world_id: str, steps: int = 1,
    ) -> List[dict]:
        """Advance the narrative by N beats, auto-managing scene/act boundaries.

        If no act exists, plans one.  If no scene exists, composes one.
        If the scene's beats are exhausted, completes the scene.
        If all planned scenes are done, completes the act.

        Returns a list of events: {"type": "beat"|"scene_completed"|"act_completed", ...}
        """
        world = self._get_world(world_id)
        events: List[dict] = []

        for _ in range(steps):
            # Ensure we have an act
            if not world.acts or self._current_act(world).status == "completed":
                act = await self.plan_act(world_id)
                events.append({"type": "act_planned", "act_number": act.number, "title": act.title})

            act = self._current_act(world)

            # Ensure we have a scene
            if not act.scenes or self._current_scene(world).status == "completed":
                # Check if we've exhausted planned scenes
                if act.plan and len(act.scenes) >= len(act.plan.planned_scenes):
                    completed_act = await self.complete_act(world_id)
                    events.append({
                        "type": "act_completed",
                        "act_number": completed_act.number,
                        "world_events": [we.description for we in completed_act.world_events],
                    })
                    # Plan next act
                    new_act = await self.plan_act(world_id)
                    events.append({"type": "act_planned", "act_number": new_act.number, "title": new_act.title})
                    act = new_act

                scene = await self.compose_next_scene(world_id)
                # Generate beat actions for this scene
                actions = await self.generate_beat_actions(world_id, scene.id)
                # Store planned actions on the scene for sequential resolution
                scene._planned_actions = actions  # type: ignore[attr-defined]
                events.append({
                    "type": "scene_composed",
                    "scene_number": scene.number,
                    "actors": scene.actors,
                    "setting": scene.setting,
                    "beat_count": len(actions),
                })

            scene = self._current_scene(world)

            # Get next action to resolve
            planned = getattr(scene, "_planned_actions", [])
            beat_idx = len(scene.beats)

            if beat_idx < len(planned):
                actor, action = planned[beat_idx]
                beat = await self.resolve_beat(world_id, scene.id, actor, action)
                events.append({
                    "type": "beat_resolved",
                    "beat_sequence": beat.sequence,
                    "actor": beat.actor,
                    "intended_action": beat.intended_action,
                    "actual_outcome": beat.actual_outcome,
                    "dice_outcome": beat.dice_roll.outcome.value if beat.dice_roll else None,
                    "raw_roll": beat.dice_roll.raw_roll if beat.dice_roll else None,
                    "final_value": beat.dice_roll.final_value if beat.dice_roll else None,
                    "prose": beat.prose,
                })
            else:
                # Scene beats exhausted — complete the scene
                completed_scene = await self.complete_scene(world_id, scene.id)
                events.append({
                    "type": "scene_completed",
                    "scene_number": completed_scene.number,
                    "beats_count": len(completed_scene.beats),
                })

        return events

    # =====================================================================
    #  DIRECTOR MODE
    # =====================================================================

    async def director_override_dice(
        self, world_id: str, actor: str, action: str, forced_roll: int,
    ) -> Beat:
        """Director forces a specific dice roll for an action."""
        world = self._get_world(world_id)
        scene = self._current_scene(world)

        world.director_interventions.append(DirectorIntervention(
            type="override_dice",
            description=f"Forced roll {forced_roll} for {actor}: {action}",
        ))

        return await self.resolve_beat(
            world_id, scene.id, actor, action, override_roll=forced_roll,
        )

    async def director_inject_event(
        self, world_id: str, event_description: str,
    ) -> WorldEvent:
        """Director injects a world event into the current act."""
        world = self._get_world(world_id)
        act = self._current_act(world)

        event = WorldEvent(
            description=event_description,
            impact_on_context=f"Director-injected: {event_description}",
            affected_characters=list(world.characters.keys()),
            affected_threads=[],
            spawned_threads=[],
        )
        act.world_events.append(event)

        world.director_interventions.append(DirectorIntervention(
            type="inject_event",
            description=event_description,
        ))

        log.info("Director injected event: %s", event_description[:80])
        return event

    async def director_redirect_character(
        self, world_id: str, character_name: str, new_direction: str,
    ) -> Character:
        """Director alters a character's ambitions or internal state."""
        world = self._get_world(world_id)
        if character_name not in world.characters:
            raise ValueError(f"Character '{character_name}' not found")

        char = world.characters[character_name]
        char.ambitions = new_direction
        char.short_term_memory.append(f"[Director] New direction: {new_direction}")

        world.director_interventions.append(DirectorIntervention(
            type="redirect_character",
            description=f"Redirected {character_name}: {new_direction}",
        ))

        log.info("Director redirected %s: %s", character_name, new_direction[:80])
        return char

    async def director_force_trope(
        self, world_id: str, trope_query: str,
    ) -> List[Trope]:
        """Director searches for and injects specific tropes into the pool."""
        world = self._get_world(world_id)
        result = self._tropes.search(trope_query, n=3)

        if result.tropes:
            world.global_trope_pool.extend(result.tropes)
            world.director_interventions.append(DirectorIntervention(
                type="force_trope",
                description=f"Injected tropes: {', '.join(t.name for t in result.tropes)}",
            ))
            log.info("Director forced tropes: %s",
                     [t.name for t in result.tropes])

        return result.tropes

    async def director_choose_thread(
        self, world_id: str, thread_index: int, new_status: str = "advancing",
    ) -> NarrativeThreadState:
        """Director manually sets a thread's status and tension."""
        world = self._get_world(world_id)
        if thread_index < 0 or thread_index >= len(world.thread_states):
            raise ValueError(f"Thread index {thread_index} out of range")

        ts = world.thread_states[thread_index]
        ts.status = new_status
        if new_status == "advancing":
            ts.tension_level = min(10, ts.tension_level + 2)

        world.director_interventions.append(DirectorIntervention(
            type="choose_thread",
            description=f"Set thread {thread_index} to {new_status}: {ts.thread[:60]}",
        ))

        log.info("Director chose thread %d → %s", thread_index, new_status)
        return ts

    # =====================================================================
    #  READ ACCESS
    # =====================================================================

    def get_world(self, world_id: str) -> NarrativeWorld:
        """Public access to world state."""
        return self._get_world(world_id)

    def list_worlds(self) -> List[dict]:
        """List all worlds with summary info."""
        return [
            {
                "id": w.id,
                "seed_description": w.seed_description[:100],
                "status": w.status,
                "mode": w.mode.value,
                "acts": len(w.acts),
                "characters": list(w.characters.keys()),
                "threads": len(w.thread_states),
            }
            for w in self._worlds.values()
        ]

    def delete_world(self, world_id: str) -> None:
        """Remove a world from the store."""
        if world_id in self._worlds:
            del self._worlds[world_id]
            log.info("World %s deleted", world_id)

    def get_dice_history(self, world_id: str) -> List[dict]:
        """Return all dice rolls with fate modifiers."""
        world = self._get_world(world_id)
        rolls = []
        for act in world.acts:
            for scene in act.scenes:
                for beat in scene.beats:
                    if beat.dice_roll:
                        rolls.append({
                            "act": act.number,
                            "scene": scene.number,
                            "beat": beat.sequence,
                            "actor": beat.actor,
                            "action": beat.intended_action,
                            "raw_roll": beat.dice_roll.raw_roll,
                            "fate_modifiers": [
                                {"trope": fm.trope.name, "modifier": fm.modifier, "rationale": fm.rationale}
                                for fm in beat.dice_roll.fate_modifiers
                            ],
                            "final_value": beat.dice_roll.final_value,
                            "outcome": beat.dice_roll.outcome.value,
                        })
        return rolls

    def get_accumulated_prose(self, world_id: str) -> str:
        """Return all accumulated theatrical prose."""
        world = self._get_world(world_id)
        return world.accumulated_prose
