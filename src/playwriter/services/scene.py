from __future__ import annotations

import re

from playwriter.llm.base import LLMProvider
from playwriter.models.character import Character
from playwriter.models.scene import Place, Scene, SceneAction
from playwriter.models.story import TCCN
from playwriter.models.trope import TropeSample
from playwriter.prompts.loader import PromptLoader


class SceneService:
    """Scene composition, evaluation, and writing.

    The key innovation is *trope injection*: random tropes from the TV-tropes
    dataset are fed as "literary fate / SCENE TROPES" into the scene-generation
    prompt, forcing the LLM away from generic outputs.
    """

    def __init__(
        self,
        strong_llm: LLMProvider,
        fast_llm: LLMProvider | None = None,
        prompts: PromptLoader | None = None,
    ):
        self._strong = strong_llm
        self._fast = fast_llm or strong_llm
        self._prompts = prompts or PromptLoader()

    async def compose_scenes(
        self,
        tccn: TCCN,
        trope_sample: TropeSample | None = None,
    ) -> list[Scene]:
        """Generate a list of scenes from a TCCN, injecting tropes as literary fate."""
        tcc_text = tccn.to_prompt_text()

        # Inject tropes into the context if provided
        if trope_sample and trope_sample.tropes:
            trope_text = trope_sample.to_prompt_text()
            tcc_text += f"\n\nSCENE TROPES (literary fate — incorporate these):\n{trope_text}"

        prompt = self._prompts.render(
            "generators",
            "INITIAL_SCENE_TCC_GENERATOR",
            tcc_context=tcc_text,
        )
        raw = await self._strong.complete(
            system_prompt="You are a master scene architect for theatrical plays.",
            user_prompt=prompt,
        )
        scenes = self._parse_scenes(raw)

        # Tag each scene with the tropes that were used
        if trope_sample:
            trope_names = [t.name for t in trope_sample.tropes]
            for scene in scenes:
                scene.tropes_used = trope_names

        return scenes

    async def evaluate_scenes(self, tccn: TCCN, scenes: list[Scene]) -> str:
        """Critique the scene progression for coherence and quality."""
        scene_text = self._scenes_to_text(tccn, scenes)
        prompt = self._prompts.render(
            "generators",
            "INITIAL_SCENE_EVALUATOR_TCC_GENERATOR",
            scene_context=scene_text,
        )
        return await self._strong.complete(
            system_prompt="You are a dramaturgical critic evaluating scene construction.",
            user_prompt=prompt,
        )

    async def write_scene(
        self,
        tccn: TCCN,
        scenes: list[Scene],
        characters: list[Character] | None = None,
    ) -> str:
        """Write theatrical prose from scene outlines."""
        scene_text = self._scenes_to_text(tccn, scenes)

        # If we have FIRST_PASS_SCENE_DESIGNER, use it with character context
        if characters:
            char_text = "\n\n".join(c.to_prompt_text() for c in characters)
            prompt = self._prompts.render(
                "generators",
                "FIRST_PASS_SCENE_DESIGNER",
                tcc_context=scene_text,
                character_descriptions=char_text,
            )
        else:
            prompt = self._prompts.render(
                "generators",
                "FIRST_PASS_WRITER_TCC_TEMPLATE",
                scene_context=scene_text,
            )

        return await self._strong.complete(
            system_prompt="You are a master playwright writing theatrical prose.",
            user_prompt=prompt,
        )

    async def design_places(self, tccn: TCCN) -> list[Place]:
        """Design places / locations for the play."""
        prompt = self._prompts.render(
            "generators",
            "FIRST_PASS_SCENE_DESIGNER",
            tcc_context=tccn.to_prompt_text(),
            character_descriptions="",
        )
        raw = await self._strong.complete(
            system_prompt="You are a world-building expert designing locations.",
            user_prompt=prompt,
        )
        # Best-effort place extraction
        places: list[Place] = []
        for block in raw.split("\n\n"):
            lines = block.strip().split("\n")
            if lines:
                name = lines[0].strip().rstrip(":")
                desc = " ".join(l.strip() for l in lines[1:])
                if name and desc:
                    places.append(Place(name=name, description=desc))
        return places or [Place(name="Main Stage", description=raw[:500])]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scenes_to_text(self, tccn: TCCN, scenes: list[Scene]) -> str:
        """Render TCCN + scenes as plain text for prompt injection."""
        parts = [f"OVERALL TELEOLOGY: {tccn.teleology}\n"]
        for s in scenes:
            parts.append(f"SCENE NUMBER ({s.number}):")
            parts.append(f"ACTORS: {', '.join(s.actors)}")
            parts.append(f"SETTING: {s.setting}")
            parts.append(f"NARRATIVE THREADS: {', '.join(s.narrative_threads)}")
            parts.append("LIST OF ACTIONS IN NARRATIVE ORDER:")
            for a in s.actions:
                parts.append(f"- {a.actor} does {a.action}")
            parts.append("")
        return "\n".join(parts)

    def _parse_scenes(self, raw: str) -> list[Scene]:
        """Parse the LLM's scene output into structured Scene models."""
        scenes: list[Scene] = []
        # Split by SCENE NUMBER pattern
        scene_blocks = re.split(r"SCENE\s+NUMBER\s*\(?(\d+)\)?:", raw, flags=re.IGNORECASE)

        # scene_blocks[0] is the preamble (teleology etc), then pairs of (number, content)
        for i in range(1, len(scene_blocks), 2):
            number = int(scene_blocks[i])
            content = scene_blocks[i + 1] if i + 1 < len(scene_blocks) else ""

            actors = self._extract_field(content, "ACTORS")
            setting = self._extract_field(content, "SETTING")
            threads = self._extract_field(content, "NARRATIVE THREADS")
            actions = self._extract_actions(content)

            scenes.append(
                Scene(
                    number=number,
                    actors=[a.strip() for a in actors.split(",") if a.strip()],
                    setting=setting,
                    narrative_threads=[t.strip() for t in threads.split(",") if t.strip()],
                    actions=actions,
                )
            )

        return scenes

    @staticmethod
    def _extract_field(text: str, field: str) -> str:
        match = re.search(rf"{field}\s*:\s*(.+?)(?:\n[A-Z]|\Z)", text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_actions(text: str) -> list[SceneAction]:
        actions: list[SceneAction] = []
        action_section = re.search(
            r"LIST OF ACTIONS.*?:\s*\n(.*?)(?:\n\s*\n|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not action_section:
            return actions

        for line in action_section.group(1).strip().split("\n"):
            line = line.strip().lstrip("-").strip()
            if not line:
                continue
            # Try "actor does action" pattern
            match = re.match(r"(.+?)\s+(?:does|talks|says|goes|walks|enters|exits|looks)\s+(.+)", line, re.IGNORECASE)
            if match:
                actions.append(SceneAction(actor=match.group(1).strip(), action=line))
            else:
                actions.append(SceneAction(actor="", action=line))

        return actions
