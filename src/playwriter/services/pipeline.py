from __future__ import annotations

from playwriter.llm.base import LLMProvider
from playwriter.models.character import Character
from playwriter.models.scene import Scene
from playwriter.models.story import TCCN
from playwriter.prompts.loader import PromptLoader
from playwriter.services.character import CharacterService
from playwriter.services.scene import SceneService
from playwriter.services.seeding import SeedingService
from playwriter.services.trope import TropeService


class PipelineService:
    """Full end-to-end pipeline: seed → TCCN → characters → scenes → prose.

    Orchestrates all services into the complete Playwriter pipeline.
    """

    def __init__(
        self,
        strong_llm: LLMProvider,
        fast_llm: LLMProvider | None = None,
        prompts: PromptLoader | None = None,
        trope_service: TropeService | None = None,
    ):
        self._prompts = prompts or PromptLoader()
        self._seeding = SeedingService(strong_llm, self._prompts)
        self._characters = CharacterService(strong_llm, fast_llm, self._prompts)
        self._scenes = SceneService(strong_llm, fast_llm, self._prompts)
        self._tropes = trope_service or TropeService()

    async def full_pipeline(
        self,
        seed_description: str,
        trope_count: int = 5,
        refine_rounds: int = 1,
    ) -> dict:
        """Run the complete pipeline and return all generated game data.

        Returns a dict with keys:
        - tccn: The generated TCCN seed
        - characters: list of full Character profiles
        - tropes_used: the TropeSample injected
        - scenes: list of Scene objects
        - evaluation: scene critique text
        - prose: theatrical prose of the first scene
        """
        # 1. Generate seed → TCCN
        tccn = await self._seeding.generate_seed(seed_description)

        # 2. Generate characters
        characters: list[Character] = []
        for char_summary in tccn.characters:
            character = await self._characters.generate(tccn, char_summary)
            characters.append(character)

        # 3. Refine characters
        refined: list[Character] = []
        for character in characters:
            r = await self._characters.refine(character, tccn, rounds=refine_rounds)
            refined.append(r)
        characters = refined

        # 4. Sample random tropes for anti-regression-to-mean injection
        trope_sample = self._tropes.sample_random(n=trope_count)

        # 5. Compose scenes with trope injection
        scenes = await self._scenes.compose_scenes(tccn, trope_sample)

        # 6. Evaluate scenes
        evaluation = await self._scenes.evaluate_scenes(tccn, scenes)

        # 7. Write theatrical prose
        prose = await self._scenes.write_scene(tccn, scenes, characters)

        return {
            "tccn": tccn.model_dump(),
            "characters": [c.model_dump() for c in characters],
            "tropes_used": trope_sample.model_dump(),
            "scenes": [s.model_dump() for s in scenes],
            "evaluation": evaluation,
            "prose": prose,
        }
