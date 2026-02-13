"""Stochastic action resolution with trope-based fate modifiers.

The dice system is the anti-collapse core of the Narrative Engine.
Actions are resolved by random d100 rolls — the LLM never decides
whether something succeeds, only narrates the outcome it's been told.

Tropes are sampled and used as "fate modifiers" that shift the roll,
creating thematic connections between the stochastic events and the
literary texture of the narrative.
"""

from __future__ import annotations

import json
import logging
import random
from typing import List, Optional

from playwriter.llm.base import LLMProvider
from playwriter.models.narrative import (
    DiceOutcome,
    DiceRoll,
    FateModifier,
)
from playwriter.models.trope import Trope, TropeSample
from playwriter.prompts.loader import PromptLoader
from playwriter.services.trope import TropeService

log = logging.getLogger(__name__)


class DiceService:
    """Stochastic action resolution with trope-based fate modifiers.

    Resolution table (d100 after fate modifiers, clamped 1-100):
        1-5:    CATASTROPHIC_FAILURE  — action backfires spectacularly
        6-30:   FAILURE               — action simply does not succeed
        31-60:  MIXED                 — partial success with complication
        61-90:  SUCCESS               — action works as intended
        91-100: CRITICAL_SUCCESS      — beyond expectations
    """

    def __init__(
        self,
        fast_llm: LLMProvider,
        trope_service: TropeService,
        prompts: Optional[PromptLoader] = None,
    ):
        self._fast = fast_llm
        self._tropes = trope_service
        self._prompts = prompts or PromptLoader()

    # ── Pure randomness ──────────────────────────────────────────────────

    @staticmethod
    def roll_d100() -> int:
        """Pure random d100 roll. Never LLM-generated."""
        return random.randint(1, 100)

    @staticmethod
    def classify_outcome(value: int) -> DiceOutcome:
        """Map a 1-100 value to the five-tier outcome ladder."""
        if value <= 5:
            return DiceOutcome.CATASTROPHIC_FAILURE
        if value <= 30:
            return DiceOutcome.FAILURE
        if value <= 60:
            return DiceOutcome.MIXED
        if value <= 90:
            return DiceOutcome.SUCCESS
        return DiceOutcome.CRITICAL_SUCCESS

    # ── Fate modifiers (LLM-assessed, bounded) ──────────────────────────

    async def assess_fate_modifiers(
        self,
        action: str,
        actor: str,
        active_tropes: List[Trope],
        scene_context: str,
    ) -> List[FateModifier]:
        """Ask the fast LLM to determine how active tropes modify this roll.

        Each trope produces a signed modifier bounded to [-30, +30].
        A "Tragic Hubris" trope might give -20 to an overconfident action.
        An "Unlikely Hero" trope might give +15 to an underdog attempt.
        """
        if not active_tropes:
            return []

        tropes_text = "\n".join(
            f"- {t.name}: {t.description[:200]}" for t in active_tropes
        )
        prompt = self._prompts.render(
            "assessors",
            "FATE_MODIFIER_ASSESSOR",
            action=action,
            actor=actor,
            tropes_text=tropes_text,
            scene_context=scene_context[:500],
        )

        log.info(
            "Assessing fate modifiers: actor=%s, action=%s, tropes=%d",
            actor, action[:60], len(active_tropes),
        )

        try:
            raw = await self._fast.complete(
                system_prompt=(
                    "You assess how literary tropes modify the probability of "
                    "character actions. Return ONLY valid JSON."
                ),
                user_prompt=prompt,
                json_mode=True,
                max_tokens=1024,
            )
            data = json.loads(raw)
            modifiers = []
            items = data if isinstance(data, list) else data.get("modifiers", [])
            for item in items:
                trope_name = item.get("trope_name", "")
                # Find matching trope
                matching = [t for t in active_tropes if t.name == trope_name]
                trope = matching[0] if matching else active_tropes[0]
                modifier_val = max(-30, min(30, int(item.get("modifier", 0))))
                modifiers.append(FateModifier(
                    trope=trope,
                    modifier=modifier_val,
                    rationale=item.get("rationale", ""),
                ))
            log.info("Fate modifiers assessed: %s",
                     [(m.trope.name, m.modifier) for m in modifiers])
            return modifiers
        except Exception as exc:
            log.warning("Fate modifier assessment failed, using neutral: %s", exc)
            return [
                FateModifier(trope=t, modifier=0, rationale="(assessment failed)")
                for t in active_tropes
            ]

    # ── Full resolution ──────────────────────────────────────────────────

    async def resolve_action(
        self,
        action: str,
        actor: str,
        scene_context: str,
        trope_pool: Optional[List[Trope]] = None,
        n_tropes: int = 2,
        override_roll: Optional[int] = None,
    ) -> DiceRoll:
        """Full stochastic resolution: sample tropes, roll dice, apply modifiers.

        Parameters
        ----------
        action:
            What the character is attempting.
        actor:
            The character's name.
        scene_context:
            Summary of the current scene for the LLM to contextualize.
        trope_pool:
            Pool to sample tropes from. If None, samples globally.
        n_tropes:
            How many tropes to activate as fate (default 2).
        override_roll:
            For director mode — force a specific roll value.
        """
        # 1. Select tropes for this beat
        if trope_pool and len(trope_pool) >= n_tropes:
            active_tropes = random.sample(trope_pool, n_tropes)
        else:
            sample = self._tropes.sample_random(n=n_tropes)
            active_tropes = sample.tropes

        # 2. Assess fate modifiers via LLM
        modifiers = await self.assess_fate_modifiers(
            action, actor, active_tropes, scene_context,
        )

        # 3. Roll (or use director override)
        raw = override_roll if override_roll is not None else self.roll_d100()
        log.info("Dice roll: raw=%d, override=%s", raw, override_roll is not None)

        # 4. Apply modifiers, clamp to 1-100
        total_modifier = sum(m.modifier for m in modifiers)
        final = max(1, min(100, raw + total_modifier))

        # 5. Classify outcome
        outcome = self.classify_outcome(final)
        log.info(
            "Dice resolved: raw=%d, modifier=%+d, final=%d, outcome=%s",
            raw, total_modifier, final, outcome.value,
        )

        return DiceRoll(
            raw_roll=raw,
            fate_modifiers=modifiers,
            final_value=final,
            outcome=outcome,
            action_description=action,
            actor=actor,
        )
