from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from playwriter.config import settings
from playwriter.models.trope import Trope, TropeSample


class TropeService:
    """Load, cache, sample, and search the TV-tropes dataset.

    The core purpose is to inject *randomness* into LLM prompts:
    sampling a handful of random tropes as "literary fate" forces the model
    off its regression-to-the-mean tendencies and towards more inventive
    scene construction.
    """

    def __init__(self, data_dir: str | Path | None = None):
        self._data_dir = Path(data_dir or settings.tropes_data_dir)
        self._tropes_df: pd.DataFrame | None = None
        self._media_dfs: dict[str, pd.DataFrame] = {}

    def _ensure_tropes_loaded(self) -> pd.DataFrame:
        if self._tropes_df is None:
            path = self._data_dir / "tropes.csv"
            self._tropes_df = pd.read_csv(path, index_col=0)
            # Normalise column names to what we expect
            cols = self._tropes_df.columns.tolist()
            if len(cols) >= 3:
                self._tropes_df.columns = (["trope_id", "name", "description"] + cols[3:])[:len(cols)]
        return self._tropes_df

    def _ensure_media_loaded(self, media: str) -> pd.DataFrame:
        """Load a media-specific tropes file (tv, film, lit)."""
        if media not in self._media_dfs:
            filename_map = {
                "tv": "tv_tropes.csv",
                "film": "film_tropes.csv",
                "lit": "lit_tropes.csv",
            }
            if media not in filename_map:
                raise ValueError(f"Unknown media type '{media}'. Choose from: tv, film, lit")
            path = self._data_dir / filename_map[media]
            self._media_dfs[media] = pd.read_csv(path, index_col=0)
        return self._media_dfs[media]

    def sample_random(self, n: int = 5) -> TropeSample:
        """Sample *n* random tropes from the master list (~216K tropes)."""
        df = self._ensure_tropes_loaded()
        sample = df.sample(n=min(n, len(df)))
        tropes = [
            Trope(
                trope_id=str(row.get("trope_id", "")),
                name=str(row.get("name", "")),
                description=str(row.get("description", ""))[:500],
            )
            for _, row in sample.iterrows()
        ]
        return TropeSample(tropes=tropes, source="random")

    def sample_by_media(
        self, media: str, title: str | None = None, n: int = 5
    ) -> TropeSample:
        """Sample tropes from a specific media type, optionally for a title."""
        df = self._ensure_media_loaded(media)
        if title:
            # Filter by title (case-insensitive partial match)
            mask = df.iloc[:, 0].str.contains(title, case=False, na=False)
            filtered = df[mask]
            if filtered.empty:
                filtered = df
        else:
            filtered = df
        sample = filtered.sample(n=min(n, len(filtered)))
        tropes = []
        for _, row in sample.iterrows():
            tropes.append(
                Trope(
                    trope_id=str(row.get("trope_id", row.get("Trope", ""))),
                    name=str(row.iloc[1]) if len(row) > 1 else "",
                    description=str(row.iloc[2])[:500] if len(row) > 2 else "",
                )
            )
        return TropeSample(tropes=tropes, source="by_media")

    def search(self, query: str, n: int = 10) -> TropeSample:
        """Simple text search across trope names and descriptions."""
        df = self._ensure_tropes_loaded()
        mask = (
            df["name"].str.contains(query, case=False, na=False)
            | df["description"].str.contains(query, case=False, na=False)
        )
        matches = df[mask].head(n)
        tropes = [
            Trope(
                trope_id=str(row.get("trope_id", "")),
                name=str(row.get("name", "")),
                description=str(row.get("description", ""))[:500],
            )
            for _, row in matches.iterrows()
        ]
        return TropeSample(tropes=tropes, source="filtered")

    @staticmethod
    def format_for_prompt(sample: TropeSample) -> str:
        """Render a trope sample as the 'literary fate' text for prompt injection."""
        return sample.to_prompt_text()
