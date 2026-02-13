from __future__ import annotations

from pathlib import Path

from playwriter.config import settings


class PromptLoader:
    """Loads prompt templates from .txt files and renders them with variables.

    Template format uses ``{variable_name}`` placeholders — the same format
    already used by every existing prompt file.
    """

    def __init__(self, templates_dir: str | Path | None = None):
        self._dir = Path(templates_dir or settings.prompts_dir)
        self._cache: dict[str, str] = {}

    def load(self, category: str, name: str) -> str:
        """Load raw template text.

        Example::

            loader.load("generators", "INITIAL_HISTORY_TCC_GENERATOR")
        """
        key = f"{category}/{name}"
        if key not in self._cache:
            path = self._dir / category / f"{name}.txt"
            self._cache[key] = path.read_text(encoding="utf-8")
        return self._cache[key]

    def render(self, category: str, name: str, **variables: str) -> str:
        """Load a template and substitute ``{var}`` placeholders.

        Only placeholders whose keys appear in *variables* are replaced;
        others are left untouched (safe partial rendering).
        """
        template = self.load(category, name)
        for k, v in variables.items():
            template = template.replace(f"{{{k}}}", v)
        return template
