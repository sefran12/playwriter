from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider API keys ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""

    # --- Default provider & model tier ---
    default_provider: str = "openai"  # openai | anthropic | groq

    # Temperature defaults (mirroring the notebook's 0.2 / 0.3 split)
    default_strong_temperature: float = 0.2
    default_fast_temperature: float = 0.3

    # --- Data paths ---
    tropes_data_dir: str = str(_PROJECT_ROOT / "data" / "TVTropesData")
    prompts_dir: str = str(
        _PROJECT_ROOT / "src" / "playwriter" / "prompts" / "templates"
    )

    # --- Database ---
    database_url: str = f"sqlite+aiosqlite:///{_PROJECT_ROOT / 'playwriter.db'}"


settings = Settings()
