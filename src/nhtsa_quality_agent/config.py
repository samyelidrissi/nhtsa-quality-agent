"""Application settings, read from environment variables (never hard-coded)."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_LLM_MODEL = "claude-haiku-5-5"


class MissingSettingError(ValueError):
    """Raised when a required environment variable is missing or empty."""


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    anthropic_api_key: str
    llm_model: str = DEFAULT_LLM_MODEL
    data_dir: Path = DEFAULT_DATA_DIR


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSettingError(f"Environment variable {name} is required but not set.")
    return value


def load_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings(
        anthropic_api_key=_require_env("ANTHROPIC_API_KEY"),
        llm_model=os.environ.get("LLM_MODEL", "").strip() or DEFAULT_LLM_MODEL,
        data_dir=Path(os.environ.get("DATA_DIR") or DEFAULT_DATA_DIR),
    )
