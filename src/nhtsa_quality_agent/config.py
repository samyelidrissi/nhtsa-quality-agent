"""Application settings, read from environment variables (never hard-coded)."""

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path("data")


class MissingSettingError(ValueError):
    """Raised when a required environment variable is missing or empty."""


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    llm_api_key: str
    data_dir: Path = DEFAULT_DATA_DIR


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSettingError(f"Environment variable {name} is required but not set.")
    return value


def load_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings(
        llm_api_key=_require_env("LLM_API_KEY"),
        data_dir=Path(os.environ.get("DATA_DIR") or DEFAULT_DATA_DIR),
    )
