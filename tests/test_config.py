from pathlib import Path

import pytest

from nhtsa_quality_agent.config import DEFAULT_LLM_MODEL, MissingSettingError, load_settings


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)


def test_load_settings_reads_api_key(api_key: None) -> None:
    assert load_settings().anthropic_api_key == "test-key"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_load_settings_fails_without_api_key(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)

    with pytest.raises(MissingSettingError, match="ANTHROPIC_API_KEY"):
        load_settings()


def test_defaults(api_key: None) -> None:
    settings = load_settings()

    assert settings.data_dir == Path("data")
    assert settings.llm_model == DEFAULT_LLM_MODEL


def test_overrides(api_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", "/tmp/nhtsa")
    monkeypatch.setenv("LLM_MODEL", "claude-opus-5-5")

    settings = load_settings()

    assert settings.data_dir == Path("/tmp/nhtsa")
    assert settings.llm_model == "claude-opus-5-5"


def test_settings_are_immutable(api_key: None) -> None:
    settings = load_settings()

    with pytest.raises(AttributeError):
        settings.anthropic_api_key = "other"  # type: ignore[misc]
