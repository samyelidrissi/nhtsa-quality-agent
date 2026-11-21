from pathlib import Path

import pytest

from nhtsa_quality_agent.config import MissingSettingError, load_settings


def test_load_settings_reads_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    settings = load_settings()

    assert settings.llm_api_key == "test-key"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_load_settings_fails_without_api_key(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LLM_API_KEY", value)

    with pytest.raises(MissingSettingError, match="LLM_API_KEY"):
        load_settings()


def test_data_dir_defaults_to_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("DATA_DIR", raising=False)

    assert load_settings().data_dir == Path("data")


def test_data_dir_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", "/tmp/nhtsa")

    assert load_settings().data_dir == Path("/tmp/nhtsa")


def test_settings_are_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    settings = load_settings()

    with pytest.raises(AttributeError):
        settings.llm_api_key = "other"  # type: ignore[misc]
