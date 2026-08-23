"""Tests for typed environment configuration without loading the real `.env`."""

import pytest

from multi_agent_system import settings

ENVIRONMENT_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_MODEL",
    "OPENAI_MODEL",
    "OPENROUTER_TIMEOUT_SECONDS",
    "OPENROUTER_MAX_RETRIES",
    "LLM_MAX_TOTAL_TOKENS",
    "LLM_INPUT_COST_PER_MILLION",
    "LLM_OUTPUT_COST_PER_MILLION",
    "LLM_MAX_COST_USD",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Prevent developer machine configuration from affecting these tests."""
    monkeypatch.setattr(settings, "load_dotenv", lambda: None)
    for name in ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_can_load_without_key_for_cli_help() -> None:
    result = settings.WorkflowSettings.from_environment(require_api_key=False)

    assert result.openrouter_api_key is None
    assert result.openrouter_model == "openai/gpt-5.6-luna"
    assert result.openrouter_timeout_seconds == 60.0
    assert result.openrouter_max_retries == 2
    assert result.max_llm_tokens == 0


def test_values_are_parsed_normalized_and_secret_is_hidden(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")
    monkeypatch.setenv("OPENROUTER_MODEL", "gpt-example")
    monkeypatch.setenv("OPENROUTER_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "3")
    monkeypatch.setenv("LLM_MAX_TOTAL_TOKENS", "12000")
    monkeypatch.setenv("LLM_INPUT_COST_PER_MILLION", "2.5")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_MILLION", "10")
    monkeypatch.setenv("LLM_MAX_COST_USD", "1.25")

    result = settings.WorkflowSettings.from_environment()

    assert result.openrouter_model == "openai/gpt-example"
    assert result.openrouter_timeout_seconds == 45.5
    assert result.openrouter_max_retries == 3
    assert result.max_llm_tokens == 12000
    assert result.max_llm_cost_usd == 1.25
    assert "secret-value" not in repr(result)


def test_missing_required_api_key_is_rejected() -> None:
    with pytest.raises(settings.SettingsError, match="OPENROUTER_API_KEY"):
        settings.WorkflowSettings.from_environment()


def test_invalid_numeric_value_names_the_setting(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_MAX_RETRIES", "many")

    with pytest.raises(settings.SettingsError, match="OPENROUTER_MAX_RETRIES"):
        settings.WorkflowSettings.from_environment(require_api_key=False)


def test_cost_limit_requires_a_configured_price(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MAX_COST_USD", "1.0")

    with pytest.raises(settings.SettingsError, match="non-zero token price"):
        settings.WorkflowSettings.from_environment(require_api_key=False)
