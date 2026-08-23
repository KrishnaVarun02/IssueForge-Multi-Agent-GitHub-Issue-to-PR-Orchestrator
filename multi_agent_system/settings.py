"""Typed, validated configuration for the multi-agent workflow."""

from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.6-luna"
DEFAULT_OPENROUTER_TIMEOUT_SECONDS = 60.0
DEFAULT_OPENROUTER_MAX_RETRIES = 2


class SettingsError(ValueError):
    """Raised when an environment setting is missing or invalid."""


def _non_negative_int(name: str, default: int) -> int:
    """Read an integer environment value and reject negative numbers."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise SettingsError(f"{name} must be an integer.") from error
    if value < 0:
        raise SettingsError(f"{name} must be zero or greater.")
    return value


def _non_negative_float(name: str, default: float) -> float:
    """Read a floating-point environment value and reject negatives."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise SettingsError(f"{name} must be a number.") from error
    if value < 0:
        raise SettingsError(f"{name} must be zero or greater.")
    return value


@dataclass(frozen=True)
class WorkflowSettings:
    """One immutable snapshot of configuration loaded from the environment."""

    openrouter_api_key: str | None = field(default=None, repr=False)
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openrouter_timeout_seconds: float = DEFAULT_OPENROUTER_TIMEOUT_SECONDS
    openrouter_max_retries: int = DEFAULT_OPENROUTER_MAX_RETRIES
    max_llm_tokens: int = 0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    max_llm_cost_usd: float = 0.0

    @classmethod
    def from_environment(
        cls, *, require_api_key: bool = True
    ) -> "WorkflowSettings":
        """Load `.env`, parse values, and return a validated snapshot."""
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if require_api_key and not api_key:
            raise SettingsError("OPENROUTER_API_KEY is missing from .env.")

        model = (
            os.getenv("OPENROUTER_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_OPENROUTER_MODEL
        )
        if "/" not in model:
            model = f"openai/{model}"

        timeout = _non_negative_float(
            "OPENROUTER_TIMEOUT_SECONDS",
            DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
        )
        if timeout == 0:
            raise SettingsError("OPENROUTER_TIMEOUT_SECONDS must be greater than zero.")

        input_rate = _non_negative_float("LLM_INPUT_COST_PER_MILLION", 0.0)
        output_rate = _non_negative_float("LLM_OUTPUT_COST_PER_MILLION", 0.0)
        cost_limit = _non_negative_float("LLM_MAX_COST_USD", 0.0)
        if cost_limit > 0 and input_rate == 0 and output_rate == 0:
            raise SettingsError(
                "LLM_MAX_COST_USD requires at least one non-zero token price."
            )

        return cls(
            openrouter_api_key=api_key,
            openrouter_model=model,
            openrouter_timeout_seconds=timeout,
            openrouter_max_retries=_non_negative_int(
                "OPENROUTER_MAX_RETRIES", DEFAULT_OPENROUTER_MAX_RETRIES
            ),
            max_llm_tokens=_non_negative_int("LLM_MAX_TOTAL_TOKENS", 0),
            input_cost_per_million=input_rate,
            output_cost_per_million=output_rate,
            max_llm_cost_usd=cost_limit,
        )
