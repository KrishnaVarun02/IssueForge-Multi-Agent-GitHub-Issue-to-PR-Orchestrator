"""Create the shared OpenRouter client used by LLM-powered agents."""

from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from multi_agent_system.settings import (
    DEFAULT_OPENROUTER_MAX_RETRIES,
    DEFAULT_OPENROUTER_TIMEOUT_SECONDS,
    WorkflowSettings,
)

# Backward-compatible exported names; canonical defaults live in settings.py.
OPENROUTER_TIMEOUT_SECONDS = DEFAULT_OPENROUTER_TIMEOUT_SECONDS
OPENROUTER_MAX_RETRIES = DEFAULT_OPENROUTER_MAX_RETRIES


class OpenRouterRequestError(RuntimeError):
    """A safe operational error after SDK retries have been exhausted."""


class LLMTokenBudgetExceeded(OpenRouterRequestError):
    """Raised before an API call when the workflow budget is exhausted."""


class LLMCostBudgetExceeded(OpenRouterRequestError):
    """Raised before an API call when estimated cost reached its limit."""


def estimated_llm_cost_usd(state: dict[str, Any]) -> float:
    """Estimate cost from token totals and user-supplied per-million rates."""
    prompt_cost = (
        state.get("llm_prompt_tokens", 0)
        * state.get("llm_input_cost_per_million", 0.0)
        / 1_000_000
    )
    completion_cost = (
        state.get("llm_completion_tokens", 0)
        * state.get("llm_output_cost_per_million", 0.0)
        / 1_000_000
    )
    return round(prompt_cost + completion_cost, 8)


def enforce_llm_budgets(state: dict[str, Any], node_name: str) -> None:
    """Stop a new LLM request when a token or estimated-cost limit is met."""
    limit = int(state.get("llm_max_total_tokens", 0) or 0)
    used = int(state.get("llm_total_tokens", 0) or 0)
    if limit > 0 and used >= limit:
        raise LLMTokenBudgetExceeded(
            f"LLM token budget reached before {node_name}: "
            f"{used} of {limit} tokens already used."
        )

    cost_limit = float(state.get("llm_max_cost_usd", 0.0) or 0.0)
    estimated_cost = estimated_llm_cost_usd(state)
    if cost_limit > 0 and estimated_cost >= cost_limit:
        raise LLMCostBudgetExceeded(
            f"Estimated LLM cost budget reached before {node_name}: "
            f"${estimated_cost:.6f} of ${cost_limit:.6f} already used."
        )


def describe_openrouter_error(error: APIError) -> str:
    """Describe the failure without including prompts, tokens, or response bodies."""
    if isinstance(error, RateLimitError):
        category = "rate limit was exceeded"
    elif isinstance(error, APITimeoutError):
        category = "request timed out"
    elif isinstance(error, APIConnectionError):
        category = "connection failed"
    elif isinstance(error, APIStatusError):
        category = f"API returned HTTP {error.status_code}"
    else:
        category = "request failed"

    request_id = getattr(error, "request_id", None)
    request_note = f" Request ID: {request_id}." if request_id else ""
    return f"OpenRouter {category} after automatic retries.{request_note}"


def create_chat_completion(
    client: OpenAI,
    *,
    workflow_state: dict[str, Any] | None = None,
    node_name: str = "LLM node",
    **arguments: Any,
) -> Any:
    """Call Chat Completions and convert SDK errors into a safe exception."""
    if workflow_state is not None:
        enforce_llm_budgets(workflow_state, node_name)
    try:
        return client.chat.completions.create(**arguments)
    except APIError as error:
        raise OpenRouterRequestError(describe_openrouter_error(error)) from error


def completion_usage_updates(
    state: dict[str, Any],
    completion: Any,
    node_name: str = "",
    requested_model: str = "",
) -> dict[str, Any]:
    """Add one response's token usage to the workflow's running totals."""
    usage = getattr(completion, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(
        getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
        or prompt_tokens + completion_tokens
    )
    updates: dict[str, Any] = {
        "llm_prompt_tokens": state.get("llm_prompt_tokens", 0) + prompt_tokens,
        "llm_completion_tokens": (
            state.get("llm_completion_tokens", 0) + completion_tokens
        ),
        "llm_total_tokens": state.get("llm_total_tokens", 0) + total_tokens,
        "llm_response_count": state.get("llm_response_count", 0) + 1,
    }
    if node_name:
        usage_by_node = {
            name: dict(values)
            for name, values in state.get("llm_usage_by_node", {}).items()
        }
        node_usage = usage_by_node.get(
            node_name,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "responses": 0,
            },
        )
        usage_by_node[node_name] = {
            "prompt_tokens": node_usage["prompt_tokens"] + prompt_tokens,
            "completion_tokens": (
                node_usage["completion_tokens"] + completion_tokens
            ),
            "total_tokens": node_usage["total_tokens"] + total_tokens,
            "responses": node_usage["responses"] + 1,
        }
        updates["llm_usage_by_node"] = usage_by_node
        response_trace = [dict(item) for item in state.get("llm_response_trace", [])]
        response_trace.append(
            {
                "sequence": len(response_trace) + 1,
                "node": node_name,
                "model": (
                    getattr(completion, "model", None)
                    or requested_model
                    or "unknown"
                ),
                "response_id": getattr(completion, "id", None) or "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "outcome": "received",
                "validation_error": "",
            }
        )
        updates["llm_response_trace"] = response_trace
    return updates


def mark_latest_response_outcome(
    state: dict[str, Any], outcome: str, validation_error: str = ""
) -> dict[str, list[dict[str, Any]]]:
    """Mark the newest response accepted or rejected without storing content."""
    trace = [dict(item) for item in state.get("llm_response_trace", [])]
    if not trace:
        return {"llm_response_trace": trace}
    trace[-1]["outcome"] = outcome
    trace[-1]["validation_error"] = validation_error[:200]
    return {"llm_response_trace": trace}


def current_usage_totals(state: dict[str, Any]) -> dict[str, Any]:
    """Copy usage fields for failure returns that have no new completion."""
    totals: dict[str, Any] = {
        "llm_prompt_tokens": state.get("llm_prompt_tokens", 0),
        "llm_completion_tokens": state.get("llm_completion_tokens", 0),
        "llm_total_tokens": state.get("llm_total_tokens", 0),
        "llm_response_count": state.get("llm_response_count", 0),
    }
    if "llm_usage_by_node" in state:
        totals["llm_usage_by_node"] = state["llm_usage_by_node"]
    if "llm_response_trace" in state:
        totals["llm_response_trace"] = state["llm_response_trace"]
    return totals


def get_openrouter_client_and_model() -> tuple[OpenAI, str]:
    """Load configuration and return a client plus model name."""
    settings = WorkflowSettings.from_environment()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        max_retries=settings.openrouter_max_retries,
        timeout=settings.openrouter_timeout_seconds,
    )
    return client, settings.openrouter_model
