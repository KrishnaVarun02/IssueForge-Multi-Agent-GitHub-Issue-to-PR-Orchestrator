"""Tests for bounded and sanitized OpenRouter SDK failures."""

from types import SimpleNamespace

import httpx2
from openai import APITimeoutError

from multi_agent_system.openrouter_client import (
    LLMCostBudgetExceeded,
    LLMTokenBudgetExceeded,
    OpenRouterRequestError,
    completion_usage_updates,
    create_chat_completion,
    describe_openrouter_error,
    estimated_llm_cost_usd,
)


class FailingCompletions:
    """Act like the SDK resource after all automatic retries fail."""

    def create(self, **arguments):
        request = httpx2.Request(
            "POST", "https://openrouter.ai/api/v1/chat/completions"
        )
        raise APITimeoutError(request)


def test_timeout_description_contains_no_request_body_or_key() -> None:
    request = httpx2.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer secret-key"},
        content=b"private prompt",
    )

    message = describe_openrouter_error(APITimeoutError(request))

    assert message == "OpenRouter request timed out after automatic retries."
    assert "secret-key" not in message
    assert "private prompt" not in message


def test_completion_wrapper_raises_safe_domain_error() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )

    try:
        create_chat_completion(client, model="test-model", messages=[])
    except OpenRouterRequestError as error:
        assert "timed out" in str(error)
    else:
        raise AssertionError("SDK timeout was not converted")


def test_completion_usage_accumulates_across_responses() -> None:
    first = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
    )
    second = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
        )
    )

    state = completion_usage_updates({}, first)
    state.update(completion_usage_updates(state, second))

    assert state == {
        "llm_prompt_tokens": 150,
        "llm_completion_tokens": 30,
        "llm_total_tokens": 180,
        "llm_response_count": 2,
    }


def test_exhausted_token_budget_prevents_api_call() -> None:
    completions = SimpleNamespace(create=lambda **arguments: None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    called = False

    def record_call(**arguments):
        nonlocal called
        called = True

    completions.create = record_call

    try:
        create_chat_completion(
            client,
            workflow_state={
                "llm_total_tokens": 500,
                "llm_max_total_tokens": 500,
            },
            node_name="llm_planner",
            model="test-model",
            messages=[],
        )
    except LLMTokenBudgetExceeded as error:
        assert "500 of 500" in str(error)
        assert "llm_planner" in str(error)
    else:
        raise AssertionError("Exhausted token budget was not enforced")

    assert called is False


def test_zero_token_limit_keeps_budget_unlimited() -> None:
    completion = SimpleNamespace(choices=[])
    completions = SimpleNamespace(create=lambda **arguments: completion)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = create_chat_completion(
        client,
        workflow_state={
            "llm_total_tokens": 50_000,
            "llm_max_total_tokens": 0,
        },
        model="test-model",
        messages=[],
    )

    assert result is completion


def test_cost_estimate_uses_separate_input_and_output_rates() -> None:
    state = {
        "llm_prompt_tokens": 750_000,
        "llm_completion_tokens": 250_000,
        "llm_input_cost_per_million": 2.0,
        "llm_output_cost_per_million": 8.0,
    }

    assert estimated_llm_cost_usd(state) == 3.5


def test_estimated_cost_budget_prevents_api_call() -> None:
    called = False

    def record_call(**arguments):
        nonlocal called
        called = True

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=record_call)
        )
    )
    state = {
        "llm_prompt_tokens": 750_000,
        "llm_completion_tokens": 250_000,
        "llm_input_cost_per_million": 2.0,
        "llm_output_cost_per_million": 8.0,
        "llm_max_cost_usd": 3.5,
    }

    try:
        create_chat_completion(
            client,
            workflow_state=state,
            node_name="llm_code_writer",
            model="test-model",
            messages=[],
        )
    except LLMCostBudgetExceeded as error:
        assert "$3.500000 of $3.500000" in str(error)
        assert "llm_code_writer" in str(error)
    else:
        raise AssertionError("Estimated cost budget was not enforced")

    assert called is False


def test_completion_usage_is_grouped_by_agent_and_retry() -> None:
    planner = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
    )
    writer = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=50,
            total_tokens=250,
        )
    )
    state = completion_usage_updates({}, planner, "llm_planner")
    state.update(completion_usage_updates(state, writer, "llm_code_writer"))
    state.update(completion_usage_updates(state, writer, "llm_code_writer"))

    assert state["llm_total_tokens"] == 620
    assert state["llm_usage_by_node"]["llm_planner"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "responses": 1,
    }
    assert state["llm_usage_by_node"]["llm_code_writer"] == {
        "prompt_tokens": 400,
        "completion_tokens": 100,
        "total_tokens": 500,
        "responses": 2,
    }


def test_response_trace_records_order_model_and_identifier() -> None:
    first = SimpleNamespace(
        id="response-1",
        model="provider/model-a",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        ),
    )
    second = SimpleNamespace(
        id="response-2",
        model=None,
        usage=SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=50,
            total_tokens=250,
        ),
    )
    state = completion_usage_updates(
        {}, first, "llm_code_writer", "requested/model"
    )
    state.update(
        completion_usage_updates(
            state, second, "llm_code_writer", "requested/model"
        )
    )

    assert state["llm_response_trace"] == [
        {
            "sequence": 1,
            "node": "llm_code_writer",
            "model": "provider/model-a",
            "response_id": "response-1",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "outcome": "received",
            "validation_error": "",
        },
        {
            "sequence": 2,
            "node": "llm_code_writer",
            "model": "requested/model",
            "response_id": "response-2",
            "prompt_tokens": 200,
            "completion_tokens": 50,
            "total_tokens": 250,
            "outcome": "received",
            "validation_error": "",
        },
    ]
