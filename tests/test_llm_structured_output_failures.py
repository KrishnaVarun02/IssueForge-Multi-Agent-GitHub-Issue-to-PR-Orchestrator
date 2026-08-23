"""Ensure simple LLM nodes reject malformed JSON without crashing the graph."""

import json
from types import SimpleNamespace

from multi_agent_system import llm_code_reader as reader_module
from multi_agent_system import llm_planner as planner_module


def fake_client_with(content: str) -> SimpleNamespace:
    """Return a client whose only completion contains the supplied text."""
    completion = SimpleNamespace(
        id="response-bad",
        model="test-model",
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )
    completions = SimpleNamespace(create=lambda **arguments: completion)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class SequentialCompletions:
    """Return each supplied content value once, in order."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0

    def create(self, **arguments):
        content = self.contents[self.calls]
        self.calls += 1
        return SimpleNamespace(
            id=f"response-{self.calls}",
            model="test-model",
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )


def sequential_client(contents: list[str]) -> tuple[SimpleNamespace, SequentialCompletions]:
    """Build a fake client plus its observable completions resource."""
    completions = SequentialCompletions(contents)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_code_reader_rejects_invalid_json_as_failed_state(monkeypatch) -> None:
    client = fake_client_with("not json")
    monkeypatch.setattr(
        reader_module,
        "get_openrouter_client_and_model",
        lambda: (client, "test-model"),
    )

    result = reader_module.llm_code_reader(
        {
            "issue": "Fix checkout",
            "repository_files": ["app.py"],
        }
    )

    assert result["llm_status"] == "failed"
    assert result["failed_node"] == "llm_code_reader"
    assert result["llm_response_trace"][0]["outcome"] == "rejected"
    assert result["llm_response_count"] == 2


def test_planner_rejects_invalid_json_as_failed_state(monkeypatch) -> None:
    client = fake_client_with("not json")
    monkeypatch.setattr(
        planner_module,
        "get_openrouter_client_and_model",
        lambda: (client, "test-model"),
    )

    result = planner_module.llm_planner(
        {
            "issue": "Fix checkout",
            "code_context": "checkout.py",
        }
    )

    assert result["llm_status"] == "failed"
    assert result["failed_node"] == "llm_planner"
    assert result["llm_response_trace"][0]["outcome"] == "rejected"
    assert result["llm_response_count"] == 2


def test_code_reader_accepts_second_structured_attempt(monkeypatch) -> None:
    valid = json.dumps(
        {"files": ["app.py"], "reasoning": "Contains checkout code."}
    )
    client, completions = sequential_client(["not json", valid])
    monkeypatch.setattr(
        reader_module,
        "get_openrouter_client_and_model",
        lambda: (client, "test-model"),
    )
    monkeypatch.setattr(
        reader_module,
        "read_selected_files",
        lambda state, paths: {
            "selected_files": paths,
            "selected_file_contents": {"app.py": "pass\n"},
            "code_context": "pass",
        },
    )

    result = reader_module.llm_code_reader(
        {"issue": "Fix checkout", "repository_files": ["app.py"]}
    )

    assert completions.calls == 2
    assert result["llm_status"] == "ok"
    assert [item["outcome"] for item in result["llm_response_trace"]] == [
        "rejected",
        "accepted",
    ]


def test_planner_accepts_second_structured_attempt(monkeypatch) -> None:
    valid = json.dumps(
        {
            "summary": "Fix checkout safely.",
            "steps": ["Change code", "Add tests"],
            "risks": ["Regression"],
        }
    )
    client, completions = sequential_client(["not json", valid])
    monkeypatch.setattr(
        planner_module,
        "get_openrouter_client_and_model",
        lambda: (client, "test-model"),
    )

    result = planner_module.llm_planner(
        {"issue": "Fix checkout", "code_context": "checkout.py"}
    )

    assert completions.calls == 2
    assert result["llm_status"] == "ok"
    assert result["plan_steps"] == ["Change code", "Add tests"]
    assert [item["outcome"] for item in result["llm_response_trace"]] == [
        "rejected",
        "accepted",
    ]
