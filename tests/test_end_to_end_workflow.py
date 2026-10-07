"""Exercise the complete graph with real agents and fake external boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from multi_agent_system import llm_code_reader as reader_module
from multi_agent_system import llm_code_writer as code_writer_module
from multi_agent_system import llm_planner as planner_module
from multi_agent_system import validated_test_writer as test_writer_module
from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.llm_langgraph_workflow import build_llm_graph


class FakeCompletions:
    """Return queued structured responses without making network requests."""

    def __init__(self, contents: list[str], prefix: str) -> None:
        self.contents = contents
        self.prefix = prefix
        self.calls = 0

    def create(self, **arguments):
        content = self.contents[self.calls]
        self.calls += 1
        return SimpleNamespace(
            id=f"{self.prefix}-{self.calls}",
            model="test/provider-model",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=25,
                total_tokens=125,
            ),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )


def fake_client(contents: list[str], prefix: str) -> tuple[SimpleNamespace, FakeCompletions]:
    """Build an SDK-shaped client and return its observable queue."""
    completions = FakeCompletions(contents, prefix)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def fake_issue_reader(state: AgentState) -> AgentState:
    """Return a GitHub issue without contacting GitHub."""
    return {
        "issue": "Change the command-line greeting and add a regression test",
        "issue_title": "Update greeting",
        "issue_body": "Use hi instead of hello.",
        "issue_number": 99,
        "issue_url": state["issue_url"],
        "repository_owner": "example",
        "repository_name": "project",
        "execution_log": ["github_issue_reader"],
    }


def fake_branch_preparer(state: AgentState) -> AgentState:
    """Represent a prepared commit without invoking Git or changing files."""
    return {
        "branch_prepared": True,
        "branch_name": "agent/issue-99-update-greeting",
        "branch_status": "prepared",
        "commit_sha": "abc123",
        "base_branch": "main",
        "execution_log": ["git_branch_preparer"],
    }


def fake_pr_opener(state: AgentState) -> AgentState:
    """Return a PR URL without pushing or contacting GitHub."""
    return {
        "pr_status": "created",
        "pr_url": "https://github.com/example/project/pull/100",
        "execution_log": ["github_pr_opener"],
    }


def test_complete_issue_to_pr_flow_with_retry_review_and_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    """Verify the entire workflow while every external side effect is fake."""
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")

    reader_json = json.dumps(
        {"files": ["app.py"], "reasoning": "Contains the greeting."}
    )
    planner_json = json.dumps(
        {
            "summary": "Update the greeting and add a test.",
            "steps": ["Edit app.py", "Add a regression test"],
            "risks": ["CLI output regression"],
        }
    )
    patch_json = json.dumps(
        {
            "summary": "Change hello to hi.",
            "edits": [
                {
                    "path": "app.py",
                    "start_line": 1,
                    "end_line": 1,
                    "replacement": "print('hi')",
                }
            ],
        }
    )
    test_json = json.dumps(
        {
            "summary": "Verify the greeting.",
            "files": [
                {
                    "path": "tests/test_generated_greeting.py",
                    "content": (
                        "def test_generated_greeting():\n"
                        "    assert 'hi' == 'hi'\n"
                    ),
                }
            ],
            "suggested_command": "python3 -m pytest",
        }
    )

    reader_client, reader_calls = fake_client([reader_json], "reader")
    planner_client, planner_calls = fake_client([planner_json], "planner")
    code_client, code_calls = fake_client(
        ["incomplete json", patch_json], "writer"
    )
    test_client, test_calls = fake_client(
        [test_json, test_json], "tests"
    )
    monkeypatch.setattr(
        reader_module,
        "get_openrouter_client_and_model",
        lambda: (reader_client, "requested/model"),
    )
    monkeypatch.setattr(
        planner_module,
        "get_openrouter_client_and_model",
        lambda: (planner_client, "requested/model"),
    )
    monkeypatch.setattr(
        code_writer_module,
        "get_openrouter_client_and_model",
        lambda: (code_client, "requested/model"),
    )
    monkeypatch.setattr(
        test_writer_module,
        "get_openrouter_client_and_model",
        lambda: (test_client, "requested/model"),
    )

    test_runs = 0

    def fake_test_runner(state: AgentState) -> AgentState:
        nonlocal test_runs
        test_runs += 1
        passed = test_runs == 2
        return {
            "tests_passed": passed,
            "test_status": "passed" if passed else "failed",
            "test_output": "1 passed" if passed else "AssertionError",
            "sandbox_kind": "fake-docker",
            "execution_log": ["sandbox_test_runner"],
        }

    graph = build_llm_graph(
        issue_reader_node=fake_issue_reader,
        test_runner_node=fake_test_runner,
        branch_preparer_node=fake_branch_preparer,
        pr_opener_node=fake_pr_opener,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "end-to-end-test"}}

    waiting = graph.invoke(
        {
            "issue_url": "https://github.com/example/project/issues/99",
            "repo_path": str(tmp_path),
            "execute_tests": True,
            "workflow_thread_id": "end-to-end-test",
            "execution_log": [],
        },
        config=config,
    )

    assert waiting["__interrupt__"]
    assert waiting["tests_passed"] is True
    assert waiting["test_execution_attempts"] == 2
    assert code_calls.calls == 2
    assert test_calls.calls == 2
    assert reader_calls.calls == 1
    assert planner_calls.calls == 1
    assert waiting["llm_response_count"] == 6
    assert [
        item["outcome"]
        for item in waiting["llm_response_trace"]
        if item["node"] == "llm_code_writer"
    ] == ["rejected", "accepted"]

    final = graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )

    assert final["approval_status"] == "approved"
    assert final["branch_status"] == "prepared"
    assert final["pr_url"] == "https://github.com/example/project/pull/100"
    assert final["node_timings_ms"]["test_writer"]["calls"] == 2
    assert final["node_timings_ms"]["test_runner"]["calls"] == 2
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert not (tmp_path / "tests").exists()
