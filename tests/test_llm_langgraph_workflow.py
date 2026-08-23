"""Test the LLM workflow without spending money or calling an API."""

from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.llm_langgraph_workflow import build_llm_graph
from multi_agent_system.llm_langgraph_workflow import route_after_test_execution
from multi_agent_system.llm_langgraph_workflow import timed_node


def fake_planner(state: AgentState) -> AgentState:
    """Return the same shape as the real LLM Planner."""
    return {
        "plan": f"Fake plan for: {state['issue']}",
        "plan_steps": ["Change the code", "Add tests"],
        "plan_risks": ["Regression risk"],
        "execution_log": ["llm_planner"],
    }


def test_timed_node_accumulates_repeated_calls() -> None:
    ticks = iter([1.0, 1.025, 2.0, 2.040])
    measured = timed_node(
        "planner",
        lambda state: {"plan": "done"},
        clock=lambda: next(ticks),
    )

    state = measured({})
    state.update(measured(state))

    assert state["node_timings_ms"]["planner"] == {
        "calls": 2,
        "total_ms": 65.0,
        "last_ms": 40.0,
        "max_ms": 40.0,
    }


def fake_code_reader(state: AgentState) -> AgentState:
    """Select and load one known file without calling an API."""
    from multi_agent_system.repository_reader import read_selected_files

    updates = read_selected_files(state, ["app.py"])
    return {
        **updates,
        "file_selection_reasoning": "app.py contains the example code",
        "execution_log": ["llm_code_reader"],
    }


def fake_code_writer(state: AgentState) -> AgentState:
    """Return a proposed patch without calling an API or changing a file."""
    return {
        "patch_summary": "Update the greeting",
        "changed_files": ["app.py"],
        "patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-print('hello')\n+print('hi')",
        "code_generation_status": "generated",
        "code_generation_error": "",
        "execution_log": ["llm_code_writer"],
    }


def fake_test_writer(state: AgentState) -> AgentState:
    """Return a test proposal without calling an API or writing a file."""
    test_patch = (
        "--- /dev/null\n+++ b/tests/test_app.py\n"
        "@@ -0,0 +1 @@\n+assert True"
    )
    return {
        "test_summary": "Test the greeting",
        "test_files": ["tests/test_app.py"],
        "test_patch": test_patch,
        "test_command": "python3 -m pytest",
        "tests": test_patch,
        "test_generation_status": "generated",
        "test_generation_error": "",
        "execution_log": ["llm_test_writer"],
    }


def fake_test_runner(state: AgentState) -> AgentState:
    """Pretend the sandbox tests passed."""
    return {
        "tests_passed": True,
        "test_status": "passed",
        "test_output": "1 passed",
        "execution_log": ["sandbox_test_runner"],
    }


def fake_human_approval(state: AgentState) -> AgentState:
    """Approve without interrupting this graph-integration test."""
    return {
        "pull_request_approved": True,
        "approval_status": "approved",
        "approval_feedback": "",
        "execution_log": ["human_approval"],
    }


def fake_branch_preparer(state: AgentState) -> AgentState:
    """Pretend an approved local branch and commit were created."""
    return {
        "branch_prepared": True,
        "branch_name": "agent/issue-manual-fix-login-button",
        "branch_status": "prepared",
        "commit_sha": "abc123",
        "execution_log": ["git_branch_preparer"],
    }


def fake_pr_opener(state: AgentState) -> AgentState:
    """Return a fake URL so this test can never push to GitHub."""
    return {
        "pr_status": "created",
        "pr_url": "https://github.com/example/project/pull/123",
        "execution_log": ["github_pr_opener"],
    }


def test_llm_graph_accepts_fake_llm_nodes(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    graph = build_llm_graph(
        planner_node=fake_planner,
        code_reader_node=fake_code_reader,
        code_writer_node=fake_code_writer,
        test_writer_node=fake_test_writer,
        test_runner_node=fake_test_runner,
        approval_node=fake_human_approval,
        branch_preparer_node=fake_branch_preparer,
        pr_opener_node=fake_pr_opener,
    )
    result = graph.invoke(
        {
            "issue": "Fix the login button",
            "repo_path": str(tmp_path),
            "execution_log": [],
        }
    )

    assert result["repository_files"] == ["app.py"]
    assert result["selected_files"] == ["app.py"]
    assert "print('hello')" in result["code_context"]
    assert result["plan_steps"] == ["Change the code", "Add tests"]
    assert result["plan_risks"] == ["Regression risk"]
    assert result["changed_files"] == ["app.py"]
    assert result["patch"].startswith("--- a/app.py")
    assert result["test_files"] == ["tests/test_app.py"]
    assert result["test_command"] == "python3 -m pytest"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "print('hello')"
    assert not (tmp_path / "tests" / "test_app.py").exists()
    assert "llm_planner" in result["execution_log"]
    assert "llm_code_writer" in result["execution_log"]
    assert "llm_test_writer" in result["execution_log"]
    assert "sandbox_test_runner" in result["execution_log"]
    assert result["approval_status"] == "approved"
    assert result["branch_status"] == "prepared"
    assert result["tests_passed"] is True
    assert result["execution_log"][:2] == [
        "manual_issue_input",
        "repository_indexer",
    ]
    assert result["execution_log"][2] == "llm_code_reader"
    assert result["pr_url"] == "https://github.com/example/project/pull/123"
    assert result["pr_status"] == "created"


def test_rejection_feedback_loops_back_to_code_writer(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    graph = build_llm_graph(
        planner_node=fake_planner,
        code_reader_node=fake_code_reader,
        code_writer_node=fake_code_writer,
        test_writer_node=fake_test_writer,
        test_runner_node=fake_test_runner,
        branch_preparer_node=fake_branch_preparer,
        pr_opener_node=fake_pr_opener,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "revision-loop-test"}}

    first_review = graph.invoke(
        {
            "issue": "Fix the login button",
            "repo_path": str(tmp_path),
            "workflow_thread_id": "revision-loop-test",
            "execution_log": [],
        },
        config=config,
    )
    assert first_review["__interrupt__"][0].value["revision_count"] == 0

    second_review = graph.invoke(
        Command(
            resume={
                "decision": "reject",
                "feedback": "Make the patch smaller.",
            }
        ),
        config=config,
    )
    assert second_review["__interrupt__"][0].value["revision_count"] == 1
    assert second_review["execution_log"].count("llm_code_writer") == 2
    assert second_review["execution_log"].count("sandbox_test_runner") == 2

    final = graph.invoke(
        Command(resume={"decision": "approve", "feedback": ""}),
        config=config,
    )
    assert final["approval_status"] == "approved"
    assert final["pr_status"] == "created"


def test_failed_sandbox_tests_are_regenerated_once(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")
    test_runner_calls = 0
    feedback_seen: list[str] = []

    def recording_test_writer(state: AgentState) -> AgentState:
        feedback_seen.append(state.get("test_output", ""))
        return fake_test_writer(state)

    def flaky_test_runner(state: AgentState) -> AgentState:
        nonlocal test_runner_calls
        test_runner_calls += 1
        if test_runner_calls == 1:
            return {
                "tests_passed": False,
                "test_status": "failed",
                "test_output": "AssertionError: expected hi",
                "execution_log": ["sandbox_test_runner"],
            }
        return fake_test_runner(state)

    graph = build_llm_graph(
        planner_node=fake_planner,
        code_reader_node=fake_code_reader,
        code_writer_node=fake_code_writer,
        test_writer_node=recording_test_writer,
        test_runner_node=flaky_test_runner,
        approval_node=fake_human_approval,
        branch_preparer_node=fake_branch_preparer,
        pr_opener_node=fake_pr_opener,
    )

    result = graph.invoke(
        {
            "issue": "Fix the login button",
            "repo_path": str(tmp_path),
            "execution_log": [],
        }
    )

    assert test_runner_calls == 2
    assert feedback_seen == ["", "AssertionError: expected hi"]
    assert result["test_execution_attempts"] == 2
    assert result["tests_passed"] is True
    assert result["execution_log"].count("llm_test_writer") == 2
    assert result["execution_log"].count("sandbox_test_runner") == 2


def test_infrastructure_failure_does_not_retry_test_generation() -> None:
    route = route_after_test_execution(
        {
            "tests_passed": False,
            "test_status": "docker_unavailable",
            "test_execution_attempts": 1,
        }
    )

    assert route == "end"


def test_llm_api_failure_stops_before_downstream_agents(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")

    def failed_code_reader(state: AgentState) -> AgentState:
        return {
            "llm_status": "failed",
            "llm_error": "OpenRouter request timed out after automatic retries.",
            "failed_node": "llm_code_reader",
            "execution_log": ["llm_code_reader"],
        }

    graph = build_llm_graph(code_reader_node=failed_code_reader)
    result = graph.invoke(
        {
            "issue": "Fix the login button",
            "repo_path": str(tmp_path),
            "execution_log": [],
        }
    )

    assert result["llm_status"] == "failed"
    assert result["execution_log"] == [
        "manual_issue_input",
        "repository_indexer",
        "llm_code_reader",
    ]
    assert "classifier" not in result["execution_log"]
