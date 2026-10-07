"""Tests that prove the deterministic workflow follows the correct paths."""

from multi_agent_system.workflow import run_workflow


def test_simple_issue_skips_researcher() -> None:
    result = run_workflow("Fix the login button")

    assert result["complexity"] == "simple"
    assert "research" not in result
    assert "researcher" not in result["execution_log"]


def test_complex_issue_runs_researcher() -> None:
    issue = "Users cannot complete checkout after changing currency when a discount is active"
    result = run_workflow(issue)

    assert result["complexity"] == "complex"
    assert "research" in result
    assert "researcher" in result["execution_log"]

def test_empty_issue_is_classified_as_simple() -> None:
    result = run_workflow("")

    assert result["complexity"] == "simple"
    assert "researcher" not in result["execution_log"]


def test_complete_workflow_creates_all_simulated_outputs() -> None:
    result = run_workflow("Fix the login button")

    assert "code_context" in result
    assert "plan" in result
    assert "patch" in result
    assert "tests" in result
    assert result["pr_url"].endswith("(simulated)")
    assert result["execution_log"] == [
        "code_reader",
        "classifier",
        "planner",
        "code_writer",
        "test_writer",
        "pr_opener",
    ]
