"""Tests for the LangGraph implementation of the workflow."""

from multi_agent_system.langgraph_workflow import run_langgraph_workflow


def test_langgraph_simple_route_skips_researcher() -> None:
    result = run_langgraph_workflow("Fix the login button")

    assert result["complexity"] == "simple"
    assert "research" not in result
    assert "researcher" not in result["execution_log"]


def test_langgraph_complex_route_runs_researcher() -> None:
    issue = "Users cannot complete checkout after changing currency when a discount is active"
    result = run_langgraph_workflow(issue)

    assert result["complexity"] == "complex"
    assert "research" in result
    assert result["execution_log"] == [
        "code_reader",
        "classifier",
        "researcher",
        "planner",
        "code_writer",
        "test_writer",
        "pr_opener",
    ]
