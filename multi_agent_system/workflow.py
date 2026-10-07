"""Deterministic workflow used by the application and its tests."""

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Shared data produced and consumed by workflow agents."""

    issue: str
    code_context: str
    complexity: str
    research: str
    plan: str
    patch: str
    tests: str
    pr_url: str
    execution_log: list[str]


def code_reader(state: AgentState) -> AgentState:
    state["code_context"] = f"Relevant module for: {state['issue']}"
    state["execution_log"].append("code_reader")
    return state


def classify_issue(state: AgentState) -> AgentState:
    word_count = len(state["issue"].split())
    state["complexity"] = "complex" if word_count > 8 else "simple"
    state["execution_log"].append("classifier")
    return state


def researcher(state: AgentState) -> AgentState:
    state["research"] = "Inspect dependencies and recent changes."
    state["execution_log"].append("researcher")
    return state


def planner(state: AgentState) -> AgentState:
    if "research" in state:
        state["plan"] = "Update related modules, handle edge cases, and add tests."
    else:
        state["plan"] = "Update the relevant module and add one regression test."

    state["execution_log"].append("planner")
    return state


def code_writer(state: AgentState) -> AgentState:
    state["patch"] = f"Simulated code change based on plan: {state['plan']}"
    state["execution_log"].append("code_writer")
    return state


def test_writer(state: AgentState) -> AgentState:
    state["tests"] = f"Simulated test for patch: {state['patch']}"
    state["execution_log"].append("test_writer")
    return state


def pr_opener(state: AgentState) -> AgentState:
    state["pr_url"] = "https://github.com/example/project/pull/123 (simulated)"
    state["execution_log"].append("pr_opener")
    return state


def run_workflow(issue: str) -> AgentState:
    state: AgentState = {"issue": issue, "execution_log": []}
    state = code_reader(state)
    state = classify_issue(state)

    if state["complexity"] == "complex":
        state = researcher(state)

    state = planner(state)
    state = code_writer(state)
    state = test_writer(state)
    state = pr_opener(state)
    return state
