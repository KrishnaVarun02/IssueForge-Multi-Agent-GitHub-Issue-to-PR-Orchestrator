"""LangGraph implementation of the deterministic multi-agent workflow."""

from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict, total=False):
    """Shared state that LangGraph carries between nodes."""

    issue: str
    issue_url: str
    issue_number: int
    issue_title: str
    issue_body: str
    issue_labels: list[str]
    repository_owner: str
    repository_name: str
    github_authentication: str
    repo_path: str
    repository_files: list[str]
    selected_files: list[str]
    selected_file_contents: dict[str, str]
    file_selection_reasoning: str
    code_context: str
    complexity: str
    research: str
    plan: str
    plan_steps: list[str]
    plan_risks: list[str]
    llm_status: str
    llm_error: str
    failed_node: str
    llm_prompt_tokens: int
    llm_completion_tokens: int
    llm_total_tokens: int
    llm_response_count: int
    llm_max_total_tokens: int
    llm_input_cost_per_million: float
    llm_output_cost_per_million: float
    llm_max_cost_usd: float
    llm_usage_by_node: dict[str, dict[str, int]]
    llm_response_trace: list[dict[str, str | int]]
    node_timings_ms: dict[str, dict[str, int | float]]
    patch_summary: str
    changed_files: list[str]
    patch: str
    code_generation_status: str
    code_generation_error: str
    test_summary: str
    test_files: list[str]
    test_patch: str
    test_command: str
    tests: str
    test_generation_status: str
    test_generation_error: str
    execute_tests: bool
    tests_passed: bool
    test_status: str
    test_output: str
    test_execution_attempts: int
    sandbox_kind: str
    workflow_thread_id: str
    pull_request_approved: bool
    approval_status: str
    approval_feedback: str
    revision_count: int
    branch_prepared: bool
    branch_name: str
    branch_status: str
    commit_sha: str
    base_branch: str
    pr_status: str
    pr_url: str
    execution_log: Annotated[list[str], add]


def code_reader(state: AgentState) -> AgentState:
    """Return the code context discovered for an issue."""
    return {
        "code_context": f"Relevant module for: {state['issue']}",
        "execution_log": ["code_reader"],
    }


def classifier(state: AgentState) -> AgentState:
    """Return a simple or complex classification."""
    word_count = len(state["issue"].split())
    complexity = "complex" if word_count > 8 else "simple"
    return {"complexity": complexity, "execution_log": ["classifier"]}


def route_by_complexity(
    state: AgentState,
) -> Literal["researcher", "planner"]:
    """Tell LangGraph which node should run after the classifier."""
    if state["complexity"] == "complex":
        return "researcher"

    return "planner"


def researcher(state: AgentState) -> AgentState:
    """Return additional research for a complex issue."""
    return {
        "research": "Inspect dependencies and recent changes.",
        "execution_log": ["researcher"],
    }


def planner(state: AgentState) -> AgentState:
    """Return an implementation plan based on available state."""
    if "research" in state:
        plan = "Update related modules, handle edge cases, and add tests."
    else:
        plan = "Update the relevant module and add one regression test."

    return {"plan": plan, "execution_log": ["planner"]}


def code_writer(state: AgentState) -> AgentState:
    """Return a simulated patch based on the plan."""
    patch = f"Simulated code change based on plan: {state['plan']}"
    return {"patch": patch, "execution_log": ["code_writer"]}


def test_writer(state: AgentState) -> AgentState:
    """Return a simulated regression test based on the patch."""
    tests = f"Simulated test for patch: {state['patch']}"
    return {"tests": tests, "execution_log": ["test_writer"]}


def pr_opener(state: AgentState) -> AgentState:
    """Return a fake PR URL without contacting GitHub."""
    return {
        "pr_url": "https://github.com/example/project/pull/123 (simulated)",
        "execution_log": ["pr_opener"],
    }


def build_graph():
    """Register nodes and edges, then compile an executable graph."""
    builder = StateGraph(AgentState)

    builder.add_node("code_reader", code_reader)
    builder.add_node("classifier", classifier)
    builder.add_node("researcher", researcher)
    builder.add_node("planner", planner)
    builder.add_node("code_writer", code_writer)
    builder.add_node("test_writer", test_writer)
    builder.add_node("pr_opener", pr_opener)

    builder.add_edge(START, "code_reader")
    builder.add_edge("code_reader", "classifier")
    builder.add_conditional_edges(
        "classifier",
        route_by_complexity,
        {"researcher": "researcher", "planner": "planner"},
    )
    builder.add_edge("researcher", "planner")
    builder.add_edge("planner", "code_writer")
    builder.add_edge("code_writer", "test_writer")
    builder.add_edge("test_writer", "pr_opener")
    builder.add_edge("pr_opener", END)

    return builder.compile()


graph = build_graph()


def run_langgraph_workflow(issue: str) -> AgentState:
    """Invoke the compiled graph with its initial state."""
    return graph.invoke({"issue": issue, "execution_log": []})
