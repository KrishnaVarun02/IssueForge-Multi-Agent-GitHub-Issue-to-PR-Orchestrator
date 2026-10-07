"""LangGraph workflow whose LLM nodes can be replaced for testing."""

from collections.abc import Callable
from typing import Literal
import time
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from multi_agent_system.checkpointing import open_sqlite_checkpointer
from multi_agent_system.docker_runner import run_patch_in_docker
from multi_agent_system.git_branch_preparer import (
    prepare_local_branch,
    route_after_branch,
)
from multi_agent_system.github_issue_reader import load_issue_input
from multi_agent_system.github_pr_opener import open_github_pull_request
from multi_agent_system.human_approval import human_approval, route_after_approval
from multi_agent_system.langgraph_workflow import (
    AgentState,
    classifier,
    researcher,
    route_by_complexity,
)
from multi_agent_system.llm_code_reader import llm_code_reader
from multi_agent_system.llm_code_writer import llm_code_writer, route_after_code_writer
from multi_agent_system.llm_planner import llm_planner
from multi_agent_system.validated_test_writer import (
    route_after_test_writer,
    validated_test_writer,
)
from multi_agent_system.repository_reader import index_repository

MAX_TEST_EXECUTION_ATTEMPTS = 2

PlannerNode = Callable[[AgentState], AgentState]
CodeReaderNode = Callable[[AgentState], AgentState]
CodeWriterNode = Callable[[AgentState], AgentState]
TestWriterNode = Callable[[AgentState], AgentState]
TestRunnerNode = Callable[[AgentState], AgentState]
IssueReaderNode = Callable[[AgentState], AgentState]
ApprovalNode = Callable[[AgentState], AgentState]
BranchPreparerNode = Callable[[AgentState], AgentState]
PrOpenerNode = Callable[[AgentState], AgentState]
Clock = Callable[[], float]


def timed_node(
    node_name: str,
    node: Callable[[AgentState], AgentState],
    clock: Clock = time.perf_counter,
) -> Callable[[AgentState], AgentState]:
    """Wrap a graph node and accumulate its elapsed milliseconds in state."""

    def run(state: AgentState) -> AgentState:
        started_at = clock()
        result = node(state)
        elapsed_ms = round((clock() - started_at) * 1_000, 3)
        timings = {
            name: dict(values)
            for name, values in state.get("node_timings_ms", {}).items()
        }
        previous = timings.get(
            node_name,
            {"calls": 0, "total_ms": 0.0, "last_ms": 0.0, "max_ms": 0.0},
        )
        timings[node_name] = {
            "calls": previous["calls"] + 1,
            "total_ms": round(previous["total_ms"] + elapsed_ms, 3),
            "last_ms": elapsed_ms,
            "max_ms": max(previous["max_ms"], elapsed_ms),
        }
        return {**result, "node_timings_ms": timings}

    return run


def route_after_test_execution(
    state: AgentState,
) -> Literal["human_approval", "retry_tests", "end"]:
    """Approve passing tests, repair one failure, or stop safely."""
    if state.get("tests_passed", False):
        return "human_approval"
    if (
        state.get("test_status") == "failed"
        and state.get("test_execution_attempts", 0)
        < MAX_TEST_EXECUTION_ATTEMPTS
    ):
        return "retry_tests"
    return "end"


def route_after_llm_node(
    state: AgentState,
) -> Literal["continue", "end"]:
    """Stop the graph when an LLM API request exhausted its retries."""
    return "end" if state.get("llm_status") == "failed" else "continue"


def build_llm_graph(
    planner_node: PlannerNode = llm_planner,
    code_reader_node: CodeReaderNode = llm_code_reader,
    code_writer_node: CodeWriterNode = llm_code_writer,
    test_writer_node: TestWriterNode = validated_test_writer,
    test_runner_node: TestRunnerNode = run_patch_in_docker,
    issue_reader_node: IssueReaderNode = load_issue_input,
    approval_node: ApprovalNode = human_approval,
    branch_preparer_node: BranchPreparerNode = prepare_local_branch,
    pr_opener_node: PrOpenerNode = open_github_pull_request,
    checkpointer=None,
):
    """Build a graph whose LLM nodes can be replaced during tests."""
    builder = StateGraph(AgentState)

    def counted_test_runner(state: AgentState) -> AgentState:
        """Run the injected sandbox node and count bounded repair attempts."""
        result = test_runner_node(state)
        return {
            **result,
            "test_execution_attempts": state.get(
                "test_execution_attempts", 0
            )
            + 1,
        }

    builder.add_node("issue_reader", timed_node("issue_reader", issue_reader_node))
    builder.add_node(
        "repository_indexer",
        timed_node("repository_indexer", index_repository),
    )
    builder.add_node("code_reader", timed_node("code_reader", code_reader_node))
    builder.add_node("classifier", timed_node("classifier", classifier))
    builder.add_node("researcher", timed_node("researcher", researcher))
    builder.add_node("planner", timed_node("planner", planner_node))
    builder.add_node("code_writer", timed_node("code_writer", code_writer_node))
    builder.add_node("test_writer", timed_node("test_writer", test_writer_node))
    builder.add_node("test_runner", timed_node("test_runner", counted_test_runner))
    builder.add_node("human_approval", timed_node("human_approval", approval_node))
    builder.add_node(
        "branch_preparer", timed_node("branch_preparer", branch_preparer_node)
    )
    builder.add_node("pr_opener", timed_node("pr_opener", pr_opener_node))

    builder.add_edge(START, "issue_reader")
    builder.add_edge("issue_reader", "repository_indexer")
    builder.add_edge("repository_indexer", "code_reader")
    builder.add_conditional_edges(
        "code_reader",
        route_after_llm_node,
        {"continue": "classifier", "end": END},
    )
    builder.add_conditional_edges(
        "classifier",
        route_by_complexity,
        {"researcher": "researcher", "planner": "planner"},
    )
    builder.add_edge("researcher", "planner")
    builder.add_conditional_edges(
        "planner",
        route_after_llm_node,
        {"continue": "code_writer", "end": END},
    )
    builder.add_conditional_edges(
        "code_writer",
        route_after_code_writer,
        {"test_writer": "test_writer", "end": END},
    )
    builder.add_conditional_edges(
        "test_writer",
        route_after_test_writer,
        {"test_runner": "test_runner", "end": END},
    )
    builder.add_conditional_edges(
        "test_runner",
        route_after_test_execution,
        {
            "human_approval": "human_approval",
            "retry_tests": "test_writer",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "branch_preparer": "branch_preparer",
            "revise": "code_writer",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "branch_preparer",
        route_after_branch,
        {"pr_opener": "pr_opener", "end": END},
    )
    builder.add_edge("pr_opener", END)

    return builder.compile(checkpointer=checkpointer)


graph = build_llm_graph(checkpointer=InMemorySaver())


def run_llm_workflow(
    issue: str,
    repo_path: str = ".",
    execute_tests: bool = False,
    thread_id: str | None = None,
    checkpoint_path: str | None = None,
    max_llm_tokens: int = 0,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
    max_llm_cost_usd: float = 0.0,
) -> AgentState:
    """Run the graph with the real OpenRouter-powered nodes."""
    workflow_thread_id = thread_id or uuid.uuid4().hex
    input_state = {
        "issue": issue,
        "repo_path": repo_path,
        "execute_tests": execute_tests,
        "workflow_thread_id": workflow_thread_id,
        "llm_max_total_tokens": max_llm_tokens,
        "llm_input_cost_per_million": input_cost_per_million,
        "llm_output_cost_per_million": output_cost_per_million,
        "llm_max_cost_usd": max_llm_cost_usd,
        "execution_log": [],
    }
    config = {"configurable": {"thread_id": workflow_thread_id}}
    if checkpoint_path:
        with open_sqlite_checkpointer(checkpoint_path) as checkpointer:
            persistent_graph = build_llm_graph(checkpointer=checkpointer)
            return persistent_graph.invoke(input_state, config=config)
    return graph.invoke(input_state, config=config)


def run_github_issue_workflow(
    issue_url: str,
    repo_path: str = ".",
    execute_tests: bool = False,
    thread_id: str | None = None,
    checkpoint_path: str | None = None,
    max_llm_tokens: int = 0,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
    max_llm_cost_usd: float = 0.0,
) -> AgentState:
    """Run the complete graph starting from a GitHub issue URL."""
    workflow_thread_id = thread_id or uuid.uuid4().hex
    input_state = {
        "issue_url": issue_url,
        "repo_path": repo_path,
        "execute_tests": execute_tests,
        "workflow_thread_id": workflow_thread_id,
        "llm_max_total_tokens": max_llm_tokens,
        "llm_input_cost_per_million": input_cost_per_million,
        "llm_output_cost_per_million": output_cost_per_million,
        "llm_max_cost_usd": max_llm_cost_usd,
        "execution_log": [],
    }
    config = {"configurable": {"thread_id": workflow_thread_id}}
    if checkpoint_path:
        with open_sqlite_checkpointer(checkpoint_path) as checkpointer:
            persistent_graph = build_llm_graph(checkpointer=checkpointer)
            return persistent_graph.invoke(input_state, config=config)
    return graph.invoke(input_state, config=config)


def resume_workflow(
    thread_id: str,
    decision: str,
    feedback: str = "",
    checkpoint_path: str | None = None,
) -> AgentState:
    """Resume a paused workflow using the same checkpoint thread ID."""
    command = Command(resume={"decision": decision, "feedback": feedback})
    config = {"configurable": {"thread_id": thread_id}}
    if checkpoint_path:
        with open_sqlite_checkpointer(checkpoint_path) as checkpointer:
            persistent_graph = build_llm_graph(checkpointer=checkpointer)
            snapshot = persistent_graph.get_state(config)
            if not snapshot.values:
                raise ValueError(f"No saved workflow has thread ID '{thread_id}'.")
            if not snapshot.interrupts:
                raise ValueError(
                    f"Workflow '{thread_id}' is finished and is not waiting "
                    "for a decision."
                )
            return persistent_graph.invoke(command, config=config)
    return graph.invoke(command, config=config)


def load_workflow_state(
    thread_id: str, checkpoint_path: str
) -> AgentState:
    """Load the latest state for a persisted workflow without executing nodes."""
    config = {"configurable": {"thread_id": thread_id}}
    with open_sqlite_checkpointer(checkpoint_path) as checkpointer:
        persistent_graph = build_llm_graph(checkpointer=checkpointer)
        snapshot = persistent_graph.get_state(config)
        if not snapshot.values:
            raise ValueError(f"No saved workflow has thread ID '{thread_id}'.")
        return snapshot.values


def list_workflow_threads(checkpoint_path: str) -> list[dict[str, object]]:
    """Return one concise entry for every thread stored in SQLite."""
    with open_sqlite_checkpointer(checkpoint_path) as checkpointer:
        persistent_graph = build_llm_graph(checkpointer=checkpointer)
        thread_ids: list[str] = []
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config["configurable"]["thread_id"]
            if thread_id not in thread_ids:
                thread_ids.append(thread_id)

        threads = []
        for thread_id in thread_ids:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = persistent_graph.get_state(config)
            state = snapshot.values
            threads.append(
                {
                    "thread_id": thread_id,
                    "status": (
                        "waiting_for_review"
                        if snapshot.interrupts
                        else "finished"
                    ),
                    "issue": state.get("issue_title", state.get("issue", "")),
                    "test_status": state.get("test_status", "not_reached"),
                }
            )
        return threads


def main() -> None:
    """Run the compiled LLM graph from the command line."""
    issue = (
        "Users cannot complete checkout after changing currency "
        "when a discount is active"
    )
    result = run_llm_workflow(issue, repo_path=".")

    print("Execution order:")
    for agent_name in result["execution_log"]:
        print(f"- {agent_name}")

    print("\nLLM-selected files:")
    for file_path in result["selected_files"]:
        print(f"- {file_path}")

    print("\nWhy these files were selected:")
    print(result["file_selection_reasoning"])

    print("\nLLM-generated plan:")
    print(result["plan"])

    print("\nSteps:")
    for number, step in enumerate(result["plan_steps"], start=1):
        print(f"{number}. {step}")

    print("\nRisks:")
    for risk in result["plan_risks"]:
        print(f"- {risk}")

    print("\nLLM-generated patch summary:")
    print(result["patch_summary"])

    print("\nProposed patch (not applied):")
    print(result["patch"])

    print("\nLLM-generated test summary:")
    print(result["test_summary"])

    print("\nProposed test patch (not applied):")
    print(result["test_patch"])

    print("\nSuggested test command (not executed):")
    print(result["test_command"])

    print("\nSandbox status:")
    print(result["test_status"])
    print(result["test_output"])

    if result.get("pr_url"):
        print("\nPull request URL:")
        print(result["pr_url"])
    else:
        print("\nPR creation status:")
        print(result.get("pr_status", "not reached"))


if __name__ == "__main__":
    main()
