"""Tests for durable SQLite workflow checkpoints."""

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.types import interrupt

from multi_agent_system.checkpointing import open_sqlite_checkpointer
from multi_agent_system.llm_langgraph_workflow import (
    list_workflow_threads,
    resume_workflow,
)


class CheckpointState(TypedDict, total=False):
    """Minimal state used to verify durable interrupt persistence."""

    task: str
    decision: str
    execution_log: Annotated[list[str], add]


def approval_node(state: CheckpointState) -> CheckpointState:
    """Pause until a decision is supplied through LangGraph resume."""
    decision = interrupt({"question": f"Approve this task: {state['task']}?"})
    return {"decision": str(decision), "execution_log": ["approval"]}


def build_checkpoint_graph(checkpointer):
    """Compile the minimal persisted graph used by this unit test."""
    builder = StateGraph(CheckpointState)
    builder.add_node("approval", approval_node)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=checkpointer)


def test_sqlite_graph_resumes_after_database_is_reopened(tmp_path) -> None:
    database = tmp_path / "nested" / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "restart-test"}}

    with open_sqlite_checkpointer(database) as first_checkpointer:
        first_graph = build_checkpoint_graph(first_checkpointer)
        paused = first_graph.invoke(
            {"task": "Test persistence", "execution_log": []}, config=config
        )

    assert paused["__interrupt__"]
    assert database.exists()

    with open_sqlite_checkpointer(database) as second_checkpointer:
        second_graph = build_checkpoint_graph(second_checkpointer)
        resumed = second_graph.invoke(Command(resume="approve"), config=config)

    assert resumed["decision"] == "approve"
    assert resumed["execution_log"] == ["approval"]


def test_unknown_real_workflow_thread_fails_before_running_nodes(tmp_path) -> None:
    database = tmp_path / "empty.sqlite3"

    try:
        resume_workflow(
            "missing-thread",
            "reject",
            checkpoint_path=str(database),
        )
    except ValueError as error:
        assert "No saved workflow" in str(error)
    else:
        raise AssertionError("Missing workflow was incorrectly started")

    assert list_workflow_threads(str(database)) == []
