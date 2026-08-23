"""Tests for LangGraph pause-and-resume human approval."""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from multi_agent_system.human_approval import human_approval, route_after_approval
from multi_agent_system.langgraph_workflow import AgentState


def _approval_graph():
    builder = StateGraph(AgentState)
    builder.add_node("human_approval", human_approval)
    builder.add_edge(START, "human_approval")
    builder.add_edge("human_approval", END)
    return builder.compile(checkpointer=InMemorySaver())


def _review_state() -> AgentState:
    return {
        "issue": "Fix checkout",
        "patch_summary": "Recalculate checkout totals",
        "changed_files": ["checkout.py"],
        "patch": "code patch",
        "test_summary": "Test currency changes",
        "test_files": ["tests/test_checkout.py"],
        "test_patch": "test patch",
        "test_status": "passed",
        "execution_log": [],
    }


def test_graph_pauses_and_resumes_with_approval() -> None:
    graph = _approval_graph()
    config = {"configurable": {"thread_id": "approval-test"}}

    paused = graph.invoke(_review_state(), config=config)
    assert paused["__interrupt__"][0].value["type"] == "pull_request_approval"
    assert "approval_status" not in paused

    resumed = graph.invoke(
        Command(resume={"decision": "approve", "feedback": "Looks good"}),
        config=config,
    )
    assert resumed["pull_request_approved"] is True
    assert resumed["approval_status"] == "approved"
    assert resumed["approval_feedback"] == "Looks good"


def test_rejection_is_recorded() -> None:
    graph = _approval_graph()
    config = {"configurable": {"thread_id": "rejection-test"}}
    graph.invoke(_review_state(), config=config)

    resumed = graph.invoke(
        Command(resume={"decision": "reject", "feedback": "Needs smaller diff"}),
        config=config,
    )
    assert resumed["pull_request_approved"] is False
    assert resumed["approval_status"] == "revision_requested"
    assert resumed["revision_count"] == 1


def test_rejection_with_feedback_requests_bounded_revision() -> None:
    state = {
        "pull_request_approved": False,
        "approval_status": "revision_requested",
        "revision_count": 1,
    }

    assert route_after_approval(state) == "revise"


def test_revision_limit_stops_the_graph() -> None:
    state = {
        "pull_request_approved": False,
        "approval_status": "revision_limit_reached",
        "revision_count": 3,
    }

    assert route_after_approval(state) == "end"
