"""Pause LangGraph so a person can approve or reject PR creation."""

from typing import Literal

from langgraph.types import interrupt

from multi_agent_system.langgraph_workflow import AgentState

MAX_FEEDBACK_CHARS = 2_000
MAX_REVISION_ATTEMPTS = 2


def human_approval(state: AgentState) -> AgentState:
    """Interrupt execution and convert the resume value into state updates."""
    decision = interrupt(
        {
            "type": "pull_request_approval",
            "message": "Review the tested changes before creating a pull request.",
            "issue": state.get("issue_title", state["issue"]),
            "patch_summary": state["patch_summary"],
            "changed_files": state["changed_files"],
            "patch": state["patch"],
            "test_summary": state["test_summary"],
            "test_files": state["test_files"],
            "test_patch": state["test_patch"],
            "test_status": state["test_status"],
            "llm_prompt_tokens": state.get("llm_prompt_tokens", 0),
            "llm_completion_tokens": state.get("llm_completion_tokens", 0),
            "llm_total_tokens": state.get("llm_total_tokens", 0),
            "llm_response_count": state.get("llm_response_count", 0),
            "llm_max_total_tokens": state.get("llm_max_total_tokens", 0),
            "llm_input_cost_per_million": state.get(
                "llm_input_cost_per_million", 0.0
            ),
            "llm_output_cost_per_million": state.get(
                "llm_output_cost_per_million", 0.0
            ),
            "llm_max_cost_usd": state.get("llm_max_cost_usd", 0.0),
            "llm_usage_by_node": state.get("llm_usage_by_node", {}),
            "llm_response_trace": state.get("llm_response_trace", []),
            "node_timings_ms": state.get("node_timings_ms", {}),
            "revision_count": state.get("revision_count", 0),
            "max_revision_attempts": MAX_REVISION_ATTEMPTS,
        }
    )

    if not isinstance(decision, dict):
        choice = "reject"
        feedback = "Invalid approval response."
        revision_requested = False
    else:
        choice = str(decision.get("decision", "reject")).strip().lower()
        feedback = str(decision.get("feedback", "")).strip()[:MAX_FEEDBACK_CHARS]
        revision_requested = choice == "reject" and bool(feedback)

    approved = choice == "approve"
    revision_count = state.get("revision_count", 0)
    if revision_requested:
        revision_count += 1

    if approved:
        approval_status = "approved"
    elif revision_requested and revision_count <= MAX_REVISION_ATTEMPTS:
        approval_status = "revision_requested"
    elif revision_requested:
        approval_status = "revision_limit_reached"
    else:
        approval_status = "rejected"

    return {
        "pull_request_approved": approved,
        "approval_status": approval_status,
        "approval_feedback": feedback,
        "revision_count": revision_count,
        "execution_log": ["human_approval"],
    }


def route_after_approval(
    state: AgentState,
) -> Literal["branch_preparer", "revise", "end"]:
    """Approve, revise with bounded feedback, or stop."""
    if state["pull_request_approved"]:
        return "branch_preparer"
    if state.get("approval_status") == "revision_requested":
        return "revise"
    return "end"
