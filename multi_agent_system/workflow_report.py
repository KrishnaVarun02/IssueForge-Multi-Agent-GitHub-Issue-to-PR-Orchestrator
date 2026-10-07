"""Build a small, secret-safe JSON report from a completed workflow state."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.openrouter_client import estimated_llm_cost_usd


def workflow_status(state: AgentState) -> str:
    """Translate detailed state fields into one easy-to-read final status."""
    if state.get("pr_url"):
        return "pull_request_created"
    if state.get("__interrupt__"):
        return "awaiting_human_review"
    if state.get("llm_status") == "failed":
        return "llm_request_failed"
    if state.get("code_generation_status") == "failed":
        return "code_generation_failed"
    if state.get("test_generation_status") == "failed":
        return "test_generation_failed"
    if state.get("test_status") in {
        "failed",
        "error",
        "patch_rejected",
        "timed_out",
    }:
        return "tests_failed"
    if state.get("test_status") == "awaiting_approval":
        return "tests_not_executed"
    if state.get("approval_status") == "rejected":
        return "rejected"
    if state.get("branch_status") and not state.get("branch_prepared", False):
        return "branch_preparation_failed"
    if state.get("pr_status") and state.get("pr_status") != "created":
        return "pull_request_failed"
    return "completed"


def _errors(state: AgentState) -> list[str]:
    """Collect concise errors while excluding prompts, source, and secrets."""
    errors = []
    for key in (
        "llm_error",
        "code_generation_error",
        "test_generation_error",
    ):
        value = state.get(key)
        if value and str(value) not in errors:
            errors.append(str(value))

    if state.get("test_status") in {
        "failed",
        "error",
        "patch_rejected",
        "timed_out",
    }:
        errors.append(f"Sandbox tests ended with status: {state['test_status']}")
    return errors


def build_workflow_report(state: AgentState) -> dict[str, Any]:
    """Select operational metadata from LangGraph's larger shared state."""
    execution_log = list(state.get("execution_log", []))
    usage_by_node = {}
    for node_name, usage in state.get("llm_usage_by_node", {}).items():
        cost_state = {
            "llm_prompt_tokens": usage["prompt_tokens"],
            "llm_completion_tokens": usage["completion_tokens"],
            "llm_input_cost_per_million": state.get(
                "llm_input_cost_per_million", 0.0
            ),
            "llm_output_cost_per_million": state.get(
                "llm_output_cost_per_million", 0.0
            ),
        }
        usage_by_node[node_name] = {
            **usage,
            "estimated_cost_usd": estimated_llm_cost_usd(cost_state),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow_thread_id": state.get("workflow_thread_id", ""),
        "status": workflow_status(state),
        "issue": {
            "url": state.get("issue_url", ""),
            "number": state.get("issue_number"),
            "title": state.get("issue_title") or state.get("issue", ""),
        },
        "execution": {
            "node_count": len(execution_log),
            "nodes": execution_log,
            "timings_ms": state.get("node_timings_ms", {}),
        },
        "llm_usage": {
            "responses": state.get("llm_response_count", 0),
            "prompt_tokens": state.get("llm_prompt_tokens", 0),
            "completion_tokens": state.get("llm_completion_tokens", 0),
            "total_tokens": state.get("llm_total_tokens", 0),
            "token_limit": state.get("llm_max_total_tokens", 0),
            "input_cost_per_million_usd": state.get(
                "llm_input_cost_per_million", 0.0
            ),
            "output_cost_per_million_usd": state.get(
                "llm_output_cost_per_million", 0.0
            ),
            "estimated_cost_usd": estimated_llm_cost_usd(state),
            "estimated_cost_limit_usd": state.get("llm_max_cost_usd", 0.0),
            "by_node": usage_by_node,
            "response_trace": state.get("llm_response_trace", []),
        },
        "generated_change": {
            "summary": state.get("patch_summary", ""),
            "files": list(state.get("changed_files", [])),
        },
        "generated_tests": {
            "summary": state.get("test_summary", ""),
            "files": list(state.get("test_files", [])),
            "command": state.get("test_command", ""),
            "status": state.get("test_status", "not_run"),
            "passed": state.get("tests_passed", False),
            "sandbox": state.get("sandbox_kind", ""),
        },
        "delivery": {
            "approval": state.get("approval_status", "not_requested"),
            "branch": state.get("branch_name", ""),
            "commit": state.get("commit_sha", ""),
            "pr_status": state.get("pr_status", "not_created"),
            "pr_url": state.get("pr_url", ""),
        },
        "errors": _errors(state),
    }


def save_workflow_report(state: AgentState, output_path: str | Path) -> Path:
    """Serialize the report as readable JSON and return its absolute path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_workflow_report(state), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
