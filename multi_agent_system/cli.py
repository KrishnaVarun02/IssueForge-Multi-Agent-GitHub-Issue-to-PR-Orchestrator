"""Production CLI for the complete GitHub issue-to-pull-request workflow."""

import argparse
from pathlib import Path
from typing import Any

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.openrouter_client import estimated_llm_cost_usd
from multi_agent_system.settings import SettingsError, WorkflowSettings
from multi_agent_system.llm_langgraph_workflow import (
    list_workflow_threads,
    load_workflow_state,
    resume_workflow,
    run_github_issue_workflow,
)
from multi_agent_system.workflow_report import save_workflow_report
from multi_agent_system.preflight import (
    preflight_passed,
    run_preflight_checks,
)


def parse_arguments() -> argparse.Namespace:
    """Convert terminal arguments into a Python namespace object."""
    try:
        settings = WorkflowSettings.from_environment(require_api_key=False)
    except SettingsError as error:
        raise SystemExit(f"Configuration error: {error}") from error
    parser = argparse.ArgumentParser(
        description="Run the complete GitHub issue-to-pull-request workflow."
    )
    parser.add_argument("issue_url", nargs="?", help="Full GitHub issue URL")
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Local repository root (default: current directory)",
    )
    parser.add_argument(
        "--execute-tests",
        action="store_true",
        help="Run generated changes in the Docker sandbox",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Interactively reject and revise without enabling PR creation",
    )
    parser.add_argument(
        "--create-pr",
        action="store_true",
        help="Allow an approved run to push its branch and create a PR",
    )
    parser.add_argument(
        "--report-file",
        help="Write a secret-safe JSON summary of the workflow run",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=".agent/checkpoints.sqlite3",
        help="SQLite checkpoint file used to resume interrupted runs",
    )
    parser.add_argument(
        "--resume-thread",
        help="Resume a saved workflow thread instead of starting a new run",
    )
    parser.add_argument(
        "--decision",
        choices=("approve", "reject"),
        help="Human decision supplied when resuming a saved thread",
    )
    parser.add_argument(
        "--feedback",
        default="",
        help="Revision feedback supplied with a rejected saved thread",
    )
    parser.add_argument(
        "--list-threads",
        action="store_true",
        help="List saved workflow IDs and whether each can be resumed",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check local prerequisites without starting the workflow",
    )
    parser.add_argument(
        "--max-llm-tokens",
        type=int,
        default=settings.max_llm_tokens,
        help="Stop new LLM calls after this total-token usage (0: unlimited)",
    )
    parser.add_argument(
        "--input-cost-per-million",
        type=float,
        default=settings.input_cost_per_million,
        help="Model input-token price in USD per million tokens",
    )
    parser.add_argument(
        "--output-cost-per-million",
        type=float,
        default=settings.output_cost_per_million,
        help="Model output-token price in USD per million tokens",
    )
    parser.add_argument(
        "--max-llm-cost-usd",
        type=float,
        default=settings.max_llm_cost_usd,
        help="Stop new LLM calls after this estimated USD cost (0: unlimited)",
    )
    return parser.parse_args()


def get_review_payload(state: AgentState) -> dict[str, Any] | None:
    """Return the human-review data when LangGraph has paused."""
    interrupts = state.get("__interrupt__", [])
    if not interrupts:
        return None
    return interrupts[0].value


def print_llm_usage_by_node(state: dict[str, Any]) -> None:
    """Print token and estimated-cost totals for each LLM agent."""
    usage_by_node = state.get("llm_usage_by_node", {})
    if not usage_by_node:
        return

    print("LLM usage by agent:")
    for node_name, usage in usage_by_node.items():
        cost_state = {
            **usage,
            "llm_prompt_tokens": usage["prompt_tokens"],
            "llm_completion_tokens": usage["completion_tokens"],
            "llm_input_cost_per_million": state.get(
                "llm_input_cost_per_million", 0.0
            ),
            "llm_output_cost_per_million": state.get(
                "llm_output_cost_per_million", 0.0
            ),
        }
        print(
            f"- {node_name}: {usage['responses']} responses, "
            f"{usage['total_tokens']} tokens, "
            f"${estimated_llm_cost_usd(cost_state):.6f} estimated"
        )


def print_node_timings(state: dict[str, Any]) -> None:
    """Print accumulated runtime measurements for completed graph nodes."""
    timings = state.get("node_timings_ms", {})
    if not timings:
        return

    print("Node timings:")
    for node_name, timing in timings.items():
        print(
            f"- {node_name}: {timing['calls']} calls, "
            f"{timing['total_ms']:.3f} ms total, "
            f"{timing['max_ms']:.3f} ms slowest"
        )


def print_llm_response_trace(state: dict[str, Any]) -> None:
    """Print safe identifying metadata for each received LLM response."""
    trace = state.get("llm_response_trace", [])
    if not trace:
        return

    print("LLM response trace:")
    for response in trace:
        identifier = response.get("response_id") or "not provided"
        print(
            f"- #{response['sequence']} {response['node']}: "
            f"{response['model']}, {response['total_tokens']} tokens, "
            f"{response.get('outcome', 'unknown')}, response ID {identifier}"
        )


def print_review(review: dict[str, Any]) -> None:
    """Display the important generated artifacts before approval."""
    print("\n=== HUMAN REVIEW REQUIRED ===")
    print(f"Issue: {review['issue']}")
    print(f"Test status: {review['test_status']}")
    print(
        "Revision: "
        f"{review.get('revision_count', 0)} of "
        f"{review.get('max_revision_attempts', 2)}"
    )
    print(f"\nPatch summary:\n{review['patch_summary']}")
    print(f"\nChanged files: {', '.join(review['changed_files'])}")
    print(f"\nCode patch:\n{review['patch']}")
    print(f"\nTest summary:\n{review['test_summary']}")
    print(f"\nTest files: {', '.join(review['test_files'])}")
    print(f"\nTest patch:\n{review['test_patch']}")
    print(
        "\nLLM usage: "
        f"{review.get('llm_response_count', 0)} responses, "
        f"{review.get('llm_prompt_tokens', 0)} prompt tokens, "
        f"{review.get('llm_completion_tokens', 0)} completion tokens, "
        f"{review.get('llm_total_tokens', 0)} total tokens"
    )
    if review.get("llm_max_total_tokens", 0):
        print(f"LLM token limit: {review['llm_max_total_tokens']}")
    if (
        review.get("llm_input_cost_per_million", 0.0)
        or review.get("llm_output_cost_per_million", 0.0)
    ):
        print(f"Estimated LLM cost: ${estimated_llm_cost_usd(review):.6f}")
    if review.get("llm_max_cost_usd", 0.0):
        print(f"Estimated cost limit: ${review['llm_max_cost_usd']:.6f}")
    print_llm_usage_by_node(review)
    print_llm_response_trace(review)
    print_node_timings(review)


def print_final_status(state: AgentState) -> None:
    """Explain where the graph stopped and whether it created a PR."""
    print("\nExecution order:")
    for node_name in state.get("execution_log", []):
        print(f"- {node_name}")

    print(
        "\nLLM usage: "
        f"{state.get('llm_response_count', 0)} responses, "
        f"{state.get('llm_prompt_tokens', 0)} prompt tokens, "
        f"{state.get('llm_completion_tokens', 0)} completion tokens, "
        f"{state.get('llm_total_tokens', 0)} total tokens"
    )
    if state.get("llm_max_total_tokens", 0):
        print(f"LLM token limit: {state['llm_max_total_tokens']}")
    if (
        state.get("llm_input_cost_per_million", 0.0)
        or state.get("llm_output_cost_per_million", 0.0)
    ):
        print(f"Estimated LLM cost: ${estimated_llm_cost_usd(state):.6f}")
    if state.get("llm_max_cost_usd", 0.0):
        print(f"Estimated cost limit: ${state['llm_max_cost_usd']:.6f}")
    print_llm_usage_by_node(state)
    print_llm_response_trace(state)
    print_node_timings(state)

    if state.get("pr_url"):
        print(f"\nPull request: {state['pr_url']}")
    elif state.get("pr_status"):
        print(f"\nPR status: {state['pr_status']}")
    elif state.get("test_status") != "passed":
        if (
            state.get("llm_status") == "failed"
            and state.get("failed_node")
            not in {"llm_code_writer", "llm_test_writer"}
        ):
            print(f"\nWorkflow stopped at {state['failed_node']}:")
            print(state.get("llm_error", "Unknown LLM request error"))
        elif state.get("code_generation_status") == "failed":
            print("\nWorkflow stopped at code generation:")
            print(state.get("code_generation_error", "Unknown generation error"))
        elif state.get("test_generation_status") == "failed":
            print("\nWorkflow stopped at test generation:")
            print(state.get("test_generation_error", "Unknown generation error"))
        else:
            print(f"\nWorkflow stopped at tests: {state.get('test_status')}")
            print(state.get("test_output", ""))
            print("\nGenerated change that was tested:")
            print(f"Summary: {state.get('patch_summary', 'Not available')}")
            print(f"Files: {', '.join(state.get('changed_files', []))}")
            print(state.get("patch", ""))
            print("\nGenerated tests:")
            print(f"Summary: {state.get('test_summary', 'Not available')}")
            print(f"Files: {', '.join(state.get('test_files', []))}")
            print(state.get("test_patch", ""))
    elif state.get("approval_status"):
        print(f"\nApproval status: {state['approval_status']}")


def save_report_if_requested(
    state: AgentState, report_file: str | None
) -> None:
    """Save and display the optional JSON report path."""
    if not report_file:
        return
    path = save_workflow_report(state, report_file)
    print(f"\nWorkflow report: {path}")


def resume_saved_workflow(arguments: argparse.Namespace) -> AgentState:
    """Validate safety gates and resume an interrupt stored in SQLite."""
    if not arguments.decision:
        raise SystemExit("--decision is required with --resume-thread.")
    if arguments.decision == "approve" and not arguments.create_pr:
        raise SystemExit("Approval requires --create-pr.")

    try:
        saved_state = load_workflow_state(
            arguments.resume_thread, arguments.checkpoint_db
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.decision == "approve":
        repository_name = (
            f"{saved_state['repository_owner']}/{saved_state['repository_name']}"
        )
        confirmed_repository = input(
            f"Type {repository_name} to confirm the target repository: "
        )
        if confirmed_repository.strip() != repository_name:
            raise SystemExit("Repository confirmation failed; nothing was changed.")

    try:
        return resume_workflow(
            arguments.resume_thread,
            arguments.decision,
            arguments.feedback,
            checkpoint_path=arguments.checkpoint_db,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


def print_saved_threads(checkpoint_db: str) -> None:
    """Print copyable IDs and clearly mark resumable workflows."""
    threads = list_workflow_threads(checkpoint_db)
    if not threads:
        print("No saved workflows found.")
        return

    print("Saved workflows:")
    for thread in threads:
        print(
            f"- {thread['thread_id']} | {thread['status']} | "
            f"tests: {thread['test_status']} | {thread['issue']}"
        )


def print_preflight(repo_path: str, execute_tests: bool, create_pr: bool) -> bool:
    """Display prerequisite results and return whether the run is ready."""
    checks = run_preflight_checks(
        repo_path, execute_tests=execute_tests, create_pr=create_pr
    )
    print("Preflight checks:")
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"- [{marker}] {check.name}: {check.message}")
    return preflight_passed(checks)


def display_resumed_state(
    state: AgentState, thread_id: str, report_file: str | None
) -> None:
    """Display either another review interrupt or the final resumed state."""
    print(f"\nWorkflow thread ID: {thread_id}")
    review = get_review_payload(state)
    if review is not None:
        print_review(review)
    else:
        print_final_status(state)
    save_report_if_requested(state, report_file)


def main() -> None:
    """Start, review, and optionally resume the complete workflow."""
    arguments = parse_arguments()
    if arguments.max_llm_tokens < 0:
        raise SystemExit("--max-llm-tokens must be zero or greater.")
    if (
        arguments.input_cost_per_million < 0
        or arguments.output_cost_per_million < 0
    ):
        raise SystemExit("LLM token prices must be zero or greater.")
    if arguments.max_llm_cost_usd < 0:
        raise SystemExit("--max-llm-cost-usd must be zero or greater.")
    if (
        arguments.max_llm_cost_usd > 0
        and arguments.input_cost_per_million == 0
        and arguments.output_cost_per_million == 0
    ):
        raise SystemExit(
            "A cost limit requires at least one non-zero token price."
        )
    if arguments.list_threads:
        print_saved_threads(arguments.checkpoint_db)
        return
    if arguments.resume_thread:
        state = resume_saved_workflow(arguments)
        display_resumed_state(
            state, arguments.resume_thread, arguments.report_file
        )
        return

    repository = Path(arguments.repo_path).resolve()
    ready = print_preflight(
        str(repository), arguments.execute_tests, arguments.create_pr
    )
    if not ready:
        raise SystemExit("Preflight failed; workflow was not started.")
    if arguments.preflight_only:
        return
    if not arguments.issue_url:
        raise SystemExit("issue_url is required when starting a workflow.")

    state = run_github_issue_workflow(
        arguments.issue_url,
        repo_path=str(repository),
        execute_tests=arguments.execute_tests,
        checkpoint_path=arguments.checkpoint_db,
        max_llm_tokens=arguments.max_llm_tokens,
        input_cost_per_million=arguments.input_cost_per_million,
        output_cost_per_million=arguments.output_cost_per_million,
        max_llm_cost_usd=arguments.max_llm_cost_usd,
    )
    print(f"\nWorkflow thread ID: {state['workflow_thread_id']}")
    while True:
        review = get_review_payload(state)
        if review is None:
            print_final_status(state)
            save_report_if_requested(state, arguments.report_file)
            return

        print_review(review)
        if not arguments.review and not arguments.create_pr:
            print(
                "\nPreview complete. Nothing was pushed. Add --review to "
                "request revisions or --create-pr to enable PR approval."
            )
            save_report_if_requested(state, arguments.report_file)
            return

        decision = input(
            "\nType approve, reject, or stop: "
        ).strip().lower()
        if decision == "approve" and not arguments.create_pr:
            print("\nPreview approved. Nothing was pushed because --create-pr is off.")
            save_report_if_requested(state, arguments.report_file)
            return
        if decision == "stop":
            print("\nReview stopped. Nothing was pushed.")
            save_report_if_requested(state, arguments.report_file)
            return

        feedback = ""
        if decision == "approve":
            repository_name = (
                f"{state['repository_owner']}/{state['repository_name']}"
            )
            confirmed_repository = input(
                f"Type {repository_name} to confirm the target repository: "
            )
            if confirmed_repository.strip() != repository_name:
                decision = "reject"
                feedback = "Repository confirmation failed."
        else:
            decision = "reject"
            feedback = input(
                "Describe the required revision, or leave blank to reject: "
            ).strip()

        state = resume_workflow(
            state["workflow_thread_id"],
            decision,
            feedback,
            checkpoint_path=arguments.checkpoint_db,
        )


if __name__ == "__main__":
    main()
