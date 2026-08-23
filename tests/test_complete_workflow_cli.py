"""Tests for the complete workflow CLI's display helpers."""

from dataclasses import dataclass
from multi_agent_system.cli import (
    get_review_payload,
    print_saved_threads,
    resume_saved_workflow,
    save_report_if_requested,
)


@dataclass
class FakeInterrupt:
    """Match the small part of LangGraph's Interrupt used by the CLI."""

    value: dict


def test_get_review_payload_returns_interrupt_value() -> None:
    review = {"issue": "Fix checkout", "test_status": "passed"}
    state = {"__interrupt__": [FakeInterrupt(review)]}

    assert get_review_payload(state) == review


def test_get_review_payload_returns_none_without_interrupt() -> None:
    assert get_review_payload({"test_status": "failed"}) is None


def test_save_report_if_requested_creates_json_file(tmp_path) -> None:
    output = tmp_path / "workflow.json"

    save_report_if_requested({"execution_log": ["planner"]}, str(output))

    assert output.exists()


def test_saved_approval_requires_explicit_pr_permission() -> None:
    arguments = type(
        "Arguments",
        (),
        {
            "decision": "approve",
            "create_pr": False,
            "resume_thread": "run-123",
        },
    )()

    try:
        resume_saved_workflow(arguments)
    except SystemExit as error:
        assert str(error) == "Approval requires --create-pr."
    else:
        raise AssertionError("Unsafe approval was not rejected")
