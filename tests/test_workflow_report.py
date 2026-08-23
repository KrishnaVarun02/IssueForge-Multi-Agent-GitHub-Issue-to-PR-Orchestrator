"""Tests for secret-safe workflow observability reports."""

import json
from pathlib import Path

from multi_agent_system.workflow_report import (
    build_workflow_report,
    save_workflow_report,
    workflow_status,
)


def test_report_summarizes_a_successful_run_without_patch_contents() -> None:
    state = {
        "workflow_thread_id": "run-123",
        "issue_title": "Fix checkout",
        "execution_log": ["issue_reader", "test_runner", "pr_opener"],
        "patch": "SECRET SOURCE CONTENT",
        "changed_files": ["checkout.py"],
        "test_files": ["tests/test_checkout.py"],
        "test_status": "passed",
        "tests_passed": True,
        "pr_status": "created",
        "pr_url": "https://github.com/example/repo/pull/1",
        "llm_prompt_tokens": 100,
        "llm_completion_tokens": 25,
        "llm_total_tokens": 125,
        "llm_response_count": 2,
        "llm_max_total_tokens": 1000,
        "llm_input_cost_per_million": 2.0,
        "llm_output_cost_per_million": 8.0,
        "llm_max_cost_usd": 1.0,
        "llm_usage_by_node": {
            "llm_planner": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125,
                "responses": 2,
            }
        },
        "node_timings_ms": {
            "llm_planner": {
                "calls": 2,
                "total_ms": 125.5,
                "last_ms": 60.0,
                "max_ms": 65.5,
            }
        },
        "llm_response_trace": [
            {
                "sequence": 1,
                "node": "llm_planner",
                "model": "provider/model",
                "response_id": "response-1",
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "total_tokens": 125,
            }
        ],
    }

    report = build_workflow_report(state)

    assert report["status"] == "pull_request_created"
    assert report["execution"]["node_count"] == 3
    assert report["generated_change"]["files"] == ["checkout.py"]
    assert report["llm_usage"]["total_tokens"] == 125
    assert report["llm_usage"]["responses"] == 2
    assert report["llm_usage"]["token_limit"] == 1000
    assert report["llm_usage"]["estimated_cost_usd"] == 0.0004
    assert report["llm_usage"]["estimated_cost_limit_usd"] == 1.0
    assert report["llm_usage"]["by_node"]["llm_planner"] == {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
        "responses": 2,
        "estimated_cost_usd": 0.0004,
    }
    assert report["execution"]["timings_ms"]["llm_planner"]["calls"] == 2
    assert report["llm_usage"]["response_trace"][0]["response_id"] == (
        "response-1"
    )
    assert "SECRET SOURCE CONTENT" not in json.dumps(report)


def test_report_records_a_generation_failure() -> None:
    state = {
        "code_generation_status": "failed",
        "code_generation_error": "Model returned incomplete JSON.",
    }

    assert workflow_status(state) == "code_generation_failed"
    assert build_workflow_report(state)["errors"] == [
        "Model returned incomplete JSON."
    ]


def test_report_distinguishes_skipped_and_timed_out_tests() -> None:
    assert workflow_status({"test_status": "awaiting_approval"}) == (
        "tests_not_executed"
    )
    assert workflow_status({"test_status": "timed_out"}) == "tests_failed"


def test_save_workflow_report_writes_readable_json(tmp_path: Path) -> None:
    output = save_workflow_report(
        {"execution_log": ["planner"]}, tmp_path / "reports" / "run.json"
    )

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert output.is_absolute()
    assert saved["execution"]["nodes"] == ["planner"]
    assert saved["status"] == "completed"
