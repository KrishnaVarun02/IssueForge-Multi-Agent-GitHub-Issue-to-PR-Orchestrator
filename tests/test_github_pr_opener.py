"""Unit tests for the real GitHub PR opener without network access."""

import pytest

from multi_agent_system.github_pr_opener import (
    build_pr_command,
    open_github_pull_request,
    parse_github_remote,
)


def approved_state() -> dict:
    """Return the minimum state needed to describe an approved PR."""
    return {
        "issue": "Fix checkout",
        "issue_title": "Fix checkout after currency change",
        "issue_number": 7,
        "repo_path": "/tmp/example",
        "tests_passed": True,
        "pull_request_approved": True,
        "branch_prepared": True,
        "branch_name": "agent/issue-7-fix-checkout",
        "commit_sha": "abc123",
        "patch_summary": "Repair currency conversion during checkout.",
        "changed_files": ["checkout.py"],
        "test_files": ["tests/test_checkout.py"],
        "test_status": "passed",
        "test_command": "python3 -m pytest",
    }


@pytest.mark.parametrize(
    "remote",
    [
        "https://github.com/KrishnaVarun02/Multi-Agent-Orchestration-System.git",
        "git@github.com:KrishnaVarun02/Multi-Agent-Orchestration-System.git",
    ],
)
def test_parse_github_remote(remote: str) -> None:
    assert parse_github_remote(remote) == (
        "KrishnaVarun02",
        "Multi-Agent-Orchestration-System",
    )


def test_parse_github_remote_rejects_other_hosts() -> None:
    with pytest.raises(ValueError):
        parse_github_remote("https://example.com/owner/repository.git")


def test_build_pr_command_uses_state() -> None:
    command = build_pr_command(approved_state(), "owner", "repository")

    assert command[:3] == ["gh", "pr", "create"]
    assert command[command.index("--head") + 1] == "agent/issue-7-fix-checkout"
    assert command[command.index("--base") + 1] == "main"
    assert "Closes #7" in command[command.index("--body") + 1]


def test_pr_opener_stops_before_commands_when_tests_failed() -> None:
    state = approved_state()
    state["tests_passed"] = False

    result = open_github_pull_request(state)

    assert result["pr_status"] == "tests_required"
    assert result["pr_url"] == ""
