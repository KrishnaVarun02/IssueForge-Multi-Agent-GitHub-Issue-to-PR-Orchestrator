"""Push an approved agent branch and open a GitHub pull request."""

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from multi_agent_system.langgraph_workflow import AgentState

COMMAND_TIMEOUT_SECONDS = 60
GITHUB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _run_command(
    arguments: list[str],
    *,
    repository: Path | None = None,
    allow_failure: bool = False,
    github_command: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded command directly, without a shell."""
    environment = os.environ.copy()
    if github_command:
        # A GITHUB_TOKEN loaded from .env would override `gh auth login`.
        # This project uses the GitHub CLI credential store for write access.
        environment.pop("GITHUB_TOKEN", None)

    try:
        result = subprocess.run(
            arguments,
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
            env=environment,
        )
    except FileNotFoundError as error:
        raise ValueError(f"Command is not installed: {arguments[0]}") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"Command timed out: {arguments[0]}") from error

    if result.returncode != 0 and not allow_failure:
        message = (result.stderr or result.stdout).strip()
        raise ValueError(message or f"Command failed: {arguments[0]}")
    return result


def parse_github_remote(remote_url: str) -> tuple[str, str]:
    """Extract ``(owner, repository)`` from HTTPS or SSH GitHub remotes."""
    remote_url = remote_url.strip()
    if remote_url.startswith("git@github.com:"):
        path = remote_url.removeprefix("git@github.com:")
    else:
        parsed = urlparse(remote_url)
        if parsed.hostname != "github.com":
            raise ValueError("The origin remote must point to github.com.")
        path = parsed.path.lstrip("/")

    path = path.removesuffix(".git").rstrip("/")
    parts = path.split("/")
    if len(parts) != 2 or not all(GITHUB_NAME_PATTERN.fullmatch(p) for p in parts):
        raise ValueError("The GitHub remote must contain one owner and repository.")
    return parts[0], parts[1]


def build_pr_body(state: AgentState) -> str:
    """Build a readable pull-request description from shared graph state."""
    issue_reference = (
        f"Closes #{state['issue_number']}"
        if state.get("issue_number") is not None
        else "Created from a manually supplied issue."
    )
    changed_files = state.get("changed_files", []) + state.get("test_files", [])
    file_lines = "\n".join(f"- `{path}`" for path in dict.fromkeys(changed_files))
    if not file_lines:
        file_lines = "- No file list was provided."

    return (
        "## Summary\n\n"
        f"{state.get('patch_summary', state.get('plan', 'Automated change'))}\n\n"
        "## Files changed\n\n"
        f"{file_lines}\n\n"
        "## Test result\n\n"
        f"- Status: {state.get('test_status', 'passed')}\n"
        f"- Command: `{state.get('test_command', 'not recorded')}`\n\n"
        f"{issue_reference}\n\n"
        "---\nCreated by the Multi-Agent Orchestration System after human approval."
    )


def build_pr_command(
    state: AgentState, owner: str, repository_name: str
) -> list[str]:
    """Return the argument list used to create the pull request."""
    title = state.get("issue_title", state["issue"]).splitlines()[0][:200]
    return [
        "gh",
        "pr",
        "create",
        "--repo",
        f"{owner}/{repository_name}",
        "--head",
        state["branch_name"],
        "--base",
        state.get("base_branch", "main"),
        "--title",
        title,
        "--body",
        build_pr_body(state),
    ]


def _pr_result(status: str, url: str = "") -> AgentState:
    """Return a consistent state update from this node."""
    return {
        "pr_status": status,
        "pr_url": url,
        "execution_log": ["github_pr_opener"],
    }


def _extract_pr_url(output: str, owner: str, repository_name: str) -> str:
    """Find the GitHub pull-request URL returned by the CLI."""
    prefix = f"https://github.com/{owner}/{repository_name}/pull/"
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if candidate.startswith(prefix) and candidate[len(prefix) :].isdigit():
            return candidate
    raise ValueError("GitHub CLI did not return a pull-request URL.")


def open_github_pull_request(state: AgentState) -> AgentState:
    """Verify, push, and open the previously approved local branch."""
    if not state.get("tests_passed", False):
        return _pr_result("tests_required")
    if not state.get("pull_request_approved", False):
        return _pr_result("approval_required")
    if not state.get("branch_prepared", False):
        return _pr_result("branch_required")

    repository = Path(state["repo_path"]).resolve()
    branch = state["branch_name"]

    try:
        root = Path(
            _run_command(
                ["git", "rev-parse", "--show-toplevel"], repository=repository
            ).stdout.strip()
        ).resolve()
        if root != repository:
            return _pr_result("repo_path_must_be_git_root")

        remote_url = _run_command(
            ["git", "remote", "get-url", "origin"], repository=repository
        ).stdout.strip()
        owner, repository_name = parse_github_remote(remote_url)

        expected_owner = state.get("repository_owner")
        expected_name = state.get("repository_name")
        if expected_owner and expected_name and (
            owner.lower() != expected_owner.lower()
            or repository_name.lower() != expected_name.lower()
        ):
            return _pr_result("issue_repository_mismatch")

        local_sha = _run_command(
            ["git", "rev-parse", f"refs/heads/{branch}"], repository=repository
        ).stdout.strip()
        if local_sha != state["commit_sha"]:
            return _pr_result("commit_mismatch")

        auth = _run_command(
            ["gh", "auth", "status", "--hostname", "github.com"],
            allow_failure=True,
            github_command=True,
        )
        if auth.returncode != 0:
            return _pr_result("github_auth_required")

        existing = _run_command(
            [
                "gh",
                "pr",
                "view",
                branch,
                "--repo",
                f"{owner}/{repository_name}",
                "--json",
                "url",
                "--jq",
                ".url",
            ],
            allow_failure=True,
            github_command=True,
        )
        if existing.returncode == 0 and existing.stdout.strip():
            return _pr_result(
                "already_exists",
                _extract_pr_url(existing.stdout, owner, repository_name),
            )

        _run_command(
            ["git", "push", "--set-upstream", "origin", branch],
            repository=repository,
        )
        created = _run_command(
            build_pr_command(state, owner, repository_name),
            repository=repository,
            github_command=True,
        )
        return _pr_result(
            "created", _extract_pr_url(created.stdout, owner, repository_name)
        )
    except ValueError:
        return _pr_result("pr_creation_failed")
