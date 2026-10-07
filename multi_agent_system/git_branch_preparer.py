"""Commit approved patches on a new branch using an isolated Git worktree."""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.sandbox_runner import validate_unified_diff

GIT_TIMEOUT_SECONDS = 30
MAX_BRANCH_SLUG_CHARS = 40


def _run_git(
    repository: Path, arguments: list[str], *, allow_failure: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run one bounded Git command without invoking a shell."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("Git command timed out.") from error

    if result.returncode != 0 and not allow_failure:
        raise ValueError((result.stderr or result.stdout).strip())
    return result


def make_branch_name(state: AgentState) -> str:
    """Create a predictable, Git-safe branch name from issue metadata."""
    title = state.get("issue_title", state["issue"]).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", title).strip("-")
    slug = slug[:MAX_BRANCH_SLUG_CHARS].rstrip("-") or "change"
    issue_identifier = str(state.get("issue_number", "manual"))
    return f"agent/issue-{issue_identifier}-{slug}"


def _branch_result(
    *, prepared: bool, branch: str, status: str, sha: str = ""
) -> AgentState:
    """Return consistent state updates from the branch preparer."""
    return {
        "branch_prepared": prepared,
        "branch_name": branch,
        "branch_status": status,
        "commit_sha": sha,
        "execution_log": ["git_branch_preparer"],
    }


def prepare_local_branch(state: AgentState) -> AgentState:
    """Apply approved patches to a new local branch without switching branches."""
    if not state.get("pull_request_approved", False):
        return _branch_result(
            prepared=False, branch="", status="approval_required"
        )
    if not state.get("tests_passed", False):
        return _branch_result(prepared=False, branch="", status="tests_required")

    repository = Path(state["repo_path"]).resolve()
    branch = make_branch_name(state)

    try:
        top_level = Path(
            _run_git(repository, ["rev-parse", "--show-toplevel"]).stdout.strip()
        ).resolve()
    except ValueError:
        return _branch_result(
            prepared=False, branch=branch, status="not_a_git_repository"
        )

    if top_level != repository:
        return _branch_result(
            prepared=False, branch=branch, status="repo_path_must_be_git_root"
        )

    if _run_git(repository, ["status", "--porcelain"]).stdout.strip():
        return _branch_result(
            prepared=False, branch=branch, status="dirty_repository"
        )

    if _run_git(
        repository,
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        allow_failure=True,
    ).returncode == 0:
        return _branch_result(
            prepared=False, branch=branch, status="branch_already_exists"
        )

    try:
        validate_unified_diff(state["patch"], set(state["changed_files"]))
        validate_unified_diff(state["test_patch"], set(state["test_files"]))
    except ValueError:
        return _branch_result(
            prepared=False, branch=branch, status="patch_rejected"
        )

    worktree_added = False
    prepared = False

    with tempfile.TemporaryDirectory(prefix="multi-agent-branch-") as temp_dir:
        worktree = Path(temp_dir) / "worktree"
        patch_files = [
            Path(temp_dir) / "approved-code.patch",
            Path(temp_dir) / "approved-tests.patch",
        ]
        patch_files[0].write_text(state["patch"], encoding="utf-8")
        patch_files[1].write_text(state["test_patch"], encoding="utf-8")

        try:
            _run_git(
                repository,
                ["worktree", "add", "-b", branch, str(worktree), "HEAD"],
            )
            worktree_added = True
            for patch_file in patch_files:
                _run_git(
                    worktree,
                    ["apply", "--check", "--recount", str(patch_file)],
                )
                _run_git(
                    worktree, ["apply", "--recount", str(patch_file)]
                )

            approved_paths = list(
                dict.fromkeys(state["changed_files"] + state["test_files"])
            )
            _run_git(worktree, ["add", "--", *approved_paths])

            commit_title = state.get("issue_title", state["issue"]).splitlines()[0]
            commit_message = f"fix: {commit_title}"[:72]
            _run_git(
                worktree,
                [
                    "-c",
                    "user.name=Multi-Agent Orchestrator",
                    "-c",
                    "user.email=multi-agent@localhost",
                    "commit",
                    "-m",
                    commit_message,
                ],
            )
            sha = _run_git(worktree, ["rev-parse", "HEAD"]).stdout.strip()
            prepared = True
        except ValueError:
            return _branch_result(
                prepared=False, branch=branch, status="branch_preparation_failed"
            )
        finally:
            if worktree_added:
                _run_git(
                    repository,
                    ["worktree", "remove", "--force", str(worktree)],
                    allow_failure=True,
                )
            if not prepared:
                _run_git(
                    repository,
                    ["branch", "--delete", "--force", branch],
                    allow_failure=True,
                )

    return _branch_result(
        prepared=True, branch=branch, status="prepared", sha=sha
    )


def route_after_branch(
    state: AgentState,
) -> Literal["pr_opener", "end"]:
    """Continue only when a local branch and commit were created."""
    return "pr_opener" if state["branch_prepared"] else "end"
