"""Validate patches and optionally run them in a temporary repository copy."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Literal

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.repository_reader import (
    IGNORED_DIRECTORIES,
    IGNORED_FILENAMES,
)

TEST_TIMEOUT_SECONDS = 30
MODEL_DIFF_WRAPPERS = {
    "```",
    "```diff",
    "```patch",
    "*** Begin Patch",
    "*** End Patch",
}


def normalize_model_diff(diff: str) -> str:
    """Remove only recognized outer wrappers from a model-generated diff."""
    lines = diff.strip().splitlines()
    while lines and lines[0].strip() in MODEL_DIFF_WRAPPERS:
        lines.pop(0)
    while lines and lines[-1].strip() in MODEL_DIFF_WRAPPERS:
        lines.pop()
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def _normalized_diff_path(raw_path: str) -> str | None:
    """Convert a diff header path into a safe repository-relative path."""
    path = raw_path.split(maxsplit=1)[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]

    candidate = PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or candidate.is_absolute()
        or ".." in candidate.parts
    ):
        raise ValueError(f"Unsafe path in patch: {raw_path}")
    return str(candidate)


def validate_unified_diff(diff: str, allowed_paths: set[str]) -> None:
    """Reject malformed diff headers and changes to unapproved paths."""
    header_count = 0
    header_paths: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("--- ", "+++ ")):
            header_count += 1
            normalized = _normalized_diff_path(line[4:])
            if normalized is not None:
                header_paths.append(normalized)

    if header_count < 2 or not header_paths:
        raise ValueError("Patch does not contain valid unified-diff headers.")

    unapproved_paths = sorted(set(header_paths) - allowed_paths)
    if unapproved_paths:
        raise ValueError(
            "Patch contains unapproved paths: " + ", ".join(unapproved_paths)
        )


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    """Exclude environments, caches, editor files, secrets, and symlinks."""
    ignored: set[str] = set()
    directory_path = Path(directory)
    for name in names:
        path = directory_path / name
        if (
            name in IGNORED_DIRECTORIES
            or name in IGNORED_FILENAMES
            or name == ".env"
            or name.startswith(".env.")
            or path.is_symlink()
        ):
            ignored.add(name)
    return ignored


def prepare_patched_repository(state: AgentState, temp_dir: str) -> Path:
    """Create a validated, patched repository copy inside temp_dir."""
    validate_unified_diff(state["patch"], set(state["changed_files"]))
    validate_unified_diff(state["test_patch"], set(state["test_files"]))

    repository = Path(state["repo_path"]).resolve()
    if not repository.is_dir():
        raise ValueError(f"Repository path is not a directory: {repository}")

    sandbox = Path(temp_dir) / "repository"
    shutil.copytree(repository, sandbox, ignore=_copy_ignore)
    patch_files = [
        Path(temp_dir) / "code.patch",
        Path(temp_dir) / "tests.patch",
    ]
    patch_files[0].write_text(state["patch"], encoding="utf-8")
    patch_files[1].write_text(state["test_patch"], encoding="utf-8")

    for patch_file in patch_files:
        for arguments in (
            ["--check", "--recount", str(patch_file)],
            ["--recount", str(patch_file)],
        ):
            try:
                result = subprocess.run(
                    ["git", "apply", *arguments],
                    cwd=sandbox,
                    capture_output=True,
                    text=True,
                    timeout=TEST_TIMEOUT_SECONDS,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise ValueError("Git patch operation timed out.") from error

            if result.returncode != 0:
                raise ValueError((result.stderr or result.stdout).strip())

    return sandbox


def run_patch_in_sandbox(state: AgentState) -> AgentState:
    """Validate proposals and, when approved, run pytest in a temporary copy."""
    try:
        validate_unified_diff(state["patch"], set(state["changed_files"]))
        validate_unified_diff(state["test_patch"], set(state["test_files"]))
    except ValueError as error:
        return {
            "tests_passed": False,
            "test_status": "patch_rejected",
            "test_output": str(error),
            "sandbox_kind": "temporary_directory",
            "execution_log": ["sandbox_test_runner"],
        }

    if not state.get("execute_tests", False):
        return {
            "tests_passed": False,
            "test_status": "awaiting_approval",
            "test_output": "Patches validated. Execution requires execute_tests=True.",
            "sandbox_kind": "temporary_directory",
            "execution_log": ["sandbox_test_runner"],
        }

    with tempfile.TemporaryDirectory(prefix="multi-agent-sandbox-") as temp_dir:
        try:
            sandbox = prepare_patched_repository(state, temp_dir)
        except ValueError as error:
            return {
                "tests_passed": False,
                "test_status": "patch_rejected",
                "test_output": str(error),
                "sandbox_kind": "temporary_directory",
                "execution_log": ["sandbox_test_runner"],
            }

        try:
            test_result = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=TEST_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "tests_passed": False,
                "test_status": "timed_out",
                "test_output": (
                    f"pytest exceeded the {TEST_TIMEOUT_SECONDS}-second limit."
                ),
                "sandbox_kind": "temporary_directory",
                "execution_log": ["sandbox_test_runner"],
            }

        output = (test_result.stdout + test_result.stderr).strip()
        passed = test_result.returncode == 0
        return {
            "tests_passed": passed,
            "test_status": "passed" if passed else "failed",
            "test_output": output,
            "sandbox_kind": "temporary_directory",
            "execution_log": ["sandbox_test_runner"],
        }


def route_after_tests(state: AgentState) -> Literal["human_approval", "end"]:
    """Continue to human review only when sandbox tests passed."""
    return "human_approval" if state["tests_passed"] else "end"
