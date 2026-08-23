"""Tests for isolated local branch preparation."""

import subprocess
from pathlib import Path

from multi_agent_system.git_branch_preparer import prepare_local_branch


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_approved_patch_is_committed_without_switching_branch(tmp_path: Path) -> None:
    _git(tmp_path, "init", "--quiet")
    original = 'def greeting():\n    return "hello"\n'
    (tmp_path / "app.py").write_text(original, encoding="utf-8")
    _git(tmp_path, "add", "app.py")
    _git(
        tmp_path,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "Initial commit",
    )
    original_branch = _git(tmp_path, "branch", "--show-current")

    result = prepare_local_branch(
        {
            "repo_path": str(tmp_path),
            "issue": "Update greeting",
            "issue_number": 3,
            "issue_title": "Update greeting",
            "pull_request_approved": True,
            "tests_passed": True,
            "changed_files": ["app.py"],
            "patch": (
                "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n"
                " def greeting():\n-    return \"hello\"\n+    return \"hi\"\n"
            ),
            "test_files": ["tests/test_app.py"],
            "test_patch": (
                "--- /dev/null\n+++ b/tests/test_app.py\n@@ -0,0 +1 @@\n"
                "+assert True\n"
            ),
            "execution_log": [],
        }
    )

    assert result["branch_prepared"] is True
    assert result["branch_status"] == "prepared"
    assert _git(tmp_path, "branch", "--show-current") == original_branch
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original
    assert "return \"hi\"" in _git(
        tmp_path, "show", f"{result['branch_name']}:app.py"
    )
    assert _git(tmp_path, "show", f"{result['branch_name']}:tests/test_app.py")


def test_dirty_repository_is_rejected(tmp_path: Path) -> None:
    _git(tmp_path, "init", "--quiet")
    (tmp_path / "untracked.py").write_text("data", encoding="utf-8")

    result = prepare_local_branch(
        {
            "repo_path": str(tmp_path),
            "issue": "Example",
            "pull_request_approved": True,
            "tests_passed": True,
            "changed_files": ["app.py"],
            "patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
            "test_files": ["tests/test_app.py"],
            "test_patch": (
                "--- /dev/null\n+++ b/tests/test_app.py\n"
                "@@ -0,0 +1 @@\n+assert True\n"
            ),
            "execution_log": [],
        }
    )

    assert result["branch_prepared"] is False
    assert result["branch_status"] == "dirty_repository"
