"""Tests for patch validation and isolated pytest execution."""

from pathlib import Path

import pytest

from multi_agent_system.sandbox_runner import (
    normalize_model_diff,
    run_patch_in_sandbox,
    validate_unified_diff,
)


def test_known_model_wrappers_are_removed_from_diff_edges() -> None:
    wrapped = (
        "*** Begin Patch\n--- a/app.py\n+++ b/app.py\n"
        "@@ -1 +1 @@\n-old\n+new\n*** End Patch\n"
    )

    assert normalize_model_diff(wrapped) == (
        "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n"
    )


def test_wrapper_like_text_inside_diff_is_not_removed() -> None:
    diff = (
        "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"
        "-old\n+*** End Patch\n"
    )

    assert normalize_model_diff(diff) == diff


def test_unapproved_patch_path_is_rejected() -> None:
    diff = "--- a/app.py\n+++ b/secret.py\n@@ -1 +1 @@\n-old\n+new\n"
    with pytest.raises(ValueError, match="unapproved"):
        validate_unified_diff(diff, {"app.py"})


def test_execution_requires_explicit_approval(tmp_path: Path) -> None:
    state = {
        "repo_path": str(tmp_path),
        "changed_files": ["app.py"],
        "patch": "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
        "test_files": ["tests/test_app.py"],
        "test_patch": (
            "--- /dev/null\n+++ b/tests/test_app.py\n"
            "@@ -0,0 +1 @@\n+assert True\n"
        ),
        "execute_tests": False,
        "execution_log": [],
    }

    result = run_patch_in_sandbox(state)

    assert result["tests_passed"] is False
    assert result["test_status"] == "awaiting_approval"


def test_patch_and_tests_run_only_in_temporary_copy(tmp_path: Path) -> None:
    original = 'def greeting():\n    return "hello"\n'
    (tmp_path / "app.py").write_text(original, encoding="utf-8")

    state = {
        "repo_path": str(tmp_path),
        "changed_files": ["app.py"],
        "patch": (
            "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n"
            " def greeting():\n-    return \"hello\"\n+    return \"hi\"\n"
        ),
        "test_files": ["tests/test_app.py"],
        "test_patch": (
            "--- /dev/null\n+++ b/tests/test_app.py\n@@ -0,0 +1,5 @@\n"
            "+from app import greeting\n+\n+\n"
            "+def test_greeting():\n+    assert greeting() == \"hi\"\n"
        ),
        "execute_tests": True,
        "execution_log": [],
    }

    result = run_patch_in_sandbox(state)

    assert result["tests_passed"] is True
    assert result["test_status"] == "passed"
    assert "1 passed" in result["test_output"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original
    assert not (tmp_path / "tests").exists()
