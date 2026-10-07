"""Tests for Docker command construction without starting Docker."""

from pathlib import Path

from multi_agent_system.docker_runner import (
    DOCKER_IMAGE,
    build_docker_command,
    docker_is_unavailable,
    run_patch_in_docker,
)


def test_docker_command_contains_security_limits(tmp_path: Path) -> None:
    command = build_docker_command(tmp_path, "test-container")

    assert command[:2] == ["docker", "run"]
    assert "none" == command[command.index("--network") + 1]
    assert "512m" == command[command.index("--memory") + 1]
    assert "ALL" == command[command.index("--cap-drop") + 1]
    assert "no-new-privileges" in command
    assert "--read-only" in command
    assert f"{tmp_path}:/workspace:ro" in command
    assert DOCKER_IMAGE in command


def test_docker_execution_requires_approval(tmp_path: Path) -> None:
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

    result = run_patch_in_docker(state)

    assert result["tests_passed"] is False
    assert result["test_status"] == "awaiting_approval"
    assert result["sandbox_kind"] == "docker"


def test_stopped_docker_daemon_is_recognized() -> None:
    message = "failed to connect to the Docker API; is the Docker daemon running?"
    assert docker_is_unavailable(1, message)
