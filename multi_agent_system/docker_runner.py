"""Run a validated patch and its pytest suite in a restricted Docker container."""

import subprocess
import tempfile
import uuid
from pathlib import Path

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.sandbox_runner import (
    TEST_TIMEOUT_SECONDS,
    prepare_patched_repository,
    validate_unified_diff,
)

DOCKER_IMAGE = "multi-agent-test-sandbox:latest"
DOCKER_TIMEOUT_SECONDS = TEST_TIMEOUT_SECONDS + 15


def build_docker_command(sandbox: Path, container_name: str) -> list[str]:
    """Build a Docker command without executing it."""
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--memory",
        "512m",
        "--cpus",
        "1.0",
        "--pids-limit",
        "128",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--volume",
        f"{sandbox}:/workspace:ro",
        "--workdir",
        "/workspace",
        "--pull",
        "never",
        DOCKER_IMAGE,
        "-q",
        "-p",
        "no:cacheprovider",
    ]


def _docker_result(
    *, passed: bool, status: str, output: str
) -> AgentState:
    """Return consistent LangGraph updates from the Docker runner."""
    return {
        "tests_passed": passed,
        "test_status": status,
        "test_output": output,
        "sandbox_kind": "docker",
        "execution_log": ["docker_test_runner"],
    }


def docker_is_unavailable(returncode: int, output: str) -> bool:
    """Recognize missing images, stopped daemons, and Docker startup errors."""
    lowered_output = output.lower()
    return returncode == 125 or any(
        message in lowered_output
        for message in (
            "cannot connect to the docker daemon",
            "failed to connect to the docker api",
            "is the docker daemon running",
            "no such image",
        )
    )


def run_patch_in_docker(state: AgentState) -> AgentState:
    """Validate proposals and optionally execute pytest inside Docker."""
    try:
        validate_unified_diff(state["patch"], set(state["changed_files"]))
        validate_unified_diff(state["test_patch"], set(state["test_files"]))
    except ValueError as error:
        return _docker_result(
            passed=False, status="patch_rejected", output=str(error)
        )

    if not state.get("execute_tests", False):
        return _docker_result(
            passed=False,
            status="awaiting_approval",
            output="Patches validated. Docker execution requires execute_tests=True.",
        )

    with tempfile.TemporaryDirectory(prefix="multi-agent-docker-") as temp_dir:
        try:
            sandbox = prepare_patched_repository(state, temp_dir)
        except ValueError as error:
            return _docker_result(
                passed=False, status="patch_rejected", output=str(error)
            )

        container_name = f"multi-agent-test-{uuid.uuid4().hex[:12]}"
        command = build_docker_command(sandbox, container_name)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=DOCKER_TIMEOUT_SECONDS,
                check=False,
            )
        except FileNotFoundError:
            return _docker_result(
                passed=False,
                status="docker_unavailable",
                output="Docker CLI is not installed or is not on PATH.",
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["docker", "rm", "--force", container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            return _docker_result(
                passed=False,
                status="timed_out",
                output=f"Docker pytest exceeded {DOCKER_TIMEOUT_SECONDS} seconds.",
            )

        output = (result.stdout + result.stderr).strip()
        if docker_is_unavailable(result.returncode, output):
            return _docker_result(
                passed=False, status="docker_unavailable", output=output
            )

        passed = result.returncode == 0
        return _docker_result(
            passed=passed,
            status="passed" if passed else "failed",
            output=output,
        )
