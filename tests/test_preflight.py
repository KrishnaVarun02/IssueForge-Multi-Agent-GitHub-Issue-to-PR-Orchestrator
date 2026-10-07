"""Tests for secret-safe workflow prerequisite checks."""

import subprocess

from multi_agent_system import preflight


def test_basic_preflight_does_not_require_docker_or_github(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(preflight, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-value")

    checks = preflight.run_preflight_checks(
        str(tmp_path), execute_tests=False, create_pr=False
    )

    assert preflight.preflight_passed(checks)
    assert [check.name for check in checks] == [
        "repository",
        "openrouter_key",
    ]
    assert "secret-value" not in repr(checks)


def test_docker_failure_stops_preflight(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(preflight, "load_dotenv", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "configured")
    monkeypatch.setattr(preflight, "_find_docker_cli", lambda: "/bin/docker")
    monkeypatch.setattr(
        preflight,
        "_run_command",
        lambda arguments, repository=None: subprocess.CompletedProcess(
            arguments, 1, "", "daemon unavailable"
        ),
    )

    checks = preflight.run_preflight_checks(
        str(tmp_path), execute_tests=True, create_pr=False
    )

    assert not preflight.preflight_passed(checks)
    assert next(
        check for check in checks if check.name == "docker_engine"
    ).message == "Docker engine is unavailable; start Rancher Desktop"
    assert next(
        check for check in checks if check.name == "sandbox_image"
    ).message == "cannot inspect image until Docker engine is running"
