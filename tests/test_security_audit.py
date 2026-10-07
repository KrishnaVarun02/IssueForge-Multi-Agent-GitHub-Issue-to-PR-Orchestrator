"""Tests for the executable project security audit."""

from pathlib import Path

from multi_agent_system.security_audit import run_security_audit


def test_repository_passes_every_security_invariant() -> None:
    repository = Path(__file__).resolve().parents[1]

    checks = run_security_audit(repository)

    assert len(checks) == 6
    assert all(check.passed for check in checks), [
        check.message for check in checks if not check.passed
    ]
