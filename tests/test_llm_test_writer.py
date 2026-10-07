"""Tests for deterministic Test Writer path validation."""

from multi_agent_system.llm_test_writer import is_safe_test_path


def test_safe_pytest_paths_are_allowed() -> None:
    assert is_safe_test_path("tests/test_checkout.py")
    assert is_safe_test_path("src/tests/unit/test_currency.py")
    assert is_safe_test_path("test_checkout.py")


def test_unsafe_or_non_test_paths_are_rejected() -> None:
    assert not is_safe_test_path("../tests/test_secret.py")
    assert not is_safe_test_path("/tmp/test_checkout.py")
    assert not is_safe_test_path("tests/test_checkout.sh")
    assert not is_safe_test_path("src/checkout.py")
