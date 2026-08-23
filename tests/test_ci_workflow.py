"""Static safety checks for the GitHub Actions workflow."""

from pathlib import Path


def ci_workflow() -> str:
    """Read the workflow as text without adding a YAML dependency."""
    repository = Path(__file__).resolve().parents[1]
    return (repository / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def test_ci_runs_supported_python_versions_and_all_tests() -> None:
    workflow = ci_workflow()

    assert 'python-version: ["3.12", "3.13"]' in workflow
    assert "git ls-files -z '*.py' | xargs -0 python -m py_compile" in workflow
    assert "python -m pip check" in workflow
    assert "python -m pytest -q" in workflow


def test_ci_builds_sandbox_only_after_python_tests() -> None:
    workflow = ci_workflow()

    assert "needs: python-tests" in workflow
    assert "docker/Dockerfile.sandbox" in workflow
    assert "docker run --rm multi-agent-test-sandbox:ci --version" in workflow


def test_ci_has_read_only_permissions_and_no_application_secrets() -> None:
    workflow = ci_workflow()

    assert "permissions:\n  contents: read" in workflow
    assert "OPENROUTER_API_KEY" not in workflow
    assert "GITHUB_TOKEN:" not in workflow
    assert "pull-requests: write" not in workflow
