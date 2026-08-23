"""Keep important README instructions synchronized with the repository."""

from pathlib import Path


def repository_and_readme() -> tuple[Path, str]:
    """Return the project root and README contents."""
    repository = Path(__file__).resolve().parents[1]
    readme = (repository / "README.md").read_text(encoding="utf-8")
    return repository, readme


def test_readme_documents_required_workflows() -> None:
    _, readme = repository_and_readme()

    for section in (
        "## Workflow architecture",
        "## Installation",
        "## Configuration",
        "## Run a real issue safely",
        "## Checkpoints and resume",
        "## Testing and security",
        "## Safety model",
        "## Troubleshooting",
    ):
        assert section in readme

    assert "--preflight-only" in readme
    assert "--execute-tests" in readme
    assert "--create-pr" in readme
    assert "--list-threads" in readme
    assert "--resume-thread" in readme


def test_readme_referenced_project_files_exist() -> None:
    repository, readme = repository_and_readme()
    referenced_paths = (
        "CHANGELOG.md",
        ".env.example",
        ".github/workflows/ci.yml",
        "docker/Dockerfile.sandbox",
        "multi_agent_system/cli.py",
        "multi_agent_system/settings.py",
        "tests/test_end_to_end_workflow.py",
    )

    for relative_path in referenced_paths:
        assert relative_path in readme
        assert (repository / relative_path).exists()


def test_readme_contains_no_real_credentials() -> None:
    _, readme = repository_and_readme()

    assert "sk-or-v1-" not in readme
    assert "ghp_" not in readme
    assert "github_pat_" not in readme
