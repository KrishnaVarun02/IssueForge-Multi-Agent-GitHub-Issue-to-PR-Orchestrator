"""Check local prerequisites before spending tokens or mutating Git state."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from dotenv import load_dotenv

from multi_agent_system.docker_runner import DOCKER_IMAGE

PREFLIGHT_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class PreflightCheck:
    """One named prerequisite and its secret-safe result."""

    name: str
    passed: bool
    message: str


def _run_command(
    arguments: list[str], repository: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a bounded local diagnostic command without a shell."""
    try:
        return subprocess.run(
            arguments,
            cwd=repository,
            capture_output=True,
            text=True,
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(arguments, 127, "", "")


def _find_docker_cli() -> str | None:
    """Find Docker on PATH or in Rancher Desktop's standard macOS directory."""
    executable = shutil.which("docker")
    if executable:
        return executable
    rancher_docker = Path.home() / ".rd" / "bin" / "docker"
    return str(rancher_docker) if rancher_docker.is_file() else None


def run_preflight_checks(
    repo_path: str,
    *,
    execute_tests: bool,
    create_pr: bool,
) -> list[PreflightCheck]:
    """Return every applicable prerequisite without exposing credentials."""
    load_dotenv()
    repository = Path(repo_path).expanduser().resolve()
    checks = [
        PreflightCheck(
            "repository",
            repository.is_dir(),
            "repository directory found"
            if repository.is_dir()
            else "repository directory does not exist",
        ),
        PreflightCheck(
            "openrouter_key",
            bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")),
            "LLM API key is configured"
            if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            else "OPENROUTER_API_KEY is missing",
        ),
    ]

    if execute_tests:
        docker_executable = _find_docker_cli()
        docker_installed = docker_executable is not None
        checks.append(
            PreflightCheck(
                "docker_cli",
                docker_installed,
                "Docker CLI found" if docker_installed else "Docker CLI not found",
            )
        )
        docker_ready = docker_installed and _run_command(
            [docker_executable, "info", "--format", "{{.ServerVersion}}"]
        ).returncode == 0
        checks.append(
            PreflightCheck(
                "docker_engine",
                docker_ready,
                "Docker engine is running"
                if docker_ready
                else "Docker engine is unavailable; start Rancher Desktop",
            )
        )
        image_ready = docker_ready and _run_command(
            [docker_executable, "image", "inspect", DOCKER_IMAGE]
        ).returncode == 0
        if image_ready:
            image_message = f"sandbox image {DOCKER_IMAGE} found"
        elif not docker_installed:
            image_message = "cannot inspect image until Docker CLI is available"
        elif not docker_ready:
            image_message = "cannot inspect image until Docker engine is running"
        else:
            image_message = (
                f"build {DOCKER_IMAGE} with docker/Dockerfile.sandbox"
            )
        checks.append(
            PreflightCheck(
                "sandbox_image",
                image_ready,
                image_message,
            )
        )

    if create_pr:
        git_root = _run_command(
            ["git", "rev-parse", "--show-toplevel"], repository
        )
        correct_root = (
            git_root.returncode == 0
            and Path(git_root.stdout.strip()).resolve() == repository
        )
        checks.append(
            PreflightCheck(
                "git_repository",
                correct_root,
                "repository path is the Git root"
                if correct_root
                else "repository path must be the Git root",
            )
        )
        clean = correct_root and not _run_command(
            ["git", "status", "--porcelain"], repository
        ).stdout.strip()
        checks.append(
            PreflightCheck(
                "git_worktree",
                clean,
                "Git working tree is clean"
                if clean
                else "Git working tree has uncommitted changes",
            )
        )
        gh_ready = shutil.which("gh") is not None and _run_command(
            ["gh", "auth", "status", "--hostname", "github.com"]
        ).returncode == 0
        checks.append(
            PreflightCheck(
                "github_auth",
                gh_ready,
                "GitHub CLI is authenticated"
                if gh_ready
                else "run: gh auth login",
            )
        )

    return checks


def preflight_passed(checks: list[PreflightCheck]) -> bool:
    """Return True only when every applicable prerequisite passed."""
    return all(check.passed for check in checks)
