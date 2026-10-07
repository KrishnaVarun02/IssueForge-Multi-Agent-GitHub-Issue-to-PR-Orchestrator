"""Executable security invariants for the multi-agent workflow."""

import ast
from dataclasses import dataclass
from pathlib import Path
import subprocess

from multi_agent_system.docker_runner import build_docker_command
from multi_agent_system.git_branch_preparer import prepare_local_branch
from multi_agent_system.github_pr_opener import open_github_pull_request

COMMAND_TIMEOUT_SECONDS = 10
SENSITIVE_TRACKED_NAMES = {".env", "workflow-report.json"}
REQUIRED_DOCKER_ARGUMENTS = {
    "--network": "none",
    "--memory": "512m",
    "--cpus": "1.0",
    "--pids-limit": "128",
    "--cap-drop": "ALL",
    "--security-opt": "no-new-privileges",
}


@dataclass(frozen=True)
class SecurityCheck:
    """One named audit result with a human-readable explanation."""

    name: str
    passed: bool
    message: str


def _git(repository: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one read-only Git inspection without invoking a shell."""
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


def _secret_tracking_check(repository: Path) -> SecurityCheck:
    """Reject committed environment, checkpoint, key, and report artifacts."""
    result = _git(repository, ["ls-files"])
    if result.returncode != 0:
        return SecurityCheck("tracked_secrets", False, "Could not inspect Git files.")

    unsafe = []
    for tracked_path in result.stdout.splitlines():
        path = Path(tracked_path)
        name = path.name
        if (
            tracked_path in SENSITIVE_TRACKED_NAMES
            or tracked_path.startswith(".agent/")
            or (name.startswith(".env.") and name != ".env.example")
            or path.suffix in {".pem", ".key", ".sqlite3"}
            or (name.startswith("workflow-report") and path.suffix == ".json")
        ):
            unsafe.append(tracked_path)

    if unsafe:
        return SecurityCheck(
            "tracked_secrets",
            False,
            "Sensitive artifacts are tracked: " + ", ".join(unsafe),
        )
    return SecurityCheck(
        "tracked_secrets", True, "No sensitive local artifacts are tracked."
    )


def _ignored_local_artifacts_check(repository: Path) -> SecurityCheck:
    """Ensure representative local secret and state files remain ignored."""
    candidates = [
        ".env",
        ".agent/checkpoints.sqlite3",
        "workflow-report.json",
    ]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "--stdin"],
            input="\n".join(candidates) + "\n",
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SecurityCheck("ignored_local_artifacts", False, "Git check timed out.")
    ignored = set(result.stdout.splitlines())
    missing = [path for path in candidates if path not in ignored]
    if missing:
        return SecurityCheck(
            "ignored_local_artifacts",
            False,
            "Local artifacts are not ignored: " + ", ".join(missing),
        )
    return SecurityCheck(
        "ignored_local_artifacts",
        True,
        "Secrets, checkpoints, and reports are ignored by Git.",
    )


def _docker_restrictions_check(repository: Path) -> SecurityCheck:
    """Verify runtime isolation flags and the image's non-root user."""
    command = build_docker_command(Path("/tmp/security-audit"), "audit-container")
    missing = []
    for flag, expected_value in REQUIRED_DOCKER_ARGUMENTS.items():
        try:
            value = command[command.index(flag) + 1]
        except (ValueError, IndexError):
            value = None
        if value != expected_value:
            missing.append(f"{flag} {expected_value}")
    for flag in ("--read-only", "--rm"):
        if flag not in command:
            missing.append(flag)

    dockerfile = (repository / "docker/Dockerfile.sandbox").read_text(
        encoding="utf-8"
    )
    if "USER runner" not in dockerfile:
        missing.append("Dockerfile USER runner")
    if missing:
        return SecurityCheck(
            "docker_isolation",
            False,
            "Missing restrictions: " + ", ".join(missing),
        )
    return SecurityCheck(
        "docker_isolation",
        True,
        "Docker is non-root, offline, bounded, read-only, and capability-free.",
    )


def _shell_execution_check(repository: Path) -> SecurityCheck:
    """Use Python's parser to reject subprocess calls with shell=True."""
    unsafe_locations = []
    source_root = repository / "multi_agent_system"
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    unsafe_locations.append(f"{path.name}:{node.lineno}")
    if unsafe_locations:
        return SecurityCheck(
            "shell_execution",
            False,
            "shell=True found at " + ", ".join(unsafe_locations),
        )
    return SecurityCheck(
        "shell_execution", True, "No production subprocess uses shell=True."
    )


def _approval_gates_check() -> SecurityCheck:
    """Prove Git and PR nodes refuse state without explicit approval."""
    branch_result = prepare_local_branch({"pull_request_approved": False})
    pr_result = open_github_pull_request(
        {"tests_passed": True, "pull_request_approved": False}
    )
    passed = (
        branch_result["branch_status"] == "approval_required"
        and pr_result["pr_status"] == "approval_required"
    )
    return SecurityCheck(
        "approval_gates",
        passed,
        "Git branch and PR creation require explicit human approval."
        if passed
        else "A destructive delivery node bypassed human approval.",
    )


def _ci_permissions_check(repository: Path) -> SecurityCheck:
    """Require read-only CI permissions and prohibit application secrets."""
    workflow = (repository / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    passed = (
        "permissions:\n  contents: read" in workflow
        and "pull-requests: write" not in workflow
        and "OPENROUTER_API_KEY" not in workflow
        and "GITHUB_TOKEN:" not in workflow
    )
    return SecurityCheck(
        "ci_permissions",
        passed,
        "CI is read-only and receives no application credentials."
        if passed
        else "CI permissions or secret exposure requires review.",
    )


def run_security_audit(repository: str | Path = ".") -> list[SecurityCheck]:
    """Run every local, network-free security invariant."""
    root = Path(repository).resolve()
    return [
        _secret_tracking_check(root),
        _ignored_local_artifacts_check(root),
        _docker_restrictions_check(root),
        _shell_execution_check(root),
        _approval_gates_check(),
        _ci_permissions_check(root),
    ]


def main() -> None:
    """Print audit results and return a failing process status when unsafe."""
    checks = run_security_audit()
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.name}: {check.message}")
    if not all(check.passed for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
