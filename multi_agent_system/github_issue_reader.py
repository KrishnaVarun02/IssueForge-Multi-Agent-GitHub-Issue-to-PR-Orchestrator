"""Read a GitHub issue through the read-only REST API endpoint."""

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from multi_agent_system.langgraph_workflow import AgentState

GITHUB_API_VERSION = "2026-03-10"
MAX_ISSUE_BODY_CHARS = 20_000
SAFE_REPOSITORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


class GitHubLabel(BaseModel):
    """The label field used from GitHub's larger label object."""

    model_config = ConfigDict(extra="ignore")

    name: str


class GitHubIssueResponse(BaseModel):
    """The subset of GitHub's issue response required by this project."""

    model_config = ConfigDict(extra="ignore")

    number: int
    title: str
    body: str | None = None
    html_url: str
    labels: list[GitHubLabel] = Field(default_factory=list)
    pull_request: dict[str, object] | None = None


def parse_github_issue_url(issue_url: str) -> tuple[str, str, int]:
    """Return owner, repository, and issue number from a safe GitHub URL."""
    parsed = urlparse(issue_url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "github.com",
        "www.github.com",
    }:
        raise ValueError("Issue URL must use https://github.com.")

    parts = parsed.path.strip("/").split("/")
    if len(parts) != 4 or parts[2] != "issues":
        raise ValueError(
            "Expected a URL like https://github.com/OWNER/REPO/issues/NUMBER."
        )

    owner, repository, _, issue_number_text = parts
    if not SAFE_REPOSITORY_PART.fullmatch(owner) or not SAFE_REPOSITORY_PART.fullmatch(
        repository
    ):
        raise ValueError("GitHub owner or repository contains unsupported characters.")

    try:
        issue_number = int(issue_number_text)
    except ValueError as error:
        raise ValueError("GitHub issue number must be an integer.") from error
    if issue_number <= 0:
        raise ValueError("GitHub issue number must be positive.")

    return owner, repository, issue_number


def _read_github_response(request: Request) -> object:
    """Execute one GitHub request and decode its JSON response."""
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_error_message(error: HTTPError) -> str:
    """Read GitHub's public error message without exposing request headers."""
    try:
        return json.loads(error.read().decode("utf-8")).get(
            "message", "Unknown GitHub API error"
        )
    except (UnicodeError, json.JSONDecodeError):
        return "Unknown GitHub API error"


def fetch_github_issue(
    issue_url: str,
) -> tuple[str, str, int, GitHubIssueResponse, str]:
    """Fetch and validate a GitHub issue without modifying GitHub."""
    owner, repository, issue_number = parse_github_issue_url(issue_url)
    load_dotenv()

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "multi-agent-orchestration-system",
    }
    token = os.getenv("GITHUB_TOKEN")
    authentication = "anonymous"
    if token:
        headers["Authorization"] = f"Bearer {token}"
        authentication = "token"

    api_url = (
        f"https://api.github.com/repos/{owner}/{repository}/issues/{issue_number}"
    )
    try:
        request = Request(api_url, headers=headers, method="GET")
        payload = _read_github_response(request)
    except HTTPError as error:
        if error.code == 401 and token:
            anonymous_headers = {
                name: value
                for name, value in headers.items()
                if name != "Authorization"
            }
            anonymous_request = Request(
                api_url, headers=anonymous_headers, method="GET"
            )
            try:
                payload = _read_github_response(anonymous_request)
                authentication = "anonymous_fallback"
            except HTTPError as fallback_error:
                fallback_message = _github_error_message(fallback_error)
                raise RuntimeError(
                    "GITHUB_TOKEN was rejected, and anonymous access failed "
                    f"with HTTP {fallback_error.code}: {fallback_message}. "
                    "Replace the token for a private repository."
                ) from fallback_error
            except URLError as fallback_error:
                raise RuntimeError(
                    f"Could not reach GitHub API: {fallback_error.reason}"
                ) from fallback_error
        else:
            message = _github_error_message(error)
            raise RuntimeError(
                f"GitHub API returned HTTP {error.code}: {message}"
            ) from error
    except URLError as error:
        raise RuntimeError(f"Could not reach GitHub API: {error.reason}") from error

    issue = GitHubIssueResponse.model_validate(payload)
    if issue.pull_request is not None:
        raise ValueError("The supplied URL points to a pull request, not an issue.")

    return owner, repository, issue_number, issue, authentication


def github_issue_reader(state: AgentState) -> AgentState:
    """Convert one GitHub issue into LangGraph state updates."""
    issue_url = state["issue_url"]
    owner, repository, issue_number, issue, authentication = fetch_github_issue(
        issue_url
    )
    body = (issue.body or "No issue description was provided.")[
        :MAX_ISSUE_BODY_CHARS
    ]
    labels = [label.name for label in issue.labels]
    label_text = ", ".join(labels) if labels else "none"

    return {
        "issue": f"{issue.title}\n\n{body}\n\nLabels: {label_text}",
        "issue_url": issue.html_url,
        "issue_number": issue_number,
        "issue_title": issue.title,
        "issue_body": body,
        "issue_labels": labels,
        "repository_owner": owner,
        "repository_name": repository,
        "github_authentication": authentication,
        "execution_log": ["github_issue_reader"],
    }


def load_issue_input(state: AgentState) -> AgentState:
    """Load a GitHub URL when present, otherwise retain manual issue text."""
    if state.get("issue_url"):
        return github_issue_reader(state)
    if state.get("issue"):
        return {"execution_log": ["manual_issue_input"]}
    raise ValueError("Provide either issue or issue_url in the initial state.")
