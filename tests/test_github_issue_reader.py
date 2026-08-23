"""Tests for read-only GitHub issue loading without network requests."""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from multi_agent_system.github_issue_reader import (
    github_issue_reader,
    parse_github_issue_url,
)


def test_parse_github_issue_url() -> None:
    result = parse_github_issue_url(
        "https://github.com/example/project/issues/42"
    )
    assert result == ("example", "project", 42)


@pytest.mark.parametrize(
    "issue_url",
    [
        "http://github.com/example/project/issues/1",
        "https://example.com/example/project/issues/1",
        "https://github.com/example/project/pull/1",
        "https://github.com/example/project/issues/not-a-number",
    ],
)
def test_invalid_issue_urls_are_rejected(issue_url: str) -> None:
    with pytest.raises(ValueError):
        parse_github_issue_url(issue_url)


def test_issue_reader_validates_mocked_api_response() -> None:
    payload = {
        "number": 7,
        "title": "Checkout fails after changing currency",
        "body": "Recalculate the discount after a currency change.",
        "html_url": "https://github.com/example/project/issues/7",
        "labels": [{"name": "bug"}, {"name": "checkout"}],
    }
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response

    with patch(
        "multi_agent_system.github_issue_reader.urlopen",
        return_value=response,
    ):
        result = github_issue_reader(
            {"issue_url": payload["html_url"], "execution_log": []}
        )

    assert result["issue_title"] == payload["title"]
    assert result["issue_labels"] == ["bug", "checkout"]
    assert "Recalculate the discount" in result["issue"]
    assert result["execution_log"] == ["github_issue_reader"]


def test_bad_token_falls_back_for_public_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "expired-token")
    payload = {
        "number": 1,
        "title": "Public issue",
        "body": "Readable without authentication.",
        "html_url": "https://github.com/example/project/issues/1",
        "labels": [],
    }
    unauthorized = HTTPError(
        url="https://api.github.com/example",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=BytesIO(b'{"message":"Bad credentials"}'),
    )
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response

    with patch(
        "multi_agent_system.github_issue_reader.urlopen",
        side_effect=[unauthorized, response],
    ):
        result = github_issue_reader(
            {"issue_url": payload["html_url"], "execution_log": []}
        )

    assert result["github_authentication"] == "anonymous_fallback"
    assert result["issue_title"] == "Public issue"
