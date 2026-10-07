"""OpenRouter-powered Test Writer that proposes pytest tests."""

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.openrouter_client import get_openrouter_client_and_model
from multi_agent_system.repository_reader import MAX_SELECTED_FILES
from multi_agent_system.sandbox_runner import normalize_model_diff


class TestOutput(BaseModel):
    """The exact test proposal required from the model."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="A concise summary of the proposed tests")
    test_files: list[str] = Field(
        description="Relative paths of pytest files added or changed",
        max_length=MAX_SELECTED_FILES,
    )
    unified_diff: str = Field(
        description="A unified test diff without Markdown code fences"
    )
    suggested_command: str = Field(
        description="A suggested pytest command; it will not be executed"
    )


def is_safe_test_path(path: str) -> bool:
    """Accept only relative Python test paths without directory traversal."""
    candidate = PurePosixPath(path)
    if not path or "\\" in path or candidate.is_absolute():
        return False

    if ".." in candidate.parts or candidate.suffix != ".py":
        return False

    return "tests" in candidate.parts or candidate.name.startswith("test_")


def llm_test_writer(state: AgentState) -> AgentState:
    """Generate a validated pytest proposal without writing or running it."""
    client, model = get_openrouter_client_and_model()

    completion = client.chat.completions.create(
        model=model,
        max_tokens=2500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python test engineer. Propose focused pytest "
                    "tests for the supplied code patch. Return a unified diff. "
                    "Test paths must be relative, must end in .py, and must be "
                    "inside a tests directory or start with test_. Do not use "
                    "Markdown fences. Do not claim that tests were executed."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Issue:\n{state['issue']}\n\n"
                    f"Code context:\n{state['code_context']}\n\n"
                    f"Implementation plan:\n{state['plan']}\n\n"
                    f"Proposed code patch:\n{state['patch']}\n\n"
                    "Known repository files:\n"
                    + "\n".join(state.get("repository_files", []))
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "test_proposal",
                "strict": True,
                "schema": TestOutput.model_json_schema(),
            },
        },
        extra_body={"provider": {"require_parameters": True}},
    )

    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("OpenRouter did not return a test proposal.")

    proposal = TestOutput.model_validate_json(content)
    invalid_paths = [
        path for path in proposal.test_files if not is_safe_test_path(path)
    ]
    if invalid_paths:
        raise RuntimeError(
            "OpenRouter proposed unsafe test paths: " + ", ".join(invalid_paths)
        )

    normalized_diff = normalize_model_diff(proposal.unified_diff)
    if not proposal.test_files or not normalized_diff:
        raise RuntimeError("OpenRouter returned an empty test proposal.")

    return {
        "test_summary": proposal.summary,
        "test_files": proposal.test_files,
        "test_patch": normalized_diff,
        "test_command": proposal.suggested_command,
        "tests": normalized_diff,
        "execution_log": ["llm_test_writer"],
    }
