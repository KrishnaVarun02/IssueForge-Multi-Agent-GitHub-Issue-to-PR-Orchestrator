"""OpenRouter-powered Code Writer that proposes, but does not apply, a patch."""

from difflib import unified_diff
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.openrouter_client import (
    OpenRouterRequestError,
    create_chat_completion,
    completion_usage_updates,
    current_usage_totals,
    get_openrouter_client_and_model,
    mark_latest_response_outcome,
)
from multi_agent_system.repository_reader import MAX_SELECTED_FILES
from multi_agent_system.prompt_safety import (
    safe_system_prompt,
    untrusted_json_payload,
)

MAX_PATCH_ATTEMPTS = 2
MAX_PATCH_OUTPUT_TOKENS = 4_500
MAX_CODE_REPLACEMENT_CHARS = 4_000
MAX_CODE_REPLACEMENT_LINES = 80
MAX_DOCUMENT_REPLACEMENT_CHARS = 8_000
MAX_DOCUMENT_REPLACEMENT_LINES = 160
# Pydantic enforces the absolute ceiling. File-aware limits are checked below.
MAX_REPLACEMENT_CHARS = MAX_DOCUMENT_REPLACEMENT_CHARS
MAX_REPLACEMENT_LINES = MAX_DOCUMENT_REPLACEMENT_LINES
DOCUMENT_SUFFIXES = {".md", ".txt"}


class FileEdit(BaseModel):
    """One bounded line-range replacement proposed by the model."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Exact path from the selected-files list")
    start_line: int = Field(ge=1, description="First numbered line to replace")
    end_line: int = Field(ge=1, description="Last numbered line to replace")
    replacement: str = Field(
        description="Concise replacement text",
        min_length=1,
        max_length=MAX_REPLACEMENT_CHARS,
    )


class PatchOutput(BaseModel):
    """The structured edits required from the model."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="A concise summary of the proposed change")
    edits: list[FileEdit] = Field(
        description="Small bounded line-range replacements",
        min_length=1,
        max_length=MAX_SELECTED_FILES,
    )


def _patch_messages(
    state: AgentState, retry: bool, failure_reason: str = ""
) -> list[dict[str, str]]:
    """Build the prompt, making a retry explicitly shorter."""
    selected_files = state["selected_files"]
    context_sections: list[str] = []
    for path in selected_files:
        snippet = state["selected_file_contents"].get(path, "")
        numbered_lines = "\n".join(
            f"{number}: {line}"
            for number, line in enumerate(snippet.splitlines(), start=1)
        )
        context_sections.append(f"### {path}\n{numbered_lines}")
    numbered_context = "\n\n".join(context_sections)
    retry_instruction = ""
    if retry:
        retry_instruction = (
            " Your previous response was incomplete or invalid. Return a much "
            "smaller edit and ensure the JSON object is fully closed. "
            f"Validation feedback: {failure_reason}"
        )

    human_feedback = state.get("approval_feedback", "")
    revision_instruction = ""
    if human_feedback:
        revision_instruction = (
            " This is a revision. Follow the human review feedback exactly: "
            + human_feedback
        )

    return [
        {
            "role": "system",
            "content": safe_system_prompt(
                "You are a careful software engineer. Return small line-range "
                "replacements that follow the plan. start_line and end_line "
                "are inclusive and must refer to the numbered context. Change "
                "only paths in the "
                "selected-files list. For source code, keep each replacement "
                "under 80 lines and 4000 characters. For Markdown or text "
                "documentation, keep it under 160 lines and 8000 characters. "
                "Prefer concise documentation. Python "
                "will generate the unified diff, so do not return diff syntax."
                + revision_instruction
                + retry_instruction
            ),
        },
        {
            "role": "user",
            "content": untrusted_json_payload(
                issue=state["issue"],
                selected_files=selected_files,
                numbered_code_context=numbered_context,
                plan_summary=state["plan"],
                plan_steps=state.get("plan_steps", []),
                risks=state.get("plan_risks", []),
            ),
        },
    ]


def _build_patch(
    content: str, state: AgentState
) -> tuple[PatchOutput, list[str], str]:
    """Validate model edits and deterministically create a unified diff."""
    proposal = PatchOutput.model_validate_json(content)
    allowed_files = set(state["selected_files"])
    snippets = state["selected_file_contents"]
    repository = Path(state["repo_path"]).resolve()
    originals: dict[str, str] = {}
    edits_by_path: dict[str, list[FileEdit]] = {}

    for edit in proposal.edits:
        if edit.path not in allowed_files:
            raise ValueError("An edit targets a file outside the selection.")
        snippet_lines = snippets.get(edit.path, "").splitlines(keepends=True)
        if edit.start_line > edit.end_line:
            raise ValueError("start_line must not be after end_line.")
        if edit.end_line > len(snippet_lines):
            raise ValueError("An edit range exceeds the supplied context.")
        if Path(edit.path).suffix.lower() in DOCUMENT_SUFFIXES:
            character_limit = MAX_DOCUMENT_REPLACEMENT_CHARS
            line_limit = MAX_DOCUMENT_REPLACEMENT_LINES
        else:
            character_limit = MAX_CODE_REPLACEMENT_CHARS
            line_limit = MAX_CODE_REPLACEMENT_LINES
        if (
            len(edit.replacement) > character_limit
            or len(edit.replacement.splitlines()) > line_limit
        ):
            raise ValueError(
                f"Replacement for {edit.path} exceeds its limit of "
                f"{line_limit} lines or {character_limit} characters."
            )
        edits_by_path.setdefault(edit.path, []).append(edit)

    updated: dict[str, str] = {}
    for relative_path, file_edits in edits_by_path.items():
        path = (repository / relative_path).resolve()
        if repository not in path.parents:
            raise ValueError("An edit path escapes the repository.")
        original = path.read_text(encoding="utf-8")
        snippet = snippets[relative_path]
        if not original.startswith(snippet):
            raise ValueError("The selected context no longer matches the file.")

        ordered_edits = sorted(
            file_edits, key=lambda item: item.start_line, reverse=True
        )
        for earlier, later in zip(ordered_edits, ordered_edits[1:]):
            if later.end_line >= earlier.start_line:
                raise ValueError("Proposed line ranges overlap.")

        updated_lines = original.splitlines(keepends=True)
        for edit in ordered_edits:
            replacement = edit.replacement.rstrip() + "\n"
            updated_lines[edit.start_line - 1 : edit.end_line] = (
                replacement.splitlines(keepends=True)
            )
        originals[relative_path] = original
        updated[relative_path] = "".join(updated_lines)

    changed_files = list(edits_by_path)
    diff_sections: list[str] = []
    for path in changed_files:
        if originals[path] == updated[path]:
            raise ValueError("The proposed edits produced no file change.")
        diff_sections.append(
            "".join(
                unified_diff(
                    originals[path].splitlines(keepends=True),
                    updated[path].splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
        )

    generated_diff = "".join(diff_sections)
    if not generated_diff:
        raise ValueError("The proposed edits produced no patch.")
    return proposal, changed_files, generated_diff


def llm_code_writer(state: AgentState) -> AgentState:
    """Generate a validated patch proposal without changing files on disk."""
    client, model = get_openrouter_client_and_model()
    selected_files = state["selected_files"]
    last_failure = "No valid structured edit was returned."
    usage_state: dict = dict(state)

    for attempt in range(MAX_PATCH_ATTEMPTS):
        try:
            completion = create_chat_completion(
                client,
                workflow_state=usage_state,
                node_name="llm_code_writer",
                model=model,
                max_tokens=MAX_PATCH_OUTPUT_TOKENS,
                messages=_patch_messages(
                    state, retry=attempt > 0, failure_reason=last_failure
                ),
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "patch_proposal",
                        "strict": True,
                        "schema": PatchOutput.model_json_schema(),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )
        except OpenRouterRequestError as error:
            return {
                "patch_summary": "Code generation failed.",
                "changed_files": [],
                "patch": "",
                "code_generation_status": "failed",
                "code_generation_error": str(error),
                "llm_status": "failed",
                "llm_error": str(error),
                "failed_node": "llm_code_writer",
                **current_usage_totals(usage_state),
                "execution_log": ["llm_code_writer"],
            }

        usage_updates = completion_usage_updates(
            usage_state, completion, "llm_code_writer", model
        )
        usage_state.update(usage_updates)

        content = completion.choices[0].message.content
        try:
            if not content:
                raise ValueError("The model returned no content.")
            proposal, changed_files, generated_diff = _build_patch(content, state)
        except ValidationError:
            last_failure = "The response was incomplete or did not match the schema."
            usage_state.update(
                mark_latest_response_outcome(
                    usage_state, "rejected", last_failure
                )
            )
            continue
        except (OSError, UnicodeError):
            last_failure = "A selected file could not be read as text."
            usage_state.update(
                mark_latest_response_outcome(
                    usage_state, "rejected", last_failure
                )
            )
            continue
        except ValueError as error:
            last_failure = str(error)
            usage_state.update(
                mark_latest_response_outcome(
                    usage_state, "rejected", last_failure
                )
            )
            continue

        usage_state.update(mark_latest_response_outcome(usage_state, "accepted"))

        return {
            "patch_summary": proposal.summary,
            "changed_files": changed_files,
            "patch": generated_diff,
            "code_generation_status": "generated",
            "code_generation_error": "",
            "test_execution_attempts": 0,
            "llm_status": "ok",
            "llm_error": "",
            "failed_node": "",
            **current_usage_totals(usage_state),
            "execution_log": ["llm_code_writer"],
        }

    return {
        "patch_summary": "Code generation failed.",
        "changed_files": [],
        "patch": "",
        "code_generation_status": "failed",
        "code_generation_error": (
            "OpenRouter did not produce a valid edit after two attempts. "
            f"Last validation result: {last_failure}"
        ),
        **current_usage_totals(usage_state),
        "execution_log": ["llm_code_writer"],
    }


def route_after_code_writer(
    state: AgentState,
) -> Literal["test_writer", "end"]:
    """Stop cleanly when no validated code patch was generated."""
    if state.get("code_generation_status") == "failed":
        return "end"
    return "test_writer" if state.get("patch") else "end"
