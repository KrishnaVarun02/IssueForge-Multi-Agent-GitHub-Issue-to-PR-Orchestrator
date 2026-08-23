"""OpenRouter-powered selection of issue-relevant repository files."""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from multi_agent_system.langgraph_workflow import AgentState
from multi_agent_system.openrouter_client import (
    OpenRouterRequestError,
    completion_usage_updates,
    create_chat_completion,
    current_usage_totals,
    get_openrouter_client_and_model,
    mark_latest_response_outcome,
)
from multi_agent_system.repository_reader import (
    MAX_SELECTED_FILES,
    read_selected_files,
)
from multi_agent_system.prompt_safety import (
    safe_system_prompt,
    untrusted_json_payload,
)

MAX_FILE_SELECTION_ATTEMPTS = 2


class FileSelection(BaseModel):
    """The exact file-selection response required from the model."""

    model_config = ConfigDict(extra="forbid")

    files: list[str] = Field(
        description="Exact repository paths most relevant to the issue",
        max_length=MAX_SELECTED_FILES,
    )
    reasoning: str = Field(description="A concise reason for this selection")


def llm_code_reader(state: AgentState) -> AgentState:
    """Ask OpenRouter to choose files, then load only validated paths."""
    client, model = get_openrouter_client_and_model()
    repository_files = state["repository_files"]
    usage_state: dict = dict(state)
    reason = "OpenRouter returned an invalid structured file selection."

    for attempt in range(MAX_FILE_SELECTION_ATTEMPTS):
        retry_instruction = ""
        if attempt > 0:
            retry_instruction = (
                " Your previous response was invalid. Return a complete JSON "
                "object matching the schema and select at least one exact path."
            )
        try:
            completion = create_chat_completion(
                client,
                workflow_state=usage_state,
                node_name="llm_code_reader",
                model=model,
                max_tokens=800,
                messages=[
                    {
                        "role": "system",
                        "content": safe_system_prompt(
                            "You select repository files for investigating a "
                            "GitHub issue. Return at most 8 exact paths from "
                            "the supplied list. Prefer implementation, tests, "
                            "and relevant config."
                            + retry_instruction
                        ),
                    },
                    {
                        "role": "user",
                        "content": untrusted_json_payload(
                            issue=state["issue"],
                            available_repository_files=repository_files,
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "file_selection",
                        "strict": True,
                        "schema": FileSelection.model_json_schema(),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )
        except OpenRouterRequestError as error:
            return {
                "llm_status": "failed",
                "llm_error": str(error),
                "failed_node": "llm_code_reader",
                **current_usage_totals(usage_state),
                "execution_log": ["llm_code_reader"],
            }

        usage_state.update(
            completion_usage_updates(
                usage_state, completion, "llm_code_reader", model
            )
        )
        try:
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("OpenRouter returned an empty file selection.")
            selection = FileSelection.model_validate_json(content)
            allowed_files = set(repository_files)
            validated_files = [
                path for path in selection.files if path in allowed_files
            ]
            if not validated_files:
                raise ValueError("OpenRouter selected no valid repository files.")
        except (ValidationError, ValueError):
            usage_state.update(
                mark_latest_response_outcome(
                    usage_state, "rejected", reason
                )
            )
            continue

        read_updates = read_selected_files(state, validated_files)
        usage_state.update(mark_latest_response_outcome(usage_state, "accepted"))
        return {
            **read_updates,
            "file_selection_reasoning": selection.reasoning,
            "llm_status": "ok",
            "llm_error": "",
            "failed_node": "",
            **current_usage_totals(usage_state),
            "execution_log": ["llm_code_reader"],
        }

    return {
        "llm_status": "failed",
        "llm_error": f"{reason} Two attempts were rejected.",
        "failed_node": "llm_code_reader",
        **current_usage_totals(usage_state),
        "execution_log": ["llm_code_reader"],
    }
