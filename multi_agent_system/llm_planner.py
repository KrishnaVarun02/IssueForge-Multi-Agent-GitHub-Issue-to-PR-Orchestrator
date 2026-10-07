"""An OpenRouter-powered Planner node with structured output."""

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
from multi_agent_system.prompt_safety import (
    safe_system_prompt,
    untrusted_json_payload,
)

MAX_PLAN_ATTEMPTS = 2


class PlanOutput(BaseModel):
    """The exact response structure required from the model."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="A concise implementation-plan summary")
    steps: list[str] = Field(description="Ordered implementation steps")
    risks: list[str] = Field(description="Risks and edge cases to verify")


def llm_planner(state: AgentState) -> AgentState:
    """Ask OpenRouter for a structured plan and return state updates."""
    client, model = get_openrouter_client_and_model()

    issue = state["issue"]
    code_context = state["code_context"]
    research = state.get("research", "No extra research was required.")
    usage_state: dict = dict(state)
    reason = "OpenRouter returned an invalid structured plan."

    for attempt in range(MAX_PLAN_ATTEMPTS):
        retry_instruction = ""
        if attempt > 0:
            retry_instruction = (
                " Your previous response was invalid. Return one complete JSON "
                "object that exactly matches the required schema."
            )
        try:
            completion = create_chat_completion(
                client,
                workflow_state=usage_state,
                node_name="llm_planner",
                model=model,
                max_tokens=1000,
                messages=[
                    {
                        "role": "system",
                        "content": safe_system_prompt(
                            "You are a software-engineering planner. Produce a "
                            "concise, actionable implementation plan. Do not "
                            "write code."
                            + retry_instruction
                        ),
                    },
                    {
                        "role": "user",
                        "content": untrusted_json_payload(
                            issue=issue,
                            code_context=code_context,
                            research=research,
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "implementation_plan",
                        "strict": True,
                        "schema": PlanOutput.model_json_schema(),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )
        except OpenRouterRequestError as error:
            return {
                "llm_status": "failed",
                "llm_error": str(error),
                "failed_node": "llm_planner",
                **current_usage_totals(usage_state),
                "execution_log": ["llm_planner"],
            }

        usage_state.update(
            completion_usage_updates(
                usage_state, completion, "llm_planner", model
            )
        )
        try:
            content = completion.choices[0].message.content
            if not content:
                raise ValueError("OpenRouter returned an empty plan.")
            plan = PlanOutput.model_validate_json(content)
        except (ValidationError, ValueError):
            usage_state.update(
                mark_latest_response_outcome(
                    usage_state, "rejected", reason
                )
            )
            continue

        usage_state.update(mark_latest_response_outcome(usage_state, "accepted"))
        return {
            "plan": plan.summary,
            "plan_steps": plan.steps,
            "plan_risks": plan.risks,
            "llm_status": "ok",
            "llm_error": "",
            "failed_node": "",
            **current_usage_totals(usage_state),
            "execution_log": ["llm_planner"],
        }

    return {
        "llm_status": "failed",
        "llm_error": f"{reason} Two attempts were rejected.",
        "failed_node": "llm_planner",
        **current_usage_totals(usage_state),
        "execution_log": ["llm_planner"],
    }
