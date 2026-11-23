from typing import Any, Literal

from pydantic import BaseModel, Field

TurnAction = Literal["continue", "warn", "stop"]


class ToolCallRecord(BaseModel):
    """Deterministic record of a single tool invocation captured by the harness."""

    tool_name: str = Field(min_length=1, description="Name of the MCP tool invoked.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed JSON arguments passed to the tool.",
    )
    result: Any = Field(
        default=None,
        description="Structured output or error payload returned by the tool.",
    )
    is_error: bool = Field(
        default=False,
        description="True if the tool call raised an error or failed validation.",
    )


class TurnDecision(BaseModel):
    """Guardrail verdict returned to the harness after evaluating a turn."""

    turn_number: int = Field(ge=1, description="1-indexed turn number in the current run.")
    action: TurnAction = Field(
        description="'continue' to proceed normally, 'warn' to inject corrective guidance, 'stop' to abort the loop.",
    )
    is_repetitive: bool = Field(
        default=False,
        description="True when this turn repeated an earlier turn's tool call signature.",
    )
    repeated_turns: list[int] = Field(
        default_factory=list,
        description="Prior 1-indexed turn numbers whose tool calls match this turn.",
    )
    system_message: str | None = Field(
        default=None,
        description="Actionable corrective message to inject into agent messages when action is 'warn' or 'stop'.",
    )
