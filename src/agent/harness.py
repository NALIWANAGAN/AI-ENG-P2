import asyncio
from datetime import date
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Literal

from fastmcp import Client, FastMCP
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel

from agent.prompts import build_system_prompt, main_goal_prompt
from clients.openai_client import client as default_client
from core.settings import settings
from guardrails.redact import redact_text
from guardrails.track_turns import TurnTracker
from guardrails.turn_counter import turn_counter
from schema.guardrail import ToolCallRecord
from tools.audit_report import create_audit_report_mcp
from tools.read_path import create_read_path_mcp

logger = logging.getLogger(__name__)

_RETRY_IN_PATTERN = re.compile(
    r"(?:retry\s+(?:in|after)|retryDelay[\"']?\s*[:=]\s*[\"']?)\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|seconds?)?",
    re.IGNORECASE,
)
_RETRYABLE_ERRORS = (
    RateLimitError,
    InternalServerError,
    APITimeoutError,
    APIConnectionError,
)


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    """Parse server-suggested retry delay from HTTP Retry-After header or Gemini error text."""
    response = getattr(exc, "response", None)
    if response is not None:
        header_val = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if header_val is not None:
            try:
                return max(0.0, float(header_val))
            except ValueError:
                pass

    match = _RETRY_IN_PATTERN.search(str(exc))
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        return value / 1000.0 if unit == "ms" else value

    return None


async def _call_llm_with_retry(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam],
    max_retries: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> Any:
    """Call chat completions with exponential backoff + jitter for 429/5xx/network errors."""
    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except _RETRYABLE_ERRORS as exc:
            if attempt >= max_retries:
                logger.error(
                    "[LLM] Exhausted %d retries after %s.",
                    max_retries,
                    type(exc).__name__,
                )
                raise

            backoff = min(retry_base_delay * (2**attempt), retry_max_delay)
            server_hint = _extract_retry_after_seconds(exc)
            delay = max(backoff, server_hint) if server_hint is not None else backoff
            sleep_seconds = delay + random.uniform(0.0, 1.5)

            logger.warning(
                "[LLM] %s on attempt %d/%d. Retrying in %.1fs...",
                type(exc).__name__,
                attempt + 1,
                max_retries + 1,
                sleep_seconds,
            )
            await asyncio.sleep(sleep_seconds)


class AuditRunResult(BaseModel):
    """Final outcome returned by the audit harness pipeline."""

    status: Literal["completed", "stopped"]
    turns_used: int
    report_file: str | None = None
    final_message: str = ""
    stop_reason: str | None = None


def sanitize_schema(node: Any) -> Any:
    """Strip JSON Schema keywords rejected by Gemini's OpenAI-compatible tool validator."""
    if isinstance(node, list):
        return [sanitize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"additionalProperties", "title", "format"}:
            continue
        cleaned[key] = sanitize_schema(value)

    # Collapse `anyOf: [{"type": "string"}, {"type": "string"}]` (from `Path | str`) -> `{"type": "string"}`
    if "anyOf" in cleaned and isinstance(cleaned["anyOf"], list):
        non_null = [item for item in cleaned["anyOf"] if item != {"type": "null"}]
        unique_types = {item.get("type") for item in non_null if isinstance(item, dict)}
        if len(unique_types) == 1 and len(non_null) == len(cleaned["anyOf"]):
            cleaned.pop("anyOf")
            cleaned.update(non_null[0])

    return cleaned


async def get_agent_tools(mcp_client: Client) -> list[ChatCompletionToolParam]:
    """Build agent tool definitions compatible with OpenAI & Gemini chat completions."""
    mcp_tools = await mcp_client.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": sanitize_schema(tool.input_schema),
            },
        }
        for tool in mcp_tools
    ]


def format_assistant_message(msg: ChatCompletionMessage) -> ChatCompletionAssistantMessageParam:
    """Serialize an OpenAI chat completion message while preserving provider fields like Gemini `thought_signature`."""
    if isinstance(msg, BaseModel):
        return msg.model_dump(exclude_none=True)  # type: ignore[return-value]

    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (msg.tool_calls or [])
        ],
    }


async def _execute_tool_call(
    mcp_client: Client,
    tool_call: Any,
) -> tuple[dict[str, Any], ToolCallRecord, str | None, bool]:
    """Execute one MCP tool call, redact its output, and capture a TurnTracker record."""
    tool_name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"
    parsed_args: dict[str, Any] = {}
    report_file: str | None = None
    report_finalized = False

    try:
        parsed_args = json.loads(raw_args)
        if not isinstance(parsed_args, dict):
            raise ValueError("Tool arguments must decode to a JSON object")

        res = await mcp_client.call_tool(tool_name, parsed_args)
        payload = res.structured_content
        redacted_content = redact_text(json.dumps(payload, default=str))

        if tool_name == "create_audit_report" and isinstance(payload, dict):
            report_file = payload.get("file_name")
            report_finalized = bool(payload.get("is_final"))

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=parsed_args,
            result=payload,
            is_error=False,
        )
    except Exception as exc:
        error_payload = {"error": redact_text(str(exc))}
        redacted_content = json.dumps(error_payload)
        logger.warning("[TOOL ERROR] %s failed: %s", tool_name, exc)
        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=parsed_args if isinstance(parsed_args, dict) else {"raw": raw_args},
            result=error_payload,
            is_error=True,
        )

    tool_message = {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": redacted_content,
    }
    return tool_message, record, report_file, report_finalized


async def repo_auditing(
    mcp_client: Client,
    repo_path: Path | str,
    user_prompt: str,
    system_prompt: str,
    tools: list[ChatCompletionToolParam],
    client: AsyncOpenAI,
    model_name: str,
    max_turns: int | None = None,
    max_retries: int = settings.LLM_MAX_RETRIES,
    turn_delay_seconds: float = settings.LLM_TURN_DELAY_SECONDS,
    retry_base_delay: float = settings.LLM_RETRY_BASE_DELAY_SECONDS,
    retry_max_delay: float = settings.LLM_RETRY_MAX_DELAY_SECONDS,
) -> AuditRunResult:
    """Run the multi-turn async audit loop with parallel tool execution, redaction, and TurnTracker."""
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if turn_delay_seconds < 0 or retry_base_delay < 0 or retry_max_delay < 0:
        raise ValueError("delay settings must be >= 0")

    effective_max_turns = max_turns if max_turns is not None else turn_counter(repo_path)
    tracker = TurnTracker(max_turns=effective_max_turns, max_repeats=3)
    budget_turn = int(0.85 * effective_max_turns)
    write_only_tools = [
        tool for tool in tools if tool["function"]["name"] == "create_audit_report"
    ]

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n"
                "The repository root is already bound to '.'.\n"
                f"Turn budget: {effective_max_turns} turns. Write sections incrementally with "
                "create_audit_report as you confirm each area; after turn "
                f"{budget_turn}, read tools will be disabled so you can finalize the report."
            ),
        },
    ]

    active_report_file: str | None = None

    for turn in range(1, effective_max_turns + 1):
        if turn > 1 and turn_delay_seconds > 0:
            logger.info(
                "[PACE] Waiting %.1fs before turn %d to respect RPM/TPM limits.",
                turn_delay_seconds,
                turn,
            )
            await asyncio.sleep(turn_delay_seconds)

        write_only_mode = effective_max_turns >= 4 and turn > budget_turn
        active_tools = write_only_tools if (write_only_mode and write_only_tools) else tools
        if write_only_mode and write_only_tools:
            logger.info(
                "[WRITE-ONLY MODE] Turn %d/%d: restricting tools to create_audit_report only.",
                turn,
                effective_max_turns,
            )

        logger.info("[TURN %d/%d] Requesting model action.", turn, effective_max_turns)
        response = await _call_llm_with_retry(
            client=client,
            model_name=model_name,
            messages=messages,
            tools=active_tools,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )

        assistant_msg = response.choices[0].message

        # Stop condition: model replied with plain text (no tool calls)
        if not assistant_msg.tool_calls:
            return AuditRunResult(
                status="completed",
                turns_used=turn,
                report_file=active_report_file,
                final_message=redact_text(assistant_msg.content or ""),
            )

        messages.append(format_assistant_message(assistant_msg))

        # Execute 1 or N tool calls concurrently via asyncio.gather
        executions = await asyncio.gather(
            *(_execute_tool_call(mcp_client, tc) for tc in assistant_msg.tool_calls)
        )

        turn_records: list[ToolCallRecord] = []
        report_finalized = False

        for tool_msg, record, report_file, is_final in executions:
            messages.append(tool_msg)  # type: ignore[arg-type]
            turn_records.append(record)
            if report_file:
                active_report_file = report_file
            if is_final:
                report_finalized = True

        # Stop condition 2: agent explicitly finalized the Markdown audit report
        if report_finalized:
            logger.info("[DONE] Audit report finalized: %s", active_report_file)
            return AuditRunResult(
                status="completed",
                turns_used=turn,
                report_file=active_report_file,
                final_message=f"Audit report finalized: {active_report_file}",
            )

        # Evaluate TurnTracker guardrails (repetition, consecutive errors, 85% soft cap, hard cap)
        decision = tracker.record_turn(turn_records)
        if decision.action == "warn" and decision.system_message:
            messages.append({"role": "user", "content": decision.system_message})
        elif decision.action == "stop":
            return AuditRunResult(
                status="stopped",
                turns_used=turn,
                report_file=active_report_file,
                stop_reason=decision.system_message,
            )

    return AuditRunResult(
        status="stopped",
        turns_used=effective_max_turns,
        report_file=active_report_file,
        stop_reason=f"Stopped: reached max_turns limit ({effective_max_turns}).",
    )


async def run_pipeline(
    repo_path: Path | str,
    audit_path: Path | str,
    client: AsyncOpenAI | None = None,
    model_name: str | None = None,
    mcp: FastMCP | None = None,
    max_turns: int | None = None,
    max_retries: int = settings.LLM_MAX_RETRIES,
    turn_delay_seconds: float = settings.LLM_TURN_DELAY_SECONDS,
    retry_base_delay: float = settings.LLM_RETRY_BASE_DELAY_SECONDS,
    retry_max_delay: float = settings.LLM_RETRY_MAX_DELAY_SECONDS,
) -> AuditRunResult:
    """Bind deterministic repo & report sandboxes, open the MCP session, and run the audit."""
    logger.info("[PIPELINE] Starting audit pipeline.")
    resolved_repo = Path(repo_path).resolve()
    resolved_audit = Path(audit_path).resolve()

    mcp_server = mcp
    if mcp_server is None:
        mcp_server = FastMCP(name="Codebase and Schema Auditor", version="0.1")
        mcp_server.mount(create_read_path_mcp(root_dir=resolved_repo))
        mcp_server.mount(create_audit_report_mcp(output_dir=resolved_audit))

    active_client = client if client is not None else default_client
    active_model = model_name if model_name is not None else settings.OLLAMA_MODEL_NAME

    async with Client(mcp_server) as mcp_client:
        today = str(date.today())
        system_prompt = build_system_prompt()
        user_prompt = main_goal_prompt(today)
        tools = await get_agent_tools(mcp_client)

        return await repo_auditing(
            mcp_client=mcp_client,
            repo_path=resolved_repo,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            client=active_client,
            model_name=active_model,
            max_turns=max_turns,
            max_retries=max_retries,
            turn_delay_seconds=turn_delay_seconds,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )
