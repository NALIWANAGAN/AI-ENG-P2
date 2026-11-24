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
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field
import tiktoken

from agent.prompts import build_system_prompt, main_goal_prompt
from clients.openai_client import client as default_client
from core.settings import settings
from guardrails.ignored_path import list_unignored_stdlib
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
    temperature: float = settings.LLM_TEMPERATURE,
    parallel_tool_calls: bool = settings.LLM_PARALLEL_TOOL_CALLS,
) -> Any:
    """Call chat completions with decoding controls and exponential backoff + jitter for 429/5xx/network errors."""
    include_parallel = bool(tools) and parallel_tool_calls is not None
    attempt = 0

    while True:
        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
        }
        if include_parallel:
            request_kwargs["parallel_tool_calls"] = parallel_tool_calls

        try:
            return await client.chat.completions.create(**request_kwargs)
        except BadRequestError as exc:
            if include_parallel and "parallel_tool_calls" in str(exc).lower():
                logger.warning(
                    "[LLM] Provider rejected parallel_tool_calls; retrying without it."
                )
                include_parallel = False
                continue
            raise
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
            attempt += 1
            await asyncio.sleep(sleep_seconds)


_TOKEN_ENCODER: Any | None = None


def _count_text_tokens(text: str) -> int:
    """Count tokens using tiktoken with an offline ceiling fallback."""
    global _TOKEN_ENCODER
    if not text:
        return 0
    if _TOKEN_ENCODER is None:
        try:
            _TOKEN_ENCODER = tiktoken.get_encoding(settings.TOKENIZER_ENCODING)
        except Exception:
            _TOKEN_ENCODER = False
    if _TOKEN_ENCODER:
        return len(_TOKEN_ENCODER.encode(text, disallowed_special=()))
    return (len(text) + 3) // 4


class TokenIOAudit(BaseModel):
    """End-of-run token I/O breakdown and repository file coverage summary."""

    prompt_tokens: int = 0
    tool_call_tokens: int = 0
    repo_read_tokens: int = 0
    model_output_tokens: int = 0
    total_tokens: int = 0
    provider_prompt_tokens: int | None = None
    provider_completion_tokens: int | None = None
    provider_total_tokens: int | None = None
    repo_file_count: int = 0
    files_read_count: int = 0
    files_read: list[str] = Field(default_factory=list)


class AuditRunResult(BaseModel):
    """Final outcome returned by the audit harness pipeline."""

    status: Literal["completed", "stopped"]
    turns_used: int
    report_file: str | None = None
    final_message: str = ""
    stop_reason: str | None = None
    token_audit: TokenIOAudit = Field(default_factory=TokenIOAudit)


def format_token_io_audit(result: AuditRunResult) -> str:
    """Format a readable terminal/log summary of token I/O consumption and file coverage."""
    audit = result.token_audit
    lines = [
        "================ TOKEN I/O & COVERAGE AUDIT ================",
        f"Status            : {result.status} ({result.turns_used} turns)",
        f"Report file       : {result.report_file or 'none'}",
        f"Prompt tokens     : {audit.prompt_tokens:,}",
        f"Tool call tokens  : {audit.tool_call_tokens:,}",
        f"Repo read tokens  : {audit.repo_read_tokens:,}",
        f"Model output      : {audit.model_output_tokens:,}",
        "------------------------------------------------------------",
        f"Total I/O tokens  : {audit.total_tokens:,}",
    ]
    if audit.provider_total_tokens is not None:
        lines.append(
            f"Provider usage    : {audit.provider_prompt_tokens or 0:,} in / "
            f"{audit.provider_completion_tokens or 0:,} out / "
            f"{audit.provider_total_tokens:,} total"
        )
    lines.append(
        f"Repo files        : {audit.repo_file_count} (excluding guardrail-ignored files)"
    )
    lines.append(f"Files read        : {audit.files_read_count} / {audit.repo_file_count}")
    for rel_path in audit.files_read:
        lines.append(f"  - {rel_path}")
    lines.append("============================================================")
    return "\n".join(lines)


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


_INVALID_JSON_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')
_FLAT_SCHEMA_TOOLS = {
    "list_folder_content",
    "read_file_slice",
    "search_symbol",
    "create_audit_report",
}


def _safe_json_loads(raw: str) -> Any:
    """Parse JSON, repairing stray regex backslash escapes (e.g. `\\(`, `\\s`) if needed."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = _INVALID_JSON_ESCAPE.sub(r"\\\\", raw)
        return json.loads(repaired)


def _normalize_tool_args(tool_name: str, parsed_args: dict[str, Any]) -> dict[str, Any]:
    """Repair common LLM tool-argument quirks: nested/stringified `request`, regex escapes, or `line_end <= 0`."""
    if tool_name not in _FLAT_SCHEMA_TOOLS:
        return parsed_args

    normalized: dict[str, Any] = dict(parsed_args)
    if "request" in normalized:
        req: Any = normalized["request"]
        for _ in range(2):
            if isinstance(req, str):
                try:
                    req = _safe_json_loads(req)
                except json.JSONDecodeError:
                    break
        if isinstance(req, dict):
            normalized = dict(req)

    if tool_name == "read_file_slice":
        line_end = normalized.get("line_end")
        if isinstance(line_end, int) and not isinstance(line_end, bool) and line_end <= 0:
            normalized["line_end"] = None

    return normalized


def _find_matching_stem_files(
    requested_path: str,
    preflight_files: list[str],
    repo_path: Path | None,
    preflight_has_more: bool,
) -> list[str]:
    """Return all existing unignored repo-relative files sharing the requested filename stem (excluding extension)."""
    target_stem = Path(requested_path.strip()).stem.lower()
    if not target_stem:
        return []

    if preflight_has_more and repo_path is not None:
        resolved_root = repo_path.resolve()
        candidate_files = sorted(
            p.relative_to(resolved_root).as_posix()
            for p in list_unignored_stdlib(resolved_root, root_dir=resolved_root)
        )
    else:
        candidate_files = preflight_files

    return [rel for rel in candidate_files if Path(rel).stem.lower() == target_stem]


async def _execute_tool_call(
    mcp_client: Client,
    tool_call: Any,
    *,
    repo_path: Path | None = None,
    preflight_files: list[str] | None = None,
    preflight_has_more: bool = False,
) -> tuple[dict[str, Any], ToolCallRecord, str | None, list[str] | None, bool]:
    """Execute one MCP tool call, redact its output, and capture a TurnTracker record."""
    tool_name = tool_call.function.name
    raw_args = tool_call.function.arguments or "{}"
    parsed_args: dict[str, Any] = {}
    report_file: str | None = None
    sections_written: list[str] | None = None
    report_finalized = False

    try:
        parsed_args = _safe_json_loads(raw_args)
        if not isinstance(parsed_args, dict):
            raise ValueError("Tool arguments must decode to a JSON object")
        parsed_args = _normalize_tool_args(tool_name, parsed_args)

        res = await mcp_client.call_tool(tool_name, parsed_args)
        payload = res.structured_content
        redacted_content = redact_text(json.dumps(payload, default=str))

        if tool_name == "create_audit_report" and isinstance(payload, dict):
            report_file = payload.get("file_name")
            raw_sections = payload.get("sections_written")
            if isinstance(raw_sections, list):
                sections_written = [str(s) for s in raw_sections]
            report_finalized = bool(payload.get("is_final"))

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=parsed_args,
            result=payload,
            is_error=False,
        )
    except Exception as exc:
        err_text = redact_text(str(exc))
        if "path does not exist" in err_text.lower():
            if isinstance(parsed_args, dict):
                req_dict = parsed_args.get("request")
                requested_path = str(
                    req_dict.get("path", "")
                    if isinstance(req_dict, dict)
                    else parsed_args.get("path", "")
                )
            else:
                requested_path = ""
            matches = (
                _find_matching_stem_files(
                    requested_path,
                    preflight_files or [],
                    repo_path,
                    preflight_has_more,
                )
                if requested_path
                else []
            )
            if matches:
                stem = Path(requested_path.strip()).stem
                err_text = (
                    f"Path does not exist: '{requested_path}'. "
                    f"Existing files matching name '{stem}': {matches}. "
                    "Pick the exact repo-relative path you need."
                )
            else:
                err_text = (
                    f"{err_text} Use an exact repo-relative path from the Pre-flight repository map."
                )
        error_payload = {"error": err_text}
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
    return tool_message, record, report_file, sections_written, report_finalized


_SCHEMA_EXTENSIONS = {".sql", ".prisma", ".graphql", ".dbml"}
_SCHEMA_KEYWORDS = {
    "migration",
    "migrations",
    "alembic",
    "schema",
    "model",
    "models",
    "entity",
    "entities",
    "ddl",
    "database",
    "seed",
}
_DB_ACCESS_KEYWORDS = {
    "repo",
    "repos",
    "repository",
    "repositories",
    "dao",
    "query",
    "queries",
    "postgres",
    "mysql",
    "sqlite",
    "mongo",
    "client",
    "service",
    "services",
}


_TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs"}


def _is_test_path(path: Path) -> bool:
    """Return True if a repo-relative path belongs to a test directory or test file."""
    if any(part.lower() in _TEST_DIR_NAMES for part in path.parts[:-1]):
        return True
    stem = path.stem.lower()
    return (
        stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith(".test")
        or stem.endswith(".spec")
    )


def _classify_preflight_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split non-test repo-relative paths into candidate schema/model files and DB-access files."""
    schema_files: list[str] = []
    db_access_files: list[str] = []

    for raw_path in paths:
        rel = raw_path.strip()
        if not rel:
            continue
        p = Path(rel)
        if _is_test_path(p):
            continue
        suffix = p.suffix.lower()
        part_tokens = [part.lower() for part in p.parts]
        stem_tokens = re.split(r"[^a-z0-9]+", p.stem.lower())
        all_tokens = set(part_tokens + stem_tokens)

        if suffix in _SCHEMA_EXTENSIONS or (all_tokens & _SCHEMA_KEYWORDS):
            schema_files.append(rel)
        elif all_tokens & _DB_ACCESS_KEYWORDS:
            db_access_files.append(rel)

    return schema_files, db_access_files


async def _build_preflight_context(mcp_client: Client) -> tuple[str, list[str], bool]:
    """Deterministically list root files and highlight candidate non-test schema/DB files before Turn 1."""
    try:
        res = await mcp_client.call_tool("list_folder_content", {"path": ".", "batch": 1})
        payload = res.structured_content
        if not isinstance(payload, dict):
            return "", [], False
    except Exception as exc:
        logger.warning("[PREFLIGHT] Skipped root listing: %s", exc)
        return "", [], False

    raw_content = str(payload.get("content") or "")
    files = [line.strip() for line in raw_content.splitlines() if line.strip()]
    schema_files, db_access_files = _classify_preflight_paths(files)

    total_files = payload.get("total_lines", len(files))
    line_read = payload.get("line_read", f"1-{len(files)}")
    has_more = bool(payload.get("has_more", False))
    next_batch = payload.get("next_batch")

    lines = [
        "Pre-flight repository map (already computed deterministically — do NOT call list_folder_content('.') again unless has_more is true):",
        f"Total files: {total_files} | Showing: {line_read} | has_more: {has_more}"
        + (f" (next_batch: {next_batch})" if has_more and next_batch else ""),
    ]

    if schema_files:
        lines.append("\nCandidate schema / migration / model files (inspect these first in parallel):")
        lines.extend(f"- {path}" for path in schema_files[:25])
    if db_access_files:
        lines.append("\nCandidate database access / query files:")
        lines.extend(f"- {path}" for path in db_access_files[:25])

    lines.append("\nAll listed files (batch 1):")
    lines.append(raw_content if raw_content else "(empty repository)")

    return redact_text("\n".join(lines)), files, has_more


_EXPECTED_REPORT_SECTIONS = (
    "Overview",
    "Table schema",
    "ERD",
    "Query inventory",
    "Findings",
    "Limitations",
)
_COMPACTABLE_TOOLS = {"read_file_slice", "search_symbol", "list_folder_content"}


def _build_turn_state_header(
    turn: int,
    max_turns: int,
    phase: Literal["EXPLORE", "WRITE_CHECKPOINT", "FINALIZE"],
    report_file: str | None,
    sections_written: list[str],
) -> str:
    """Build a compact 1-line live progress header injected ephemerally each turn."""
    written_lower = {s.strip().lower() for s in sections_written}
    remaining = [s for s in _EXPECTED_REPORT_SECTIONS if s.lower() not in written_lower]
    written_str = ", ".join(sections_written) if sections_written else "none"
    remaining_str = ", ".join(remaining) if remaining else "none"
    report_str = report_file or "none"

    turns_left_including_this = max_turns - turn + 1

    if turn == max_turns:
        action = (
            "FINAL TURN! Read tools disabled. Save ALL remaining sections in this turn "
            "(batch multiple create_audit_report calls if needed) and set is_final=true on the last call."
        )
    elif phase == "FINALIZE":
        action = (
            f"Read tools disabled ({turns_left_including_this} turn(s) left, {len(remaining)} section(s) remaining). "
            "Batch multiple create_audit_report calls in one turn if needed, and set is_final=true on your final section."
        )
    elif phase == "WRITE_CHECKPOINT":
        action = (
            "Read tools paused this turn. Call create_audit_report to save confirmed findings "
            "(keep is_final=false if more areas remain); read tools resume next turn."
        )
    else:
        action = "Batch parallel reads/searches or save confirmed sections with create_audit_report."

    return (
        f"[STATE] Turn {turn}/{max_turns} | Phase: {phase} | Report: {report_str} | "
        f"Written: [{written_str}] | Remaining: [{remaining_str}] | Next: {action}"
    )


def _with_ephemeral_state_header(
    messages: list[ChatCompletionMessageParam],
    state_header: str,
) -> list[ChatCompletionMessageParam]:
    """Attach the live state header for one LLM call without polluting persistent message history."""
    if messages and messages[-1].get("role") == "user":
        last_msg = dict(messages[-1])
        last_content = str(last_msg.get("content") or "")
        last_msg["content"] = f"{state_header}\n\n{last_content}"
        return [*messages[:-1], last_msg]  # type: ignore[list-item]
    return [*messages, {"role": "user", "content": state_header}]


def _compact_prior_read_outputs(
    messages: list[ChatCompletionMessageParam],
    tool_call_meta: dict[str, tuple[str, dict[str, Any]]],
    sections_written: list[str],
    *,
    cutoff_index: int,
    min_chars: int = settings.LLM_COMPACT_MIN_CHARS,
) -> tuple[int, int]:
    """Replace bulky prior read/search tool payloads in-place after findings are saved to Markdown."""
    compacted_count = 0
    chars_freed = 0

    for msg in messages[:cutoff_index]:
        if msg.get("role") != "tool":
            continue
        call_id = str(msg.get("tool_call_id") or "")
        tool_name, args = tool_call_meta.get(call_id, ("", {}))
        if tool_name not in _COMPACTABLE_TOOLS:
            continue

        raw_content = str(msg.get("content") or "")
        if len(raw_content) <= min_chars:
            continue

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict) and parsed.get("compacted"):
            continue

        req_args = args.get("request") if isinstance(args.get("request"), dict) else args
        summary: dict[str, Any] = {
            "compacted": True,
            "tool": tool_name,
        }
        if isinstance(req_args, dict):
            if "path" in req_args:
                summary["path"] = req_args["path"]
            if "symbol" in req_args:
                summary["symbol"] = req_args["symbol"]

        if isinstance(parsed, dict):
            for key in ("line_read", "total_lines", "total_matches", "has_more", "next_line"):
                if key in parsed and parsed[key] is not None:
                    summary[key] = parsed[key]

        summary["saved_sections"] = list(sections_written)
        summary["note"] = "Raw output compacted after report write. Re-read only if needed."

        new_content = json.dumps(summary)
        if len(new_content) < len(raw_content):
            chars_freed += len(raw_content) - len(new_content)
            msg["content"] = new_content  # type: ignore[typeddict-item]
            compacted_count += 1

    return compacted_count, chars_freed


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
    temperature: float = settings.LLM_TEMPERATURE,
    parallel_tool_calls: bool = settings.LLM_PARALLEL_TOOL_CALLS,
    write_checkpoint_turns: int = settings.LLM_WRITE_CHECKPOINT_TURNS,
    compact_min_chars: int = settings.LLM_COMPACT_MIN_CHARS,
) -> AuditRunResult:
    """Run the multi-turn async audit loop with pre-flight mapping, phase state machine, compaction, redaction, and TurnTracker."""
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    if turn_delay_seconds < 0 or retry_base_delay < 0 or retry_max_delay < 0:
        raise ValueError("delay settings must be >= 0")
    if write_checkpoint_turns < 0:
        raise ValueError("write_checkpoint_turns must be >= 0")

    effective_max_turns = max_turns if max_turns is not None else turn_counter(repo_path)
    tracker = TurnTracker(max_turns=effective_max_turns, max_repeats=3)
    budget_turn = int(0.85 * effective_max_turns)
    write_only_tools = [
        tool for tool in tools if tool["function"]["name"] == "create_audit_report"
    ]

    preflight_context, preflight_files, preflight_has_more = await _build_preflight_context(
        mcp_client
    )
    preflight_block = f"\n\n{preflight_context}" if preflight_context else ""
    resolved_repo_path = Path(repo_path).resolve()
    repo_file_count = len(
        list_unignored_stdlib(resolved_repo_path, root_dir=resolved_repo_path)
    )

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
                f"{preflight_block}"
            ),
        },
    ]

    active_report_file: str | None = None
    sections_written: list[str] = []
    turns_since_last_write = 0
    tool_call_meta: dict[str, tuple[str, dict[str, Any]]] = {}
    files_read_set: set[str] = set()

    # Token I/O accumulators across all turns
    prompt_tok = 0
    tool_call_tok = 0
    repo_read_tok = 0
    model_output_tok = 0
    provider_in_tok = 0
    provider_out_tok = 0
    provider_tot_tok = 0
    has_provider_usage = False

    def _build_result(
        *,
        status: Literal["completed", "stopped"],
        turns_used: int,
        final_message: str = "",
        stop_reason: str | None = None,
    ) -> AuditRunResult:
        sorted_files_read = sorted(files_read_set)
        audit = TokenIOAudit(
            prompt_tokens=prompt_tok,
            tool_call_tokens=tool_call_tok,
            repo_read_tokens=repo_read_tok,
            model_output_tokens=model_output_tok,
            total_tokens=prompt_tok + tool_call_tok + repo_read_tok + model_output_tok,
            provider_prompt_tokens=provider_in_tok if has_provider_usage else None,
            provider_completion_tokens=provider_out_tok if has_provider_usage else None,
            provider_total_tokens=provider_tot_tok if has_provider_usage else None,
            repo_file_count=repo_file_count,
            files_read_count=len(sorted_files_read),
            files_read=sorted_files_read,
        )
        run_result = AuditRunResult(
            status=status,
            turns_used=turns_used,
            report_file=active_report_file,
            final_message=final_message,
            stop_reason=stop_reason,
            token_audit=audit,
        )
        logger.info("\n%s", format_token_io_audit(run_result))
        return run_result

    for turn in range(1, effective_max_turns + 1):
        if turn > 1 and turn_delay_seconds > 0:
            logger.info(
                "[PACE] Waiting %.1fs before turn %d to respect RPM/TPM limits.",
                turn_delay_seconds,
                turn,
            )
            await asyncio.sleep(turn_delay_seconds)

        if effective_max_turns >= 4 and turn > budget_turn:
            phase: Literal["EXPLORE", "WRITE_CHECKPOINT", "FINALIZE"] = "FINALIZE"
        elif (
            write_checkpoint_turns > 0
            and turns_since_last_write >= write_checkpoint_turns
            and bool(write_only_tools)
        ):
            phase = "WRITE_CHECKPOINT"
        else:
            phase = "EXPLORE"

        if phase in {"FINALIZE", "WRITE_CHECKPOINT"} and write_only_tools:
            active_tools = write_only_tools
            logger.info(
                "[PHASE %s] Turn %d/%d: restricting tools to create_audit_report only.",
                phase,
                turn,
                effective_max_turns,
            )
        else:
            active_tools = tools

        if turn > 1:
            state_header = _build_turn_state_header(
                turn=turn,
                max_turns=effective_max_turns,
                phase=phase,
                report_file=active_report_file,
                sections_written=sections_written,
            )
            request_messages = _with_ephemeral_state_header(messages, state_header)
        else:
            request_messages = messages

        # Account input tokens sent on this turn by category
        tool_call_tok += _count_text_tokens(json.dumps(active_tools, default=str))
        for msg in request_messages:
            role = msg.get("role")
            if role in {"system", "user"}:
                prompt_tok += _count_text_tokens(str(msg.get("content") or ""))
            elif role == "tool":
                cid = str(msg.get("tool_call_id") or "")
                t_name, _ = tool_call_meta.get(cid, ("", {}))
                toks = _count_text_tokens(str(msg.get("content") or ""))
                if t_name in _COMPACTABLE_TOOLS:
                    repo_read_tok += toks
                else:
                    tool_call_tok += toks
            else:
                tool_call_tok += _count_text_tokens(json.dumps(msg, default=str))

        logger.info("[TURN %d/%d] Requesting model action.", turn, effective_max_turns)
        response = await _call_llm_with_retry(
            client=client,
            model_name=model_name,
            messages=request_messages,
            tools=active_tools,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
        )

        assistant_msg = response.choices[0].message

        # Account output tokens generated by the model on this turn
        out_parts = [str(assistant_msg.content or "")]
        for tc in (assistant_msg.tool_calls or []):
            out_parts.append(str(tc.function.name or ""))
            out_parts.append(str(tc.function.arguments or ""))
        model_output_tok += _count_text_tokens("\n".join(p for p in out_parts if p))

        usage = getattr(response, "usage", None)
        if usage is not None:
            u_in = getattr(usage, "prompt_tokens", None)
            u_out = getattr(usage, "completion_tokens", None)
            u_tot = getattr(usage, "total_tokens", None)
            if (
                isinstance(u_in, int)
                and not isinstance(u_in, bool)
                and isinstance(u_out, int)
                and not isinstance(u_out, bool)
            ):
                has_provider_usage = True
                provider_in_tok += u_in
                provider_out_tok += u_out
                provider_tot_tok += (
                    u_tot
                    if isinstance(u_tot, int) and not isinstance(u_tot, bool)
                    else (u_in + u_out)
                )

        # Stop condition: model replied with plain text (no tool calls)
        if not assistant_msg.tool_calls:
            return _build_result(
                status="completed",
                turns_used=turn,
                final_message=redact_text(assistant_msg.content or ""),
            )

        pre_turn_msg_count = len(messages)
        messages.append(format_assistant_message(assistant_msg))

        # Execute 1 or N tool calls concurrently via asyncio.gather
        executions = await asyncio.gather(
            *(
                _execute_tool_call(
                    mcp_client,
                    tc,
                    repo_path=resolved_repo_path,
                    preflight_files=preflight_files,
                    preflight_has_more=preflight_has_more,
                )
                for tc in assistant_msg.tool_calls
            )
        )

        turn_records: list[ToolCallRecord] = []
        report_finalized = False
        wrote_report_this_turn = False
        gathered_info_this_turn = False

        for tool_msg, record, report_file, updated_sections, is_final in executions:
            messages.append(tool_msg)  # type: ignore[arg-type]
            turn_records.append(record)
            tool_call_meta[str(tool_msg["tool_call_id"])] = (record.tool_name, record.arguments)

            if report_file:
                active_report_file = report_file
            if updated_sections is not None:
                sections_written = updated_sections
                wrote_report_this_turn = True
            if is_final:
                report_finalized = True
            if not record.is_error and record.tool_name != "create_audit_report":
                gathered_info_this_turn = True
            if not record.is_error and record.tool_name == "read_file_slice":
                raw_read_path = record.arguments.get("path")
                if isinstance(raw_read_path, str) and raw_read_path.strip():
                    try:
                        rel_read = (
                            (resolved_repo_path / raw_read_path)
                            .resolve()
                            .relative_to(resolved_repo_path)
                            .as_posix()
                        )
                        files_read_set.add(rel_read)
                    except ValueError:
                        files_read_set.add(raw_read_path.strip())

        # Stop condition 2: agent explicitly finalized the Markdown audit report
        if report_finalized:
            logger.info("[DONE] Audit report finalized: %s", active_report_file)
            return _build_result(
                status="completed",
                turns_used=turn,
                final_message=f"Audit report finalized: {active_report_file}",
            )

        # Update phase checkpoint counter & compact older raw reads once findings are persisted
        if wrote_report_this_turn:
            turns_since_last_write = 0
            compacted, freed = _compact_prior_read_outputs(
                messages=messages,
                tool_call_meta=tool_call_meta,
                sections_written=sections_written,
                cutoff_index=pre_turn_msg_count,
                min_chars=compact_min_chars,
            )
            if compacted > 0:
                logger.info(
                    "[COMPACT] Compacted %d prior read/search output(s), freed %d chars.",
                    compacted,
                    freed,
                )
        elif gathered_info_this_turn:
            turns_since_last_write += 1

        # Evaluate TurnTracker guardrails (repetition, consecutive errors, 85% soft cap, hard cap)
        decision = tracker.record_turn(turn_records)
        if decision.action == "warn" and decision.system_message:
            messages.append({"role": "user", "content": decision.system_message})
        elif decision.action == "stop":
            return _build_result(
                status="stopped",
                turns_used=turn,
                stop_reason=decision.system_message,
            )

    return _build_result(
        status="stopped",
        turns_used=effective_max_turns,
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
    temperature: float = settings.LLM_TEMPERATURE,
    parallel_tool_calls: bool = settings.LLM_PARALLEL_TOOL_CALLS,
    write_checkpoint_turns: int = settings.LLM_WRITE_CHECKPOINT_TURNS,
    compact_min_chars: int = settings.LLM_COMPACT_MIN_CHARS,
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
            temperature=temperature,
            parallel_tool_calls=parallel_tool_calls,
            write_checkpoint_turns=write_checkpoint_turns,
            compact_min_chars=compact_min_chars,
        )
