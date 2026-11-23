import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from openai import RateLimitError
import pytest

import agent.harness as harness_module
from agent.harness import AuditRunResult, run_pipeline


def _make_rate_limit_error(message: str, retry_after: str | None = None) -> RateLimitError:
    """Build a RateLimitError without depending on httpx/httpx2 directly."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": retry_after} if retry_after is not None else {}
    mock_response.request = MagicMock()
    return RateLimitError(message=message, response=mock_response, body=None)


def _make_llm_response(
    content: str | None = None,
    tool_calls: list[tuple[str, str, dict]] | None = None,
) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response with optional tool calls."""
    mock_msg = MagicMock()
    mock_msg.content = content

    if tool_calls:
        mock_tcs = []
        for call_id, name, args in tool_calls:
            tc = MagicMock()
            tc.id = call_id
            tc.function.name = name
            tc.function.arguments = json.dumps(args)
            mock_tcs.append(tc)
        mock_msg.tool_calls = mock_tcs
    else:
        mock_msg.tool_calls = None

    choice = MagicMock()
    choice.message = mock_msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture
def harness_dirs(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "target_repo"
    reports = tmp_path / "audit_reports"
    outside = tmp_path / "outside"
    repo.mkdir()
    reports.mkdir()
    outside.mkdir()

    (outside / "secret.txt").write_text("do not read\n", encoding="utf-8")
    (repo / "schema.sql").write_text(
        "CREATE TABLE orders (\n    id SERIAL PRIMARY KEY,\n    total_cents INTEGER NOT NULL\n);\n",
        encoding="utf-8",
    )
    (repo / "config.py").write_text(
        "DB_URL = 'postgres://admin:super_secret_pw@10.0.0.8:5432/shop'\n",
        encoding="utf-8",
    )
    return {"repo": repo, "reports": reports}


@pytest.mark.anyio
class TestAuditHarnessIntegration:
    async def test_happy_full_pipeline_parallel_tools_redaction_and_report_finalization(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        captured_messages: list[list[dict]] = []

        scripted_responses = [
            # Turn 1: Orient
            _make_llm_response(
                tool_calls=[("c1", "list_folder_content", {"request": {"path": "."}})]
            ),
            # Turn 2: Parallel search + read (including file with secret credentials)
            _make_llm_response(
                tool_calls=[
                    ("c2", "search_symbol", {"request": {"path": ".", "symbol": "orders"}}),
                    ("c3", "read_file_slice", {"request": {"path": "config.py", "line_start": 1, "line_end": 5}}),
                ]
            ),
            # Turn 3: Finalize audit report
            _make_llm_response(
                tool_calls=[
                    (
                        "c4",
                        "create_audit_report",
                        {
                            "request": {
                                "title": "Shop Schema Audit",
                                "filename": "shop_audit.md",
                                "section": "Overview",
                                "content": "Found `orders` table (`schema.sql:1`).",
                                "is_final": True,
                            }
                        },
                    )
                ]
            ),
        ]

        async def fake_create(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return scripted_responses.pop(0)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await run_pipeline(
            repo_path=harness_dirs["repo"],
            audit_path=harness_dirs["reports"],
            client=mock_client,
            model_name="test-model",
            turn_delay_seconds=0,
        )

        assert isinstance(result, AuditRunResult)
        assert result.status == "completed"
        assert result.turns_used == 3
        assert result.report_file == "shop_audit.md"

        # Secret from config.py was redacted BEFORE being sent to the LLM on Turn 3
        turn3_tool_msgs = [m for m in captured_messages[2] if m.get("role") == "tool"]
        joined_tool_payloads = "\n".join(m["content"] for m in turn3_tool_msgs)
        assert "super_secret_pw" not in joined_tool_payloads
        assert "[REDACTED_CREDENTIALS]" in joined_tool_payloads
        assert "[REDACTED_IP]" in joined_tool_payloads

        # Final Markdown report exists inside audit_path
        report_text = (harness_dirs["reports"] / "shop_audit.md").read_text(encoding="utf-8")
        assert "# Shop Schema Audit" in report_text
        assert "## Overview" in report_text

    async def test_happy_self_corrects_after_sandbox_error(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        captured_messages: list[list[dict]] = []

        scripted_responses = [
            # Turn 1: Hallucinates escape path outside repo_path
            _make_llm_response(
                tool_calls=[("c1", "read_file_slice", {"request": {"path": "../outside/secret.txt"}})]
            ),
            # Turn 2: Self-corrects and finishes cleanly
            _make_llm_response(content="Could not access outside files; audit finished."),
        ]

        async def fake_create(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return scripted_responses.pop(0)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await run_pipeline(
            repo_path=harness_dirs["repo"],
            audit_path=harness_dirs["reports"],
            client=mock_client,
            model_name="test-model",
            turn_delay_seconds=0,
        )

        assert result.status == "completed"
        assert result.turns_used == 2
        # Error payload was fed back to the LLM on Turn 2 so it could self-correct
        tool_replies = [m for m in captured_messages[1] if m.get("role") == "tool"]
        assert "error" in tool_replies[0]["content"]
        assert "outside the allowed root directory" in tool_replies[0]["content"]

    async def test_sad_repetitive_loop_warns_then_triggers_kill_switch(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        captured_messages: list[list[dict]] = []
        same_call = [("c_rep", "list_folder_content", {"request": {"path": "."}})]

        scripted_responses = [
            _make_llm_response(tool_calls=same_call),
            _make_llm_response(tool_calls=same_call),
            _make_llm_response(tool_calls=same_call),
        ]

        async def fake_create(**kwargs):
            captured_messages.append(list(kwargs["messages"]))
            return scripted_responses.pop(0)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await run_pipeline(
            repo_path=harness_dirs["repo"],
            audit_path=harness_dirs["reports"],
            client=mock_client,
            model_name="test-model",
            turn_delay_seconds=0,
        )

        assert result.status == "stopped"
        assert result.turns_used == 3
        assert result.stop_reason is not None and "3 times" in result.stop_reason

        # Verify Turn 2's warning nudge was injected before Turn 3
        turn3_user_msgs = [m["content"] for m in captured_messages[2] if m.get("role") == "user"]
        assert any("Repetitive action detected" in text for text in turn3_user_msgs)

    async def test_happy_retries_on_429_respecting_retry_after_and_pacing(
        self, harness_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep_mock = AsyncMock()
        monkeypatch.setattr(harness_module.asyncio, "sleep", sleep_mock)

        rate_limit_header_err = _make_rate_limit_error("Rate limit reached", retry_after="15")
        rate_limit_body_err = _make_rate_limit_error("Quota exceeded. Please retry in 22.5s.")

        scripted_outcomes = [
            rate_limit_header_err,
            _make_llm_response(
                tool_calls=[("c1", "list_folder_content", {"request": {"path": "."}})]
            ),
            rate_limit_body_err,
            _make_llm_response(content="Audit completed after rate-limit retries."),
        ]

        async def fake_create(**kwargs):
            outcome = scripted_outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await run_pipeline(
            repo_path=harness_dirs["repo"],
            audit_path=harness_dirs["reports"],
            client=mock_client,
            model_name="test-model",
            max_retries=3,
            turn_delay_seconds=4.0,
            retry_base_delay=5.0,
            retry_max_delay=60.0,
        )

        assert result.status == "completed"
        assert result.turns_used == 2
        assert mock_client.chat.completions.create.await_count == 4

        slept = [call.args[0] for call in sleep_mock.await_args_list]
        assert len(slept) == 3
        assert slept[0] >= 15.0  # Turn 1 Retry-After header + jitter
        assert slept[1] == 4.0  # Proactive RPM pacing before Turn 2
        assert slept[2] >= 22.5  # Turn 2 Gemini "retry in 22.5s" hint + jitter

    async def test_sad_raises_rate_limit_error_after_max_retries_exhausted(
        self, harness_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep_mock = AsyncMock()
        monkeypatch.setattr(harness_module.asyncio, "sleep", sleep_mock)

        rate_limit_err = _make_rate_limit_error("Error code: 429")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

        with pytest.raises(RateLimitError):
            await run_pipeline(
                repo_path=harness_dirs["repo"],
                audit_path=harness_dirs["reports"],
                client=mock_client,
                model_name="test-model",
                max_retries=2,
                turn_delay_seconds=0,
                retry_base_delay=1.0,
                retry_max_delay=10.0,
            )

        assert mock_client.chat.completions.create.await_count == 3
        assert sleep_mock.await_count == 2

    async def test_happy_enforces_write_only_tools_after_soft_budget(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        captured_tool_names: list[list[str]] = []

        scripted_responses = [
            _make_llm_response(
                tool_calls=[("c1", "list_folder_content", {"request": {"path": "."}})]
            ),
            _make_llm_response(
                tool_calls=[("c2", "search_symbol", {"request": {"path": ".", "symbol": "orders"}})]
            ),
            _make_llm_response(
                tool_calls=[
                    ("c3", "read_file_slice", {"request": {"path": "schema.sql", "line_start": 1, "line_end": 10}})
                ]
            ),
            # Turn 4 (after 85% budget_turn=3): only create_audit_report should be offered
            _make_llm_response(
                tool_calls=[
                    (
                        "c4",
                        "create_audit_report",
                        {
                            "request": {
                                "title": "Budget Finalized Audit",
                                "filename": "budget_audit.md",
                                "section": "Overview",
                                "content": "Verified `orders` (`schema.sql:1`).",
                                "is_final": True,
                            }
                        },
                    )
                ]
            ),
        ]

        async def fake_create(**kwargs):
            captured_tool_names.append([t["function"]["name"] for t in kwargs["tools"]])
            return scripted_responses.pop(0)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await run_pipeline(
            repo_path=harness_dirs["repo"],
            audit_path=harness_dirs["reports"],
            client=mock_client,
            model_name="test-model",
            max_turns=4,
            turn_delay_seconds=0,
        )

        assert result.status == "completed"
        assert result.turns_used == 4
        assert result.report_file == "budget_audit.md"
        assert len(captured_tool_names) == 4
        assert all(len(names) == 4 for names in captured_tool_names[:3])
        assert captured_tool_names[3] == ["create_audit_report"]
        assert (harness_dirs["reports"] / "budget_audit.md").exists()
