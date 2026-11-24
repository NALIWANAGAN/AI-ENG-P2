import json
from pathlib import Path
import re
from unittest.mock import AsyncMock, MagicMock

from openai import BadRequestError, RateLimitError
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
        assert result.report_file is not None
        assert re.fullmatch(r"shop_audit-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}\.md", result.report_file)

        # Secret from config.py was redacted BEFORE being sent to the LLM on Turn 3
        turn3_tool_msgs = [m for m in captured_messages[2] if m.get("role") == "tool"]
        joined_tool_payloads = "\n".join(m["content"] for m in turn3_tool_msgs)
        assert "super_secret_pw" not in joined_tool_payloads
        assert "[REDACTED_CREDENTIALS]" in joined_tool_payloads
        assert "[REDACTED_IP]" in joined_tool_payloads

        # Final Markdown report exists inside audit_path
        report_text = (harness_dirs["reports"] / result.report_file).read_text(encoding="utf-8")
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
        assert result.report_file is not None
        assert re.fullmatch(r"budget_audit-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}\.md", result.report_file)
        assert len(captured_tool_names) == 4
        assert all(len(names) == 4 for names in captured_tool_names[:3])
        assert captured_tool_names[3] == ["create_audit_report"]
        assert (harness_dirs["reports"] / result.report_file).exists()

    async def test_happy_preflight_map_and_decoding_controls_with_parallel_fallback(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        (harness_dirs["repo"] / "src").mkdir()
        (harness_dirs["repo"] / "src" / "query_repo.py").write_text(
            "def list_orders(): return 'SELECT * FROM orders'\n",
            encoding="utf-8",
        )

        captured_kwargs: list[dict] = []

        bad_req_response = MagicMock()
        bad_req_response.status_code = 400
        bad_req_response.headers = {}
        bad_req_response.request = MagicMock()
        unsupported_parallel_err = BadRequestError(
            message="Unsupported parameter: parallel_tool_calls",
            response=bad_req_response,
            body=None,
        )

        async def fake_create(**kwargs):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                raise unsupported_parallel_err
            return _make_llm_response(content="Completed with preflight context.")

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
        assert len(captured_kwargs) == 2

        # First call passes decoding controls (temperature=0.0, parallel_tool_calls=True)
        assert captured_kwargs[0]["temperature"] == 0.0
        assert captured_kwargs[0]["parallel_tool_calls"] is True
        # Fallback retry omits parallel_tool_calls when provider rejects it
        assert "parallel_tool_calls" not in captured_kwargs[1]
        assert captured_kwargs[1]["temperature"] == 0.0

        # Pre-flight repo map & candidate schema/DB files were injected BEFORE Turn 1
        user_msg = captured_kwargs[0]["messages"][1]["content"]
        assert "Pre-flight repository map" in user_msg
        assert "Candidate schema / migration / model files" in user_msg
        assert "- schema.sql" in user_msg
        assert "Candidate database access / query files" in user_msg
        assert "- src/query_repo.py" in user_msg

    async def test_happy_phase_state_machine_enforces_write_checkpoint_and_state_header(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        captured_turns: list[tuple[list[str], list[dict]]] = []

        scripted_responses = [
            # Turn 1 (EXPLORE): read schema
            _make_llm_response(
                tool_calls=[
                    ("c1", "read_file_slice", {"request": {"path": "schema.sql", "line_start": 1, "line_end": 10}})
                ]
            ),
            # Turn 2 (EXPLORE): search symbol
            _make_llm_response(
                tool_calls=[("c2", "search_symbol", {"request": {"path": ".", "symbol": "orders"}})]
            ),
            # Turn 3 (WRITE_CHECKPOINT after 2 read turns): only create_audit_report offered
            _make_llm_response(
                tool_calls=[
                    (
                        "c3",
                        "create_audit_report",
                        {
                            "request": {
                                "title": "Phased Audit",
                                "filename": "phased_audit.md",
                                "section": "Overview",
                                "content": "Initial overview (`schema.sql:1`).",
                                "is_final": False,
                            }
                        },
                    )
                ]
            ),
            # Turn 4 (back to EXPLORE after write): all 4 tools restored, finish report
            _make_llm_response(
                tool_calls=[
                    (
                        "c4",
                        "create_audit_report",
                        {
                            "request": {
                                "section": "Table schema",
                                "content": "`orders(id, total_cents)` (`schema.sql:1-4`).",
                                "is_final": True,
                            }
                        },
                    )
                ]
            ),
        ]

        async def fake_create(**kwargs):
            tool_names = [t["function"]["name"] for t in kwargs["tools"]]
            captured_turns.append((tool_names, [dict(m) for m in kwargs["messages"]]))
            return scripted_responses.pop(0)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await run_pipeline(
            repo_path=harness_dirs["repo"],
            audit_path=harness_dirs["reports"],
            client=mock_client,
            model_name="test-model",
            max_turns=8,
            turn_delay_seconds=0,
            write_checkpoint_turns=2,
        )

        assert result.status == "completed"
        assert result.turns_used == 4
        assert result.report_file is not None
        assert re.fullmatch(r"phased_audit-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}\.md", result.report_file)

        # Toolset transitions: EXPLORE (4) -> EXPLORE (4) -> WRITE_CHECKPOINT (1) -> EXPLORE (4)
        assert len(captured_turns[0][0]) == 4
        assert len(captured_turns[1][0]) == 4
        assert captured_turns[2][0] == ["create_audit_report"]
        assert len(captured_turns[3][0]) == 4

        # Live ephemeral state headers appear only once at the end of Turn 2+ requests
        t3_last = captured_turns[2][1][-1]["content"]
        assert "[STATE] Turn 3/8 | Phase: WRITE_CHECKPOINT" in t3_last
        assert "Written: [none]" in t3_last

        t4_msgs = captured_turns[3][1]
        state_msgs_in_t4 = [m for m in t4_msgs if "[STATE]" in str(m.get("content", ""))]
        assert len(state_msgs_in_t4) == 1  # No stale [STATE] buildup across turns
        assert f"[STATE] Turn 4/8 | Phase: EXPLORE | Report: {result.report_file}" in state_msgs_in_t4[0]["content"]
        assert "Written: [Overview]" in state_msgs_in_t4[0]["content"]
        assert "Remaining: [Table schema, ERD, Query inventory, Findings, Limitations]" in state_msgs_in_t4[0]["content"]

    async def test_happy_compacts_prior_read_outputs_after_report_write(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        big_sql = "CREATE TABLE huge_orders (\n" + "".join(
            f"    col_{i} TEXT NOT NULL,\n" for i in range(1, 40)
        ) + "    id SERIAL PRIMARY KEY\n);\n"
        (harness_dirs["repo"] / "big_schema.sql").write_text(big_sql, encoding="utf-8")

        captured_messages: list[list[dict]] = []

        scripted_responses = [
            # Turn 1: Read bulky schema file
            _make_llm_response(
                tool_calls=[
                    ("c1", "read_file_slice", {"request": {"path": "big_schema.sql", "line_start": 1, "line_end": 50}})
                ]
            ),
            # Turn 2: Persist section -> triggers compaction of Turn 1's raw read output
            _make_llm_response(
                tool_calls=[
                    (
                        "c2",
                        "create_audit_report",
                        {
                            "request": {
                                "title": "Compacted Audit",
                                "filename": "compacted_audit.md",
                                "section": "Table schema",
                                "content": "Verified `huge_orders` (`big_schema.sql:1`).",
                                "is_final": False,
                            }
                        },
                    )
                ]
            ),
            # Turn 3: Finalize report
            _make_llm_response(
                tool_calls=[
                    (
                        "c3",
                        "create_audit_report",
                        {
                            "request": {
                                "section": "Limitations",
                                "content": "None.",
                                "is_final": True,
                            }
                        },
                    )
                ]
            ),
        ]

        async def fake_create(**kwargs):
            captured_messages.append([dict(m) for m in kwargs["messages"]])
            return scripted_responses.pop(0)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=fake_create)

        result = await run_pipeline(
            repo_path=harness_dirs["repo"],
            audit_path=harness_dirs["reports"],
            client=mock_client,
            model_name="test-model",
            turn_delay_seconds=0,
            compact_min_chars=200,
        )

        assert result.status == "completed"
        assert result.turns_used == 3

        # Turn 2 saw the full raw SQL payload from Turn 1
        t2_c1 = next(m for m in captured_messages[1] if m.get("tool_call_id") == "c1")
        assert "CREATE TABLE huge_orders" in t2_c1["content"]

        # Turn 3 sees the compacted summary for c1 (same tool_call_id preserved, much shorter)
        t3_c1 = next(m for m in captured_messages[2] if m.get("tool_call_id") == "c1")
        compacted_payload = json.loads(t3_c1["content"])
        assert compacted_payload["compacted"] is True
        assert compacted_payload["tool"] == "read_file_slice"
        assert compacted_payload["path"] == "big_schema.sql"
        assert compacted_payload["saved_sections"] == ["Table schema"]
        assert "CREATE TABLE huge_orders" not in t3_c1["content"]
        assert len(t3_c1["content"]) < len(t2_c1["content"]) // 3

    async def test_happy_normalizes_stringified_or_flat_request_args_and_hints_missing_path(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        captured_messages: list[list[dict]] = []

        scripted_responses = [
            # Turn 1: Double-encoded `request` JSON string + flat unwrapped args + missing file
            _make_llm_response(
                tool_calls=[
                    (
                        "c1",
                        "read_file_slice",
                        {"request": json.dumps({"path": "schema.sql", "line_start": 1, "line_end": 5})},
                    ),
                    (
                        "c2",
                        "search_symbol",
                        {"path": ".", "symbol": "orders"},
                    ),
                    (
                        "c3",
                        "read_file_slice",
                        {"request": {"path": "nonexistent.py", "line_start": 1, "line_end": 5}},
                    ),
                ]
            ),
            # Turn 2: Finish cleanly
            _make_llm_response(content="Recovered and finished."),
        ]

        async def fake_create(**kwargs):
            captured_messages.append([dict(m) for m in kwargs["messages"]])
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

        t2_tools = {m["tool_call_id"]: m["content"] for m in captured_messages[1] if m.get("role") == "tool"}
        assert "CREATE TABLE orders" in t2_tools["c1"]
        assert '"total_matches": 1' in t2_tools["c2"]
        assert "Pre-flight repository map" in t2_tools["c3"]

    async def test_happy_normalizes_negative_line_end_excludes_tests_from_candidates_and_lists_all_matching_stems(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        (harness_dirs["repo"] / "app").mkdir()
        (harness_dirs["repo"] / "src").mkdir()
        (harness_dirs["repo"] / "config").mkdir()
        (harness_dirs["repo"] / "tests").mkdir()
        (harness_dirs["repo"] / "app" / "dependencies.py").write_text("APP_DEP = 1\n", encoding="utf-8")
        (harness_dirs["repo"] / "src" / "dependencies.py").write_text("SRC_DEP = 2\n", encoding="utf-8")
        (harness_dirs["repo"] / "config" / "dependencies.yaml").write_text("dep: true\n", encoding="utf-8")
        (harness_dirs["repo"] / "tests" / "test_postgres_client.py").write_text(
            "def test_pg(): pass\n", encoding="utf-8"
        )

        captured_messages: list[list[dict]] = []

        scripted_responses = [
            # Turn 1: line_end=-1 normalized to None + missing 'dependencies.py' lists all same-stem files
            _make_llm_response(
                tool_calls=[
                    (
                        "c1",
                        "read_file_slice",
                        {"request": {"path": "schema.sql", "line_start": 1, "line_end": -1}},
                    ),
                    (
                        "c2",
                        "read_file_slice",
                        {"request": {"path": "dependencies.py", "line_start": 1, "line_end": 50}},
                    ),
                ]
            ),
            # Turn 2: Force FINAL TURN header check (max_turns=4 -> Turn 2 normal, let's finish on Turn 4)
            _make_llm_response(
                tool_calls=[
                    ("c3", "read_file_slice", {"request": {"path": "src/dependencies.py", "line_start": 1, "line_end": 10}})
                ]
            ),
            _make_llm_response(
                tool_calls=[
                    ("c4", "read_file_slice", {"request": {"path": "app/dependencies.py", "line_start": 1, "line_end": 10}})
                ]
            ),
            # Turn 4 (FINAL TURN): finalize cleanly
            _make_llm_response(content="All verified."),
        ]

        async def fake_create(**kwargs):
            captured_messages.append([dict(m) for m in kwargs["messages"]])
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

        # Pre-flight excludes tests/test_postgres_client.py from candidate priority buckets
        preflight_user_msg = captured_messages[0][1]["content"]
        candidate_section = preflight_user_msg.split("All listed files")[0]
        assert "test_postgres_client.py" not in candidate_section
        assert "tests/test_postgres_client.py" in preflight_user_msg  # still in full listing

        # Turn 2 tool responses: c1 succeeded despite line_end=-1; c2 lists ALL matching stems
        t2_tools = {m["tool_call_id"]: m["content"] for m in captured_messages[1] if m.get("role") == "tool"}
        assert "CREATE TABLE orders" in t2_tools["c1"]
        assert "app/dependencies.py" in t2_tools["c2"]
        assert "src/dependencies.py" in t2_tools["c2"]
        assert "config/dependencies.yaml" in t2_tools["c2"]

        # Turn 4 state header explicitly flags FINAL TURN
        t4_last = captured_messages[3][-1]["content"]
        assert "FINAL TURN!" in t4_last
        assert "batch multiple create_audit_report calls" in t4_last

    async def test_happy_repairs_invalid_regex_escapes_in_stringified_request(
        self, harness_dirs: dict[str, Path]
    ) -> None:
        (harness_dirs["repo"] / "queries.py").write_text(
            "def run(cur): return cur.execute('SELECT * FROM orders')\n",
            encoding="utf-8",
        )
        captured_messages: list[list[dict]] = []

        # Simulate stringified `request` containing unescaped regex backslash `execute\(`
        malformed_stringified_req = (
            '{"path": ".", "symbol": "execute\\(", "use_regex": true, "case_sensitive": false}'
        )

        scripted_responses = [
            _make_llm_response(
                tool_calls=[
                    ("c1", "search_symbol", {"request": malformed_stringified_req}),
                ]
            ),
            _make_llm_response(content="Search succeeded."),
        ]

        async def fake_create(**kwargs):
            captured_messages.append([dict(m) for m in kwargs["messages"]])
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
        t2_c1 = next(m for m in captured_messages[1] if m.get("tool_call_id") == "c1")
        assert '"total_matches": 1' in t2_c1["content"]
        assert "queries.py" in t2_c1["content"]
