import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from main import create_mcp_server
from tools.audit_report import create_audit_report_mcp


@pytest.fixture
def report_dirs(tmp_path: Path) -> dict[str, Path]:
    repo_dir = tmp_path / "repo"
    output_dir = tmp_path / "reports"
    outside_dir = tmp_path / "reports_outside"
    repo_dir.mkdir()
    output_dir.mkdir()
    outside_dir.mkdir()
    (output_dir / "README.md").write_text("# Keep intact\n", encoding="utf-8")
    return {"repo_dir": repo_dir, "output_dir": output_dir, "outside_dir": outside_dir}


def _extract_payload(call_result: Any) -> dict[str, Any]:
    assert call_result.structured_content is not None
    return dict(call_result.structured_content)


@pytest.mark.anyio
class TestAuditReportMCPIntegration:
    async def test_happy_mounted_on_main_mcp_with_clean_schemas(
        self, report_dirs: dict[str, Path]
    ) -> None:
        server = create_mcp_server(
            repo_path=report_dirs["repo_dir"],
            audit_path=report_dirs["output_dir"],
        )
        async with Client(server) as client:
            tools = await client.list_tools()

        tool_map = {t.name: t for t in tools}
        assert {"list_folder_content", "read_file_slice", "search_symbol", "create_audit_report"}.issubset(
            tool_map.keys()
        )

        report_tool = tool_map["create_audit_report"]
        assert "ctx" not in report_tool.input_schema.get("properties", {})
        req_props = report_tool.input_schema["properties"]["request"].get("properties", {})
        assert "max_report_bytes" not in req_props
        assert "output_dir" not in req_props

        assert report_tool.output_schema is not None
        assert "metadata" not in report_tool.output_schema.get("properties", {})
        assert "metadata" not in report_tool.output_schema.get("required", [])

    async def test_happy_multi_turn_writes_same_file_merges_section_redacts_and_logs(
        self, report_dirs: dict[str, Path]
    ) -> None:
        logs: list[Any] = []

        async def log_handler(message: Any) -> None:
            logs.append(message)

        server = create_mcp_server(
            repo_path=report_dirs["repo_dir"],
            audit_path=report_dirs["output_dir"],
        )
        async with Client(server, log_handler=log_handler) as client:
            p1 = _extract_payload(
                await client.call_tool(
                    "create_audit_report",
                    {
                        "request": {
                            "title": "Schema Audit",
                            "filename": "schema_audit.md",
                            "section": "Overview",
                            "content": "Connected via postgres://root:topsecret@10.0.0.1:5432/shop.",
                        }
                    },
                )
            )
            p2 = _extract_payload(
                await client.call_tool(
                    "create_audit_report",
                    {
                        "request": {
                            "section": "Findings",
                            "content": "- Found bug in `src/repo.py:12` (contact: dev@example.com).",
                        }
                    },
                )
            )
            p3 = _extract_payload(
                await client.call_tool(
                    "create_audit_report",
                    {
                        "request": {
                            "section": "Findings",
                            "content": "- Second bug in `src/repo.py:40`.",
                            "is_final": True,
                        }
                    },
                )
            )

        assert p1["file_name"] == "schema_audit.md"
        assert "metadata" not in p1
        assert p2["file_name"] == "schema_audit.md"
        assert p3["sections_written"] == ["Overview", "Findings"]
        assert p3["is_final"] is True
        assert any("Saved report section 'Overview'" in str(entry.data) for entry in logs)

        on_disk = (report_dirs["output_dir"] / "schema_audit.md").read_text(encoding="utf-8")
        assert on_disk.count("## Findings") == 1
        assert "topsecret" not in on_disk
        assert "dev@example.com" not in on_disk
        assert "[REDACTED_CREDENTIALS]" in on_disk
        assert "[REDACTED_IP]" in on_disk
        assert "[REDACTED_EMAIL]" in on_disk

    async def test_happy_parallel_section_calls_succeed_cleanly(
        self, report_dirs: dict[str, Path]
    ) -> None:
        audit_mcp = create_audit_report_mcp(output_dir=report_dirs["output_dir"])
        async with Client(audit_mcp) as client:
            await client.call_tool(
                "create_audit_report",
                {"request": {"filename": "concurrent.md", "section": "Overview", "content": "Init"}},
            )
            r1, r2 = await asyncio.gather(
                client.call_tool(
                    "create_audit_report",
                    {"request": {"section": "Table schema", "content": "- `orders` (`schema.sql:1`)"}},
                ),
                client.call_tool(
                    "create_audit_report",
                    {"request": {"section": "Query inventory", "content": "- `SELECT *` (`repo.py:3`)"}},
                ),
            )

        assert _extract_payload(r1)["file_name"] == "concurrent.md"
        assert _extract_payload(r2)["file_name"] == "concurrent.md"
        text = (report_dirs["output_dir"] / "concurrent.md").read_text(encoding="utf-8")
        assert "## Table schema" in text
        assert "## Query inventory" in text

    @pytest.mark.parametrize(
        "request_payload",
        [
            {"filename": "../escape.md", "section": "Overview", "content": "x"},
            {"filename": "evil.py", "section": "Overview", "content": "x"},
            {"filename": "README.md", "section": "Overview", "content": "x"},
            {"section": "   ", "content": "x"},
            {"section": "Overview", "content": "   "},
        ],
    )
    async def test_sad_errors_surface_as_mcp_tool_error(
        self, report_dirs: dict[str, Path], request_payload: dict[str, Any]
    ) -> None:
        audit_mcp = create_audit_report_mcp(output_dir=report_dirs["output_dir"])
        async with Client(audit_mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("create_audit_report", {"request": request_payload})

    async def test_sad_writing_after_is_final_raises_tool_error(
        self, report_dirs: dict[str, Path]
    ) -> None:
        audit_mcp = create_audit_report_mcp(output_dir=report_dirs["output_dir"])
        async with Client(audit_mcp) as client:
            await client.call_tool(
                "create_audit_report",
                {"request": {"section": "Limitations", "content": "All done.", "is_final": True}},
            )
            with pytest.raises(ToolError, match="already been finalized"):
                await client.call_tool(
                    "create_audit_report",
                    {"request": {"section": "Extra", "content": "Should fail"}},
                )
