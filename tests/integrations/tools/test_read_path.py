import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from tools.read_path import create_read_path_mcp


@pytest.fixture
def sample_mcp_repo(tmp_path: Path) -> dict[str, Path]:
    """Create an isolated sample codebase + SQL schema repo with a sibling secret dir."""
    project = tmp_path / "project"
    sibling_secret = tmp_path / "project_secret"
    project.mkdir()
    sibling_secret.mkdir()

    (sibling_secret / "outside.txt").write_text("orders total_cents top_secret\n", encoding="utf-8")

    (project / ".gitignore").write_text(
        "\n".join(
            [
                "# Ignore logs except keep.log, build dirs, and root_only.txt",
                "*.log",
                "!keep.log",
                "build/",
                "/root_only.txt",
                "ignored_dir/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    (project / "ignored_dir").mkdir()
    (project / "ignored_dir" / "hidden.py").write_text("total_cents = 999\n", encoding="utf-8")
    (project / "debug.log").write_text("total_cents debug\n", encoding="utf-8")
    (project / "keep.log").write_text("kept log line\n", encoding="utf-8")
    (project / "build").write_text("regular file named build\n", encoding="utf-8")
    (project / "root_only.txt").write_text("ignored at root\n", encoding="utf-8")

    src = project / "src"
    src.mkdir()
    (src / "root_only.txt").write_text("kept inside src\n", encoding="utf-8")
    (src / "build").mkdir()
    (src / "build" / "generated.py").write_text("total_cents = 0\n", encoding="utf-8")

    (src / "schema.sql").write_text(
        "\n".join(
            [
                "CREATE TABLE orders (",
                "    id SERIAL PRIMARY KEY,",
                "    total_cents INTEGER NOT NULL,",
                "    deleted_at TIMESTAMP NULL",
                ");",
                "ALTER TABLE orders ADD COLUMN currency VARCHAR(3);",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (src / "repo.py").write_text(
        "\n".join(
            [
                "class OrderRepo:",
                "    def get_total(self, row: dict) -> int:",
                "        sql = 'SELECT id, total_cents FROM orders WHERE valid = true'",
                "        return row.get('total_cents', 0)",
                "",
                "class OrderModel:",
                "    total_cents: int = 0",
                "    TOTAL_CENTS_LABEL = 'cents'",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assets = project / "assets"
    assets.mkdir()
    (assets / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00total_cents")

    (project / "empty.txt").write_text("", encoding="utf-8")
    (project / "long_file.txt").write_text(
        "\n".join(f"line {i:02d}: sample content for integration pagination" for i in range(1, 51))
        + "\n",
        encoding="utf-8",
    )
    (project / "minified.js").write_text(
        "a" * 300 + " total_cents " + "b" * 4700,
        encoding="utf-8",
    )
    (project / "bad_utf8.txt").write_bytes(b"valid line\n\xff\xfe invalid utf8\n")
    (project / "empty_dir").mkdir()

    return {
        "base": tmp_path,
        "project": project,
        "sibling_secret": sibling_secret,
    }


@pytest.fixture
def read_path_mcp(sample_mcp_repo: dict[str, Path]) -> FastMCP:
    """Create a FastMCP read-path server confined to the temporary integration repo."""
    return create_read_path_mcp(root_dir=sample_mcp_repo["project"])


def _extract_payload(call_result: Any) -> dict[str, Any]:
    """Return structured JSON dictionary from a FastMCP CallToolResult."""
    assert call_result.structured_content is not None
    return dict(call_result.structured_content)


# =======================================================
# MCP Tool Registration & Schema Integration Tests
# =======================================================


@pytest.mark.anyio
class TestMCPToolDiscovery:
    async def test_happy_registers_expected_tools_without_leaking_ctx_or_metadata(
        self, read_path_mcp: FastMCP
    ) -> None:
        async with Client(read_path_mcp) as client:
            tools = await client.list_tools()

        tool_map = {t.name: t for t in tools}
        assert set(tool_map.keys()) == {
            "list_folder_content",
            "read_file_slice",
            "search_symbol",
        }

        hidden_guardrails = {
            "max_tokens",
            "items_per_batch",
            "matches_per_batch",
            "max_matches_per_file",
        }
        for name, tool in tool_map.items():
            # FastMCP injects Context internally; it must never appear in the LLM input schema
            assert "ctx" not in tool.input_schema.get("properties", {}), name
            assert "request" in tool.input_schema.get("properties", {}), name

            # Server-side guardrails must never be exposed for the LLM to override
            req_props = tool.input_schema["properties"]["request"].get("properties", {})
            assert hidden_guardrails.isdisjoint(req_props.keys()), (name, req_props.keys())

            # Output schema must never advertise internal audit `metadata`
            assert tool.output_schema is not None, name
            assert "metadata" not in tool.output_schema.get("properties", {}), name
            assert "metadata" not in tool.output_schema.get("required", []), name


# =======================================================
# list_folder_content MCP Integration Tests
# =======================================================


@pytest.mark.anyio
class TestListFolderContentMCPIntegration:
    async def test_happy_lists_unignored_files_strips_metadata_and_logs_audit(
        self, read_path_mcp: FastMCP
    ) -> None:
        logs: list[Any] = []

        async def log_handler(message: Any) -> None:
            logs.append(message)

        async with Client(read_path_mcp, log_handler=log_handler) as client:
            result = await client.call_tool("list_folder_content", {"request": {"path": "."}})

        payload = _extract_payload(result)
        assert payload["read_type"] == "directory_read"
        assert payload["batch"] == 1
        assert payload["has_more"] is False
        assert payload["next_line"] is None
        assert payload["token_size"] > 0
        assert "metadata" not in payload

        listed = (payload["content"] or "").splitlines()
        assert payload["total_line_count"] == len(listed)
        assert "build" in listed
        assert "keep.log" in listed
        assert "src/schema.sql" in listed
        assert "src/repo.py" in listed
        assert "src/root_only.txt" in listed

        # Ignored paths excluded
        assert ".git/HEAD" not in listed
        assert "debug.log" not in listed
        assert "root_only.txt" not in listed
        assert "ignored_dir/hidden.py" not in listed
        assert "src/build/generated.py" not in listed

        # Audit log emitted via ctx.info
        assert any("Listed directory:" in str(entry.data) for entry in logs)

    async def test_happy_pagination_across_batches(
        self, sample_mcp_repo: dict[str, Path]
    ) -> None:
        paged_mcp = create_read_path_mcp(root_dir=sample_mcp_repo["project"], items_per_batch=3)
        async with Client(paged_mcp) as client:
            p1 = _extract_payload(
                await client.call_tool("list_folder_content", {"request": {"path": ".", "batch": 1}})
            )
            p2 = _extract_payload(
                await client.call_tool("list_folder_content", {"request": {"path": ".", "batch": 2}})
            )

        assert p1["line_read"] == "l1-3"
        assert p1["has_more"] is True
        assert p1["next_line"] == 4
        assert p2["line_read"] == "l4-6"
        assert set(p1["content"].splitlines()).isdisjoint(p2["content"].splitlines())

    async def test_edge_empty_dir_out_of_range_batch_and_token_cap(
        self, read_path_mcp: FastMCP, sample_mcp_repo: dict[str, Path]
    ) -> None:
        async with Client(read_path_mcp) as client:
            empty = _extract_payload(
                await client.call_tool("list_folder_content", {"request": {"path": "empty_dir"}})
            )
            oob = _extract_payload(
                await client.call_tool("list_folder_content", {"request": {"path": ".", "batch": 99}})
            )

        capped_mcp = create_read_path_mcp(root_dir=sample_mcp_repo["project"], max_tokens=10)
        async with Client(capped_mcp) as client:
            capped = _extract_payload(
                await client.call_tool("list_folder_content", {"request": {"path": ".", "batch": 1}})
            )

        assert empty["content"] == ""
        assert empty["total_line_count"] == 0
        assert empty["line_read"] is None
        assert empty["has_more"] is False

        assert oob["content"] == ""
        assert oob["line_read"] is None
        assert oob["has_more"] is False

        assert capped["token_size"] <= 10
        assert capped["has_more"] is True

    @pytest.mark.parametrize(
        "request_payload",
        [
            {"path": "src/repo.py"},  # File instead of directory
            {"path": "nonexistent_dir"},  # Missing directory
            {"path": ".", "batch": 0},  # Invalid batch bound
            {"path": ".."},  # Parent escape
            {"path": "../project_secret"},  # Sibling prefix escape
        ],
    )
    async def test_sad_errors_surface_as_mcp_tool_error(
        self, read_path_mcp: FastMCP, request_payload: dict[str, Any]
    ) -> None:
        async with Client(read_path_mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("list_folder_content", {"request": request_payload})


# =======================================================
# read_file_slice MCP Integration Tests
# =======================================================


@pytest.mark.anyio
class TestReadFileSliceMCPIntegration:
    async def test_happy_slice_and_next_line_continuation_with_audit_log(
        self, read_path_mcp: FastMCP
    ) -> None:
        logs: list[Any] = []

        async def log_handler(message: Any) -> None:
            logs.append(message)

        async with Client(read_path_mcp, log_handler=log_handler) as client:
            s1 = _extract_payload(
                await client.call_tool(
                    "read_file_slice",
                    {"request": {"path": "long_file.txt", "line_start": 10, "line_end": 15}},
                )
            )
            s2 = _extract_payload(
                await client.call_tool(
                    "read_file_slice",
                    {"request": {"path": "long_file.txt", "line_start": s1["next_line"], "line_end": 20}},
                )
            )

        assert s1["read_type"] == "file_slice"
        assert s1["line_read"] == "l10-15"
        assert s1["next_line"] == 16
        assert s1["has_more"] is True
        assert "metadata" not in s1
        assert s1["content"].splitlines()[0].startswith("line 10:")

        assert s2["line_read"] == "l16-20"
        assert s2["content"].splitlines()[0].startswith("line 16:")
        assert any("Read slice:" in str(entry.data) for entry in logs)

    async def test_edge_empty_eof_token_cap_minified_and_invalid_utf8(
        self, read_path_mcp: FastMCP, sample_mcp_repo: dict[str, Path]
    ) -> None:
        async with Client(read_path_mcp) as client:
            empty = _extract_payload(
                await client.call_tool("read_file_slice", {"request": {"path": "empty.txt"}})
            )
            beyond_eof = _extract_payload(
                await client.call_tool(
                    "read_file_slice",
                    {"request": {"path": "long_file.txt", "line_start": 100, "line_end": 120}},
                )
            )
            bad_utf8 = _extract_payload(
                await client.call_tool("read_file_slice", {"request": {"path": "bad_utf8.txt"}})
            )

        capped_mcp = create_read_path_mcp(root_dir=sample_mcp_repo["project"], max_tokens=40)
        async with Client(capped_mcp) as client:
            capped = _extract_payload(
                await client.call_tool(
                    "read_file_slice",
                    {"request": {"path": "long_file.txt", "line_start": 1, "line_end": 50}},
                )
            )
            minified = _extract_payload(
                await client.call_tool("read_file_slice", {"request": {"path": "minified.js"}})
            )

        assert empty["content"] == "" and empty["total_line_count"] == 0 and empty["line_read"] is None
        assert beyond_eof["content"] == "" and beyond_eof["line_read"] is None and beyond_eof["has_more"] is False
        assert capped["token_size"] <= 40 and capped["has_more"] is True and capped["next_line"] is not None
        assert minified["content"].endswith("... [TRUNCATED]") and minified["token_size"] <= 40
        assert "valid line" in bad_utf8["content"] and "\ufffd" in bad_utf8["content"]

    @pytest.mark.parametrize(
        "request_payload",
        [
            {"path": "src", "line_start": 1, "line_end": 5},  # Directory instead of file
            {"path": "src/missing.py", "line_start": 1, "line_end": 5},  # Missing file
            {"path": "long_file.txt", "line_start": 0},  # Invalid line_start
            {"path": "long_file.txt", "line_start": 20, "line_end": 10},  # line_end < line_start
            {"path": "../project_secret/outside.txt"},  # Relative escape
            {"path": "src/../../project_secret/outside.txt"},  # Nested traversal escape
            {"path": "src/repo.py\x00.txt"},  # Null byte injection
        ],
    )
    async def test_sad_errors_surface_as_mcp_tool_error(
        self, read_path_mcp: FastMCP, request_payload: dict[str, Any]
    ) -> None:
        async with Client(read_path_mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("read_file_slice", {"request": request_payload})

    async def test_sad_absolute_outside_path_blocked(
        self, read_path_mcp: FastMCP, sample_mcp_repo: dict[str, Path]
    ) -> None:
        outside_file = str(sample_mcp_repo["sibling_secret"] / "outside.txt")
        async with Client(read_path_mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("read_file_slice", {"request": {"path": outside_file}})


# =======================================================
# search_symbol MCP Integration Tests
# =======================================================


@pytest.mark.anyio
class TestSearchSymbolMCPIntegration:
    async def test_happy_all_and_definitions_modes_strip_metadata_and_log_audit(
        self, read_path_mcp: FastMCP
    ) -> None:
        logs: list[Any] = []

        async def log_handler(message: Any) -> None:
            logs.append(message)

        async with Client(read_path_mcp, log_handler=log_handler) as client:
            all_res = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {"request": {"path": ".", "symbol": "total_cents", "mode": "all"}},
                )
            )
            def_res = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {"request": {"path": "src", "symbol": "total_cents", "mode": "definitions"}},
                )
            )
            sql_only = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {"request": {"path": ".", "symbol": "total_cents", "file_extensions": ["sql"]}},
                )
            )

        assert all_res["read_type"] == "search_symbol"
        assert "metadata" not in all_res
        all_paths = [item["path"] for item in all_res["results"]]
        assert all_paths == ["minified.js", "src/repo.py", "src/schema.sql"]
        assert all_res["total_matches"] == 5

        assert [item["path"] for item in def_res["results"]] == ["src/repo.py", "src/schema.sql"]
        assert def_res["total_matches"] == 2
        assert [item["path"] for item in sql_only["results"]] == ["src/schema.sql"]
        assert any("Searched 'total_cents'" in str(entry.data) for entry in logs)

    async def test_edge_pagination_per_file_cap_token_cap_and_no_matches(
        self, sample_mcp_repo: dict[str, Path]
    ) -> None:
        noisy = sample_mcp_repo["project"] / "noisy.py"
        noisy.write_text(
            "\n".join(f"item_{i} = total_cents + {i}" for i in range(1, 25)) + "\n",
            encoding="utf-8",
        )

        paged_mcp = create_read_path_mcp(
            root_dir=sample_mcp_repo["project"],
            max_matches_per_file=5,
            matches_per_batch=2,
        )
        async with Client(paged_mcp) as client:
            p1 = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {"request": {"path": "noisy.py", "symbol": "total_cents", "batch": 1}},
                )
            )
            p3 = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {"request": {"path": "noisy.py", "symbol": "total_cents", "batch": 3}},
                )
            )
            no_hits = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {"request": {"path": ".", "symbol": "nonexistent_xyz_symbol"}},
                )
            )

        capped_mcp = create_read_path_mcp(root_dir=sample_mcp_repo["project"], max_tokens=20)
        async with Client(capped_mcp) as client:
            tok_capped = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {"request": {"path": ".", "symbol": "total_cents"}},
                )
            )

        assert p1["total_matches"] == 5
        assert p1["results"][0]["truncated_in_file"] is True
        assert p1["line_read"] == "l1-2" and p1["has_more"] is True
        assert p3["line_read"] == "l5-5" and p3["has_more"] is False

        assert tok_capped["token_size"] <= 20 and tok_capped["has_more"] is True
        assert no_hits["total_matches"] == 0 and no_hits["results"] == [] and no_hits["line_read"] is None

    @pytest.mark.parametrize(
        "request_payload",
        [
            {"path": ".", "symbol": ""},  # Empty symbol
            {"path": ".", "symbol": "   "},  # Whitespace-only symbol
            {"path": ".", "symbol": "[unclosed", "use_regex": True},  # Invalid regex
            {"path": "src/missing.py", "symbol": "orders"},  # Missing path
            {"path": "..", "symbol": "orders"},  # Parent traversal escape
            {"path": "../project_secret/outside.txt", "symbol": "orders"},  # Sibling escape
        ],
    )
    async def test_sad_errors_surface_as_mcp_tool_error(
        self, read_path_mcp: FastMCP, request_payload: dict[str, Any]
    ) -> None:
        async with Client(read_path_mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("search_symbol", {"request": request_payload})


# =======================================================
# End-to-End Auditor Workflow & Parallel Execution Tests
# =======================================================


@pytest.mark.anyio
class TestEndToEndAuditorWorkflow:
    async def test_happy_three_tool_funnel_from_tree_to_schema_to_code_slice(
        self, read_path_mcp: FastMCP
    ) -> None:
        """Simulate an autonomous auditor: list -> read schema -> search symbol -> read code slice."""
        async with Client(read_path_mcp) as client:
            tree = _extract_payload(
                await client.call_tool("list_folder_content", {"request": {"path": "src"}})
            )
            files = tree["content"].splitlines()
            assert "src/schema.sql" in files

            schema = _extract_payload(
                await client.call_tool("read_file_slice", {"request": {"path": "src/schema.sql"}})
            )
            assert "total_cents INTEGER NOT NULL" in schema["content"]

            hits = _extract_payload(
                await client.call_tool(
                    "search_symbol",
                    {
                        "request": {
                            "path": "src",
                            "symbol": "total_cents",
                            "file_extensions": [".py"],
                        }
                    },
                )
            )
            assert hits["results"][0]["path"] == "src/repo.py"
            hit_line = hits["results"][0]["lines"][0]

            code_slice = _extract_payload(
                await client.call_tool(
                    "read_file_slice",
                    {
                        "request": {
                            "path": hits["results"][0]["path"],
                            "line_start": max(1, hit_line - 2),
                            "line_end": hit_line + 2,
                        }
                    },
                )
            )
            assert "SELECT id, total_cents FROM orders" in code_slice["content"]

    async def test_happy_parallel_tool_calls_execute_concurrently(
        self, read_path_mcp: FastMCP
    ) -> None:
        """Verify all 3 async tools can be invoked concurrently via asyncio.gather."""
        async with Client(read_path_mcp) as client:
            res_list, res_slice, res_search = await asyncio.gather(
                client.call_tool("list_folder_content", {"request": {"path": "src"}}),
                client.call_tool(
                    "read_file_slice",
                    {"request": {"path": "src/schema.sql", "line_start": 1, "line_end": 5}},
                ),
                client.call_tool(
                    "search_symbol",
                    {"request": {"path": "src", "symbol": "orders", "mode": "definitions"}},
                ),
            )

        p_list = _extract_payload(res_list)
        p_slice = _extract_payload(res_slice)
        p_search = _extract_payload(res_search)

        assert p_list["read_type"] == "directory_read"
        assert p_slice["read_type"] == "file_slice"
        assert p_search["read_type"] == "search_symbol"
        assert p_search["total_matches"] == 2
