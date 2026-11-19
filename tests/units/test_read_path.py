import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from schema.read_path import (
    FileContent,
    FileMetadata,
    ListFolderRequest,
    ReadFileSliceRequest,
    SearchSymbolRequest,
    SearchSymbolResult,
)
from services.read_path import ReadPath


@pytest.fixture
def sample_repo(tmp_path: Path) -> dict[str, Path]:
    """Create an isolated sample repository plus a sibling prefix-collision directory."""
    project = tmp_path / "project"
    sibling_secret = tmp_path / "project_secret"
    project.mkdir()
    sibling_secret.mkdir()

    # Sibling outside root_dir (for path-confinement & prefix-collision tests)
    (sibling_secret / "outside.txt").write_text("orders total_cents secret\n", encoding="utf-8")

    # .gitignore with wildcards, negation, directory-only, and root-anchored patterns
    (project / ".gitignore").write_text(
        "\n".join(
            [
                "# Comments should be ignored",
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

    # Always-ignored .git directory
    (project / ".git").mkdir()
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    # Ignored directory & ignored log vs negated log
    (project / "ignored_dir").mkdir()
    (project / "ignored_dir" / "hidden.py").write_text("total_cents = 999\n", encoding="utf-8")
    (project / "debug.log").write_text("total_cents debug\n", encoding="utf-8")
    (project / "keep.log").write_text("kept log line\n", encoding="utf-8")

    # Regular file named `build` should NOT be ignored by `build/` (directory-only rule)
    (project / "build").write_text("regular file named build\n", encoding="utf-8")

    # Root-anchored `/root_only.txt` ignored at root, kept in subdir
    (project / "root_only.txt").write_text("ignored at root\n", encoding="utf-8")

    # Application source & SQL schema files
    src = project / "src"
    src.mkdir()
    (src / "root_only.txt").write_text("kept inside src\n", encoding="utf-8")

    # Nested `build/` directory SHOULD be ignored by `build/`
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

    # Binary file containing null bytes + symbol text (should be skipped by search_symbol)
    assets = project / "assets"
    assets.mkdir()
    (assets / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00total_cents")

    # Edge-case text files
    (project / "empty.txt").write_text("", encoding="utf-8")
    (project / "long_file.txt").write_text(
        "\n".join(f"line {i:02d}: sample content for pagination testing" for i in range(1, 51)) + "\n",
        encoding="utf-8",
    )
    (project / "minified.js").write_text(
        "a" * 300 + " total_cents " + "b" * 4700,
        encoding="utf-8",
    )
    (project / "bad_utf8.txt").write_bytes(b"valid line\n\xff\xfe invalid utf8\n")

    # Empty subdirectory
    (project / "empty_dir").mkdir()

    return {
        "base": tmp_path,
        "project": project,
        "sibling_secret": sibling_secret,
    }


@pytest.fixture
def reader(sample_repo: dict[str, Path]) -> ReadPath:
    return ReadPath(root_dir=sample_repo["project"])


# =======================================================
# Initialization Tests
# =======================================================

class TestReadPathInit:
    def test_init_with_valid_root_dir(self, sample_repo: dict[str, Path]) -> None:
        rf = ReadPath(root_dir=sample_repo["project"])
        assert rf.root_dir == sample_repo["project"].resolve()

    def test_init_defaults_to_cwd(self) -> None:
        rf = ReadPath()
        assert rf.root_dir == Path.cwd().resolve()

    def test_init_missing_root_raises_file_not_found(self, sample_repo: dict[str, Path]) -> None:
        with pytest.raises(FileNotFoundError):
            ReadPath(root_dir=sample_repo["base"] / "does_not_exist")

    def test_init_file_as_root_raises_not_a_directory(self, sample_repo: dict[str, Path]) -> None:
        with pytest.raises(NotADirectoryError):
            ReadPath(root_dir=sample_repo["project"] / "empty.txt")


# =======================================================
# list_folder_content Tests
# =======================================================

class TestListFolderContent:
    def test_happy_lists_unignored_files_respecting_gitignore(self, reader: ReadPath) -> None:
        res = reader.list_folder_content(ListFolderRequest(path="."))

        assert isinstance(res, FileContent)
        assert res.read_type == "directory_read"
        assert res.batch == 1
        assert res.has_more is False
        assert res.token_size > 0
        assert isinstance(res.metadata, FileMetadata)

        listed = (res.content or "").splitlines()
        assert res.total_line_count == len(listed)
        assert res.line_read == f"l1-{len(listed)}"

        # Included files
        assert "build" in listed
        assert "keep.log" in listed
        assert "src/root_only.txt" in listed
        assert "src/schema.sql" in listed
        assert "src/repo.py" in listed

        # Excluded by .gitignore or default .git/
        assert ".git/HEAD" not in listed
        assert "debug.log" not in listed
        assert "root_only.txt" not in listed
        assert "ignored_dir/hidden.py" not in listed
        assert "src/build/generated.py" not in listed

    def test_happy_subdirectory_returns_root_relative_paths_and_applies_root_gitignore(
        self, reader: ReadPath
    ) -> None:
        res = reader.list_folder_content(ListFolderRequest(path="src"))
        listed = (res.content or "").splitlines()

        assert listed == ["src/repo.py", "src/root_only.txt", "src/schema.sql"]
        # Returned path can be passed straight into read_file_slice
        file_res = reader.read_file_slice(ReadFileSliceRequest(path=listed[0]))
        assert "OrderRepo" in (file_res.content or "")

    def test_happy_pagination_across_batches(self, reader: ReadPath) -> None:
        page1 = reader.list_folder_content(ListFolderRequest(path=".", batch=1, items_per_batch=3))
        page2 = reader.list_folder_content(ListFolderRequest(path=".", batch=2, items_per_batch=3))

        assert page1.line_read == "l1-3"
        assert page1.has_more is True
        assert page2.line_read == "l4-6"
        assert page1.total_line_count == page2.total_line_count
        assert set((page1.content or "").splitlines()).isdisjoint((page2.content or "").splitlines())

    def test_edge_empty_directory(self, reader: ReadPath) -> None:
        res = reader.list_folder_content(ListFolderRequest(path="empty_dir"))
        assert res.content == ""
        assert res.total_line_count == 0
        assert res.line_read is None
        assert res.token_size == 0
        assert res.has_more is False

    def test_edge_out_of_range_batch(self, reader: ReadPath) -> None:
        res = reader.list_folder_content(ListFolderRequest(path=".", batch=99, items_per_batch=50))
        assert res.content == ""
        assert res.line_read is None
        assert res.token_size == 0
        assert res.has_more is False

    def test_edge_max_tokens_caps_directory_output(self, reader: ReadPath) -> None:
        res = reader.list_folder_content(
            ListFolderRequest(path=".", batch=1, items_per_batch=100, max_tokens=10)
        )
        assert res.token_size <= 10
        assert res.has_more is True

    def test_sad_file_path_raises_not_a_directory_error(self, reader: ReadPath) -> None:
        with pytest.raises(NotADirectoryError):
            reader.list_folder_content(ListFolderRequest(path="src/repo.py"))

    def test_sad_missing_path_raises_file_not_found_error(self, reader: ReadPath) -> None:
        with pytest.raises(FileNotFoundError):
            reader.list_folder_content(ListFolderRequest(path="missing_folder"))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"path": ".", "batch": 0},
            {"path": ".", "items_per_batch": 0},
            {"path": ".", "max_tokens": 0},
        ],
    )
    def test_sad_invalid_request_params_raise_validation_error(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            ListFolderRequest(**kwargs)


# =======================================================
# read_file_slice Tests
# =======================================================

class TestReadPathSlice:
    def test_happy_reads_exact_inclusive_line_range_and_next_line(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(
            ReadFileSliceRequest(path="long_file.txt", line_start=10, line_end=15)
        )
        lines = (res.content or "").splitlines()
        assert res.read_type == "file_slice"
        assert res.total_line_count == 50
        assert res.line_read == "l10-15"
        assert res.next_line == 16
        assert len(lines) == 6
        assert lines[0].startswith("line 10:")
        assert lines[-1].startswith("line 15:")
        assert res.has_more is True

        cont = reader.read_file_slice(
            ReadFileSliceRequest(path="long_file.txt", line_start=res.next_line, line_end=20)
        )
        assert cont.line_read == "l16-20"
        assert (cont.content or "").splitlines()[0].startswith("line 16:")

    def test_happy_default_line_start_and_line_end_and_end_past_eof(self, reader: ReadPath) -> None:
        from_top = reader.read_file_slice(ReadFileSliceRequest(path="long_file.txt"))
        assert from_top.line_read == "l1-50"
        assert from_top.has_more is False
        assert from_top.next_line is None

        res = reader.read_file_slice(
            ReadFileSliceRequest(path="long_file.txt", line_start=45, line_end=999)
        )
        assert res.line_read == "l45-50"
        assert res.has_more is False
        assert res.next_line is None

        res_default = reader.read_file_slice(
            ReadFileSliceRequest(path="long_file.txt", line_start=45)
        )
        assert res_default.line_read == "l45-50"
        assert res_default.has_more is False
        assert res_default.next_line is None

    def test_happy_accepts_absolute_path_inside_root(
        self, reader: ReadPath, sample_repo: dict[str, Path]
    ) -> None:
        abs_file = sample_repo["project"] / "src" / "schema.sql"
        res = reader.read_file_slice(ReadFileSliceRequest(path=abs_file))
        assert "CREATE TABLE orders" in (res.content or "")
        assert res.metadata.file_name == "schema.sql"

    def test_edge_empty_file(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(ReadFileSliceRequest(path="empty.txt"))
        assert res.content == ""
        assert res.total_line_count == 0
        assert res.line_read is None
        assert res.next_line is None
        assert res.token_size == 0
        assert res.has_more is False

    def test_edge_single_line_slice(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(
            ReadFileSliceRequest(path="long_file.txt", line_start=5, line_end=5)
        )
        assert res.line_read == "l5-5"
        assert res.next_line == 6
        assert (res.content or "").startswith("line 05:")
        assert res.has_more is True

    def test_edge_line_start_beyond_eof(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(
            ReadFileSliceRequest(path="long_file.txt", line_start=100, line_end=120)
        )
        assert res.content == ""
        assert res.line_read is None
        assert res.next_line is None
        assert res.has_more is False

    def test_edge_token_budget_stops_early_and_sets_next_line(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(
            ReadFileSliceRequest(path="long_file.txt", line_start=1, line_end=50, max_tokens=40)
        )
        assert res.token_size <= 40
        assert res.has_more is True
        assert res.line_read is not None and not res.line_read.endswith("-50")
        assert res.next_line is not None and res.next_line > 1

    def test_edge_minified_single_line_truncates_within_token_budget(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(ReadFileSliceRequest(path="minified.js", max_tokens=50))
        assert res.line_read == "l1-1"
        assert (res.content or "").endswith("... [TRUNCATED]")
        assert res.token_size <= 50
        assert res.has_more is False
        assert res.next_line is None

    def test_edge_invalid_utf8_bytes_replaced_without_crashing(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(ReadFileSliceRequest(path="bad_utf8.txt"))
        assert "valid line" in (res.content or "")
        assert "\ufffd" in (res.content or "")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"path": "long_file.txt", "line_start": 0},
            {"path": "long_file.txt", "line_start": 20, "line_end": 10},
            {"path": "long_file.txt", "max_tokens": 0},
        ],
    )
    def test_sad_invalid_request_params_raise_validation_error(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            ReadFileSliceRequest(**kwargs)

    def test_sad_directory_raises_is_a_directory_error(self, reader: ReadPath) -> None:
        with pytest.raises(IsADirectoryError):
            reader.read_file_slice(ReadFileSliceRequest(path="src", line_start=1, line_end=5))

    def test_sad_missing_file_raises_file_not_found_error(self, reader: ReadPath) -> None:
        with pytest.raises(FileNotFoundError):
            reader.read_file_slice(ReadFileSliceRequest(path="nope.txt", line_start=1, line_end=5))


# =======================================================
# search_symbol Tests
# =======================================================
class TestSearchSymbol:
    def test_happy_mode_all_catches_sql_dict_code_and_skips_ignored_and_binary(
        self, reader: ReadPath
    ) -> None:
        res = reader.search_symbol(SearchSymbolRequest(path=".", symbol="total_cents", mode="all"))

        assert isinstance(res, SearchSymbolResult)
        assert res.read_type == "search_symbol"
        assert res.mode == "all"
        paths = [item.path for item in res.results]
        assert paths == ["minified.js", "src/repo.py", "src/schema.sql"]
        assert res.total_files_matched == 3
        assert res.total_matches == 5

        # Ignored and binary files must never appear
        assert "ignored_dir/hidden.py" not in paths
        assert "debug.log" not in paths
        assert "src/build/generated.py" not in paths
        assert "assets/logo.png" not in paths

    def test_happy_mode_definitions_finds_python_and_sql_declarations(
        self, reader: ReadPath
    ) -> None:
        col_defs = reader.search_symbol(
            SearchSymbolRequest(path="src", symbol="total_cents", mode="definitions")
        )
        assert [item.path for item in col_defs.results] == ["src/repo.py", "src/schema.sql"]
        assert col_defs.total_matches == 2
        assert col_defs.results[0].lines == [7]
        assert col_defs.results[1].lines == [3]

        tbl_defs = reader.search_symbol(
            SearchSymbolRequest(path="src", symbol="orders", mode="definitions")
        )
        assert [item.path for item in tbl_defs.results] == ["src/schema.sql"]
        assert tbl_defs.results[0].lines == [1, 6]

        cls_defs = reader.search_symbol(
            SearchSymbolRequest(path="src", symbol="OrderRepo", mode="definitions")
        )
        assert cls_defs.total_matches == 1
        assert cls_defs.results[0].lines == [1]

    def test_happy_single_file_search_returns_root_relative_path(self, reader: ReadPath) -> None:
        res = reader.search_symbol(
            SearchSymbolRequest(path="src/repo.py", symbol="get_total", mode="definitions")
        )
        assert res.total_matches == 1
        assert res.results[0].path == "src/repo.py"
        assert res.results[0].lines == [2]

    def test_happy_file_extensions_filter_with_and_without_dot(self, reader: ReadPath) -> None:
        sql_no_dot = reader.search_symbol(
            SearchSymbolRequest(path=".", symbol="total_cents", file_extensions=["sql"])
        )
        sql_with_dot = reader.search_symbol(
            SearchSymbolRequest(path=".", symbol="total_cents", file_extensions=[".SQL"])
        )
        assert [x.path for x in sql_no_dot.results] == ["src/schema.sql"]
        assert [x.path for x in sql_with_dot.results] == ["src/schema.sql"]

    def test_happy_whole_word_and_case_sensitivity_and_regex(self, reader: ReadPath) -> None:
        # whole_word=True prevents 'id' matching 'valid'
        id_whole = reader.search_symbol(
            SearchSymbolRequest(path="src/repo.py", symbol="valid", whole_word=True)
        )
        assert id_whole.total_matches == 1
        partial = reader.search_symbol(
            SearchSymbolRequest(path="src/repo.py", symbol="lid", whole_word=True)
        )
        assert partial.total_matches == 0
        partial_loose = reader.search_symbol(
            SearchSymbolRequest(path="src/repo.py", symbol="lid", whole_word=False)
        )
        assert partial_loose.total_matches == 1

        # case_sensitive
        ci = reader.search_symbol(
            SearchSymbolRequest(path="src/repo.py", symbol="total_cents_label", case_sensitive=False)
        )
        cs = reader.search_symbol(
            SearchSymbolRequest(path="src/repo.py", symbol="total_cents_label", case_sensitive=True)
        )
        assert ci.total_matches == 1
        assert cs.total_matches == 0

        # use_regex
        rx = reader.search_symbol(
            SearchSymbolRequest(path="src/repo.py", symbol=r"Order(Repo|Model)", use_regex=True)
        )
        assert rx.total_matches == 2

    def test_edge_pagination_and_max_matches_per_file(
        self, reader: ReadPath, sample_repo: dict[str, Path]
    ) -> None:
        noisy = sample_repo["project"] / "noisy.py"
        noisy.write_text("\n".join(f"v_{i} = total_cents + {i}" for i in range(1, 25)) + "\n", encoding="utf-8")

        p1 = reader.search_symbol(
            SearchSymbolRequest(
                path="noisy.py",
                symbol="total_cents",
                max_matches_per_file=5,
                matches_per_batch=2,
                batch=1,
            )
        )
        assert p1.total_matches == 5
        assert p1.results[0].truncated_in_file is True
        assert p1.line_read == "l1-2"
        assert p1.has_more is True

        p3 = reader.search_symbol(
            SearchSymbolRequest(
                path="noisy.py",
                symbol="total_cents",
                max_matches_per_file=5,
                matches_per_batch=2,
                batch=3,
            )
        )
        assert p3.line_read == "l5-5"
        assert p3.has_more is False

    def test_edge_long_line_snippet_is_centered_and_bounded(self, reader: ReadPath) -> None:
        res = reader.search_symbol(SearchSymbolRequest(path="minified.js", symbol="total_cents"))
        snippet = res.results[0].matches[0].snippet
        assert "total_cents" in snippet
        assert snippet.startswith("...") and snippet.endswith("...")
        assert len(snippet) <= ReadPath._SNIPPET_MAX_CHARS + 6

    def test_edge_max_tokens_caps_search_matches(self, reader: ReadPath) -> None:
        res = reader.search_symbol(
            SearchSymbolRequest(path=".", symbol="total_cents", max_tokens=20)
        )
        assert res.token_size <= 20
        assert res.has_more is True

    def test_edge_no_matches_and_out_of_range_batch(self, reader: ReadPath) -> None:
        empty = reader.search_symbol(SearchSymbolRequest(path=".", symbol="nonexistent_xyz_symbol"))
        assert empty.total_matches == 0
        assert empty.total_files_matched == 0
        assert empty.results == []
        assert empty.line_read is None
        assert empty.token_size == 0
        assert empty.has_more is False

        oob = reader.search_symbol(SearchSymbolRequest(path=".", symbol="total_cents", batch=50))
        assert oob.total_matches > 0
        assert oob.results == []
        assert oob.line_read is None
        assert oob.has_more is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"path": ".", "symbol": ""},
            {"path": ".", "symbol": "   "},
            {"path": ".", "symbol": "orders", "batch": 0},
            {"path": ".", "symbol": "orders", "matches_per_batch": 0},
            {"path": ".", "symbol": "orders", "max_matches_per_file": 0},
            {"path": ".", "symbol": "orders", "max_tokens": 0},
        ],
    )
    def test_sad_invalid_request_params_raise_validation_error(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError):
            SearchSymbolRequest(**kwargs)

    def test_sad_invalid_regex_raises_value_error(self, reader: ReadPath) -> None:
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            reader.search_symbol(SearchSymbolRequest(path=".", symbol="[unclosed", use_regex=True))

    def test_sad_missing_path_raises_file_not_found_error(self, reader: ReadPath) -> None:
        with pytest.raises(FileNotFoundError):
            reader.search_symbol(SearchSymbolRequest(path="src/missing.py", symbol="orders"))


# =======================================================
# Path Confinement Guardrail Tests
# =======================================================


class TestPathConfinementGuardrail:
    @pytest.mark.parametrize(
        "escape_kind",
        [
            "parent_dir",
            "relative_sibling",
            "nested_traversal",
            "absolute_sibling_existing",
            "absolute_sibling_nonexistent",
        ],
    )
    def test_sad_escape_attempts_raise_permission_error_on_all_methods(
        self, reader: ReadPath, sample_repo: dict[str, Path], escape_kind: str
    ) -> None:
        sibling = sample_repo["sibling_secret"]
        paths_map = {
            "parent_dir": "..",
            "relative_sibling": "../project_secret/outside.txt",
            "nested_traversal": "src/../../project_secret/outside.txt",
            "absolute_sibling_existing": str(sibling / "outside.txt"),
            "absolute_sibling_nonexistent": str(sibling / "does_not_exist.txt"),
        }
        bad_path = paths_map[escape_kind]

        with pytest.raises(PermissionError):
            reader.list_folder_content(ListFolderRequest(path=bad_path))
        with pytest.raises(PermissionError):
            reader.read_file_slice(ReadFileSliceRequest(path=bad_path, line_start=1, line_end=5))
        with pytest.raises(PermissionError):
            reader.search_symbol(SearchSymbolRequest(path=bad_path, symbol="orders"))

    def test_sad_null_byte_path_raises_value_error(self, reader: ReadPath) -> None:
        with pytest.raises(ValueError, match="null bytes"):
            reader.read_file_slice(ReadFileSliceRequest(path="src/repo.py\x00.txt"))

    def test_edge_symlink_escape_is_blocked_and_skipped(
        self, reader: ReadPath, sample_repo: dict[str, Path]
    ) -> None:
        link_path = sample_repo["project"] / "escape_link.txt"
        target_outside = sample_repo["sibling_secret"] / "outside.txt"
        try:
            os.symlink(target_outside, link_path)
        except OSError:
            pytest.skip("OS does not permit unprivileged symlink creation")

        with pytest.raises(PermissionError):
            reader.read_file_slice(ReadFileSliceRequest(path="escape_link.txt"))

        listed = reader.list_folder_content(ListFolderRequest(path="."))
        assert "escape_link.txt" not in (listed.content or "")

        search = reader.search_symbol(SearchSymbolRequest(path=".", symbol="secret"))
        assert all(item.path != "escape_link.txt" for item in search.results)


# =======================================================
# Tokenizer & Metadata Behavior Tests
# =======================================================


class TestTokenizerAndMetadata:
    def test_happy_encoder_branch_counts_and_truncates(self, reader: ReadPath) -> None:
        class ByteEncoder:
            def encode(self, text: str, disallowed_special: tuple = ()) -> list[int]:
                return list(text.encode("utf-8"))

            def decode(self, tokens: list[int]) -> str:
                return bytes(tokens).decode("utf-8", errors="ignore")

        reader._encoder = ByteEncoder()
        assert reader._count_tokens("hello") == 5
        assert reader._count_tokens("") == 0

        res = reader.read_file_slice(ReadFileSliceRequest(path="minified.js", max_tokens=30))
        assert (res.content or "").endswith("... [TRUNCATED]")
        assert res.token_size <= 30

    def test_happy_offline_fallback_branch_counts_and_truncates(self, reader: ReadPath) -> None:
        reader._encoder = None
        assert reader._count_tokens("abcdefgh") == 2
        assert reader._count_tokens("") == 0

        res = reader.read_file_slice(ReadFileSliceRequest(path="minified.js", max_tokens=30))
        assert (res.content or "").endswith("... [TRUNCATED]")
        assert res.token_size <= 30

    def test_happy_metadata_excluded_cleanly_for_llm_payload(self, reader: ReadPath) -> None:
        res = reader.read_file_slice(ReadFileSliceRequest(path="src/repo.py"))
        assert res.metadata.file_name == "repo.py"
        assert res.metadata.modification_date is not None

        llm_payload = res.model_dump(exclude={"metadata"})
        assert "metadata" not in llm_payload
        assert llm_payload["read_type"] == "file_slice"
