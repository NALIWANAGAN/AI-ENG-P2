from datetime import datetime
import fnmatch
from pathlib import Path
import re
import tiktoken
from core.settings import settings
from schema.read_path import (
    FileContent,
    FileMetadata,
    ListFolderRequest,
    ReadFileSliceRequest,
    SearchMode,
    SearchSymbol,
    SearchSymbolRequest,
    SearchSymbolResult,
    SymbolMatch,
)


class ReadPath:
    _TRUNCATION_SUFFIX = "... [TRUNCATED]"
    _SNIPPET_MAX_CHARS = 200
    _SQL_COLUMN_TYPES = (
        r"BIGINT|INT|INTEGER|SMALLINT|TINYINT|SERIAL|BIGSERIAL|SMALLSERIAL|"
        r"VARCHAR|CHAR|CHARACTER|NVARCHAR|TEXT|CLOB|BOOLEAN|BOOL|"
        r"TIMESTAMP|TIMESTAMPTZ|DATETIME|DATE|TIME|INTERVAL|"
        r"NUMERIC|DECIMAL|FLOAT|DOUBLE|REAL|MONEY|UUID|JSON|JSONB|XML|BYTEA|BLOB|ENUM"
    )

    def __init__(
        self,
        root_dir: Path | str | None = None,
        max_tokens: int = settings.MAX_READ_TOKENS,
        items_per_batch: int = settings.DEFAULT_ITEMS_PER_BATCH,
        matches_per_batch: int = settings.DEFAULT_SEARCH_MATCHES_PER_BATCH,
        max_matches_per_file: int = settings.DEFAULT_MAX_MATCHES_PER_FILE,
        encoding_name: str = settings.TOKENIZER_ENCODING,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if items_per_batch < 1:
            raise ValueError("items_per_batch must be >= 1")
        if matches_per_batch < 1:
            raise ValueError("matches_per_batch must be >= 1")
        if max_matches_per_file < 1:
            raise ValueError("max_matches_per_file must be >= 1")

        base = Path(root_dir) if root_dir is not None else Path.cwd()
        self._validate_existence(base)
        if not base.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {base}")
        self._root_dir = base.resolve()
        self._max_tokens = max_tokens
        self._items_per_batch = items_per_batch
        self._matches_per_batch = matches_per_batch
        self._max_matches_per_file = max_matches_per_file

        try:
            self._encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._encoder = None

    @property
    def root_dir(self) -> Path:
        """Resolved root directory that all read/search operations are confined to."""
        return self._root_dir

    def list_folder_content(self, request: ListFolderRequest) -> FileContent:
        """List nested unignored relative file paths under `request.path` in paginated, token-capped batches."""
        root = self._resolve_and_validate_path(request.path)
        if not root.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {request.path}")

        rel_paths = sorted(
            entry.relative_to(self._root_dir).as_posix()
            for entry in self._auto_list_unignored_stdlib(root)
        )

        line_start = (request.batch - 1) * self._items_per_batch + 1
        line_end = request.batch * self._items_per_batch

        content, actual_start, actual_end, token_size, has_more = self._slice_lines_within_budget(
            lines=rel_paths,
            line_start=line_start,
            line_end=line_end,
            max_tokens=self._max_tokens,
        )

        return FileContent(
            content=content,
            read_type="directory_read",
            total_line_count=len(rel_paths),
            batch=request.batch,
            line_read=f"l{actual_start}-{actual_end}" if actual_start else None,
            next_line=actual_end + 1 if has_more else None,
            token_size=token_size,
            has_more=has_more,
            metadata=self._file_metadata(root),
        )

    def read_file_slice(self, request: ReadFileSliceRequest) -> FileContent:
        """Read a targeted 1-indexed line range from a UTF-8 file, capped by `self._max_tokens`."""
        path = self._resolve_and_validate_path(request.path)
        if path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {request.path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {request.path}")

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        target_end = (
            request.line_end
            if request.line_end is not None
            else request.line_start + settings.DEFAULT_LINES_PER_BATCH - 1
        )

        content, actual_start, actual_end, token_size, has_more = self._slice_lines_within_budget(
            lines=lines,
            line_start=request.line_start,
            line_end=target_end,
            max_tokens=self._max_tokens,
        )

        return FileContent(
            content=content,
            read_type="file_slice",
            total_line_count=len(lines),
            batch=1,
            line_read=f"l{actual_start}-{actual_end}" if actual_start else None,
            next_line=actual_end + 1 if has_more else None,
            token_size=token_size,
            has_more=has_more,
            metadata=self._file_metadata(path),
        )

    def search_symbol(self, request: SearchSymbolRequest) -> SearchSymbolResult:
        """Search for a symbol (table, column, function, class, etc.) across a file or directory.
        Respects `.gitignore`, skips binary files, supports `'all'` (text/SQL/dict/code) and
        `'definitions'` (Python/SQL/JS/TS/Prisma declarations) modes, and applies batch + token caps.
        """
        target_path = self._resolve_and_validate_path(request.path)

        pattern = self._compile_symbol_pattern(
            symbol=request.symbol,
            mode=request.mode,
            whole_word=request.whole_word,
            case_sensitive=request.case_sensitive,
            use_regex=request.use_regex,
        )

        normalized_exts: set[str] | None = None
        if request.file_extensions:
            normalized_exts = {
                ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                for ext in request.file_extensions
                if ext.strip()
            }

        if target_path.is_file():
            files_to_scan = [(target_path.relative_to(self._root_dir).as_posix(), target_path)]
            audit_root = target_path
        elif target_path.is_dir():
            files_to_scan = sorted(
                (entry.relative_to(self._root_dir).as_posix(), entry)
                for entry in self._auto_list_unignored_stdlib(target_path)
            )
            audit_root = target_path
        else:
            raise ValueError(f"Path is neither a regular file nor a directory: {request.path}")

        # Collect matches in deterministic order: (rel_path, line_num, snippet, truncated_in_file)
        flat_matches: list[tuple[str, int, str, bool]] = []
        matched_files: set[str] = set()

        for rel_path, file_path in files_to_scan:
            if normalized_exts is not None and file_path.suffix.lower() not in normalized_exts:
                continue
            if self._is_binary_file(file_path):
                continue

            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            file_hits: list[tuple[int, str]] = []
            truncated_in_file = False

            for line_num, raw_line in enumerate(lines, start=1):
                match = pattern.search(raw_line)
                if match is None:
                    continue
                if len(file_hits) >= self._max_matches_per_file:
                    truncated_in_file = True
                    break
                file_hits.append((line_num, self._format_snippet(raw_line, match)))

            if file_hits:
                matched_files.add(rel_path)
                for line_num, snippet in file_hits:
                    flat_matches.append((rel_path, line_num, snippet, truncated_in_file))

        total_matches = len(flat_matches)
        start_idx = (request.batch - 1) * self._matches_per_batch
        end_idx = min(total_matches, request.batch * self._matches_per_batch)

        if total_matches == 0 or start_idx >= total_matches:
            return SearchSymbolResult(
                symbol=request.symbol,
                mode=request.mode,
                read_type="search_symbol",
                results=[],
                total_matches=total_matches,
                total_files_matched=len(matched_files),
                batch=request.batch,
                line_read=None,
                token_size=0,
                has_more=False,
                metadata=self._file_metadata(audit_root),
            )

        selected_flat: list[tuple[str, int, str, bool]] = []
        used_tokens = 0

        for item in flat_matches[start_idx:end_idx]:
            rel_path, line_num, snippet, _ = item
            entry_tokens = self._count_tokens(f"{rel_path}:{line_num}: {snippet}\n")

            if selected_flat and (used_tokens + entry_tokens > self._max_tokens):
                break

            if not selected_flat and entry_tokens > self._max_tokens:
                prefix_tokens = self._count_tokens(f"{rel_path}:{line_num}: ")
                snippet_budget = max(1, self._max_tokens - prefix_tokens)
                truncated_snippet = self._truncate_line_to_budget(snippet, snippet_budget)
                item = (rel_path, line_num, truncated_snippet, item[3])
                selected_flat.append(item)
                used_tokens = self._count_tokens(f"{rel_path}:{line_num}: {truncated_snippet}\n")
                break

            selected_flat.append(item)
            used_tokens += entry_tokens

        grouped: dict[str, SearchSymbol] = {}
        for rel_path, line_num, snippet, truncated_in_file in selected_flat:
            if rel_path not in grouped:
                grouped[rel_path] = SearchSymbol(
                    path=rel_path,
                    lines=[],
                    matches=[],
                    truncated_in_file=truncated_in_file,
                )
            grouped[rel_path].lines.append(line_num)
            grouped[rel_path].matches.append(
                SymbolMatch(line_number=line_num, snippet=snippet)
            )

        results = list(grouped.values())
        actual_start = start_idx + 1
        actual_end = start_idx + len(selected_flat)
        has_more = actual_end < total_matches

        payload_text = "\n".join(
            f"{rel_path}:{line_num}: {snippet}"
            for rel_path, line_num, snippet, _ in selected_flat
        )

        return SearchSymbolResult(
            symbol=request.symbol,
            mode=request.mode,
            read_type="search_symbol",
            results=results,
            total_matches=total_matches,
            total_files_matched=len(matched_files),
            batch=request.batch,
            line_read=f"l{actual_start}-{actual_end}",
            token_size=self._count_tokens(payload_text),
            has_more=has_more,
            metadata=self._file_metadata(audit_root),
        )

    def _compile_symbol_pattern(
        self,
        symbol: str,
        mode: SearchMode,
        whole_word: bool,
        case_sensitive: bool,
        use_regex: bool,
    ) -> re.Pattern[str]:
        flags = 0 if case_sensitive else re.IGNORECASE
        raw_pat = symbol if use_regex else re.escape(symbol)

        if mode == "all":
            final_pat = rf"(?<!\w)(?:{raw_pat})(?!\w)" if whole_word else raw_pat
        else:
            sym_pat = rf"(?:{raw_pat})(?!\w)" if whole_word else rf"(?:{raw_pat})"
            final_pat = "|".join([
                # Python / JS / TS / Prisma declarations
                rf"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:def|class|function|interface|type|enum|model|const|let|var)\s+{sym_pat}",
                # Python class attributes, Pydantic fields, ORM columns, module variables
                rf"^\s*{sym_pat}\s*(?::[^=]+)?=(?!=)",
                # SQL DDL definitions (tables, views, indexes, types, functions, triggers, schemas)
                rf"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?(?:UNIQUE\s+)?(?:TABLE|VIEW|MATERIALIZED\s+VIEW|INDEX|TYPE|ENUM|FUNCTION|PROCEDURE|TRIGGER|SCHEMA)\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:[\w\"`]+\.)?[\"`]?{sym_pat}",
                # SQL ALTER TABLE target or ADD COLUMN
                rf"^\s*ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:[\w\"`]+\.)?[\"`]?{sym_pat}",
                rf"^\s*ADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?[\"`]?{sym_pat}",
                # SQL column definitions inside CREATE TABLE
                rf"^\s*[\"`]?{sym_pat}[\"`]?\s+(?:{self._SQL_COLUMN_TYPES})\b",
            ])

        try:
            return re.compile(final_pat, flags)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern '{symbol}': {exc}") from exc

    def _is_binary_file(self, path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                chunk = f.read(4096)
            return b"\x00" in chunk
        except OSError:
            return True

    def _format_snippet(self, line: str, match: re.Match[str]) -> str:
        max_chars = self._SNIPPET_MAX_CHARS
        stripped = line.strip()
        if len(stripped) <= max_chars:
            return stripped

        leading_ws = len(line) - len(line.lstrip())
        rel_start = max(0, match.start() - leading_ws)
        rel_end = max(rel_start, match.end() - leading_ws)

        half_window = max_chars // 2
        win_start = max(0, rel_start - half_window)
        win_end = min(len(stripped), max(rel_end + half_window, win_start + max_chars))
        win_start = max(0, win_end - max_chars)

        prefix = "..." if win_start > 0 else ""
        suffix = "..." if win_end < len(stripped) else ""
        return f"{prefix}{stripped[win_start:win_end]}{suffix}"

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return (len(text) + 3) // 4

    def _truncate_line_to_budget(self, line: str, max_tokens: int) -> str:
        suffix = self._TRUNCATION_SUFFIX
        suffix_tokens = self._count_tokens(suffix)
        content_budget = max(1, max_tokens - suffix_tokens)

        if self._encoder is not None:
            encoded = self._encoder.encode(line, disallowed_special=())
            return self._encoder.decode(encoded[:content_budget]) + suffix
        max_chars = max(1, (max_tokens * 4) - len(suffix))
        return line[:max_chars] + suffix

    def _slice_lines_within_budget(
        self,
        lines: list[str],
        line_start: int,
        line_end: int,
        max_tokens: int,
    ) -> tuple[str, int, int, int, bool]:
        """Slice 1-indexed lines [line_start, line_end] without exceeding `max_tokens`.

        Returns:
            Tuple of (content, actual_start, actual_end, token_size, has_more).
        """
        total = len(lines)
        if total == 0 or line_start > total:
            return "", 0, 0, 0, False

        start_idx = line_start - 1
        end_idx = min(total, line_end)

        selected: list[str] = []
        used_tokens = 0

        for line in lines[start_idx:end_idx]:
            line_tokens = self._count_tokens(line + "\n")

            if selected and (used_tokens + line_tokens > max_tokens):
                break

            if not selected and line_tokens > max_tokens:
                truncated = self._truncate_line_to_budget(line, max_tokens)
                selected.append(truncated)
                used_tokens = self._count_tokens(truncated)
                break

            selected.append(line)
            used_tokens += line_tokens

        content = "\n".join(selected)
        actual_start = start_idx + 1
        actual_end = start_idx + len(selected)
        has_more = actual_end < total
        return content, actual_start, actual_end, self._count_tokens(content), has_more

    def _file_metadata(self, path: Path) -> FileMetadata:
        """Collect OS file-system metadata for a path."""
        resolved = path.resolve()
        stat = resolved.stat()
        birthtime = getattr(stat, "st_birthtime", None)

        return FileMetadata(
            file_name=resolved.name,
            file_path=str(resolved),
            creation_date=datetime.fromtimestamp(birthtime) if birthtime is not None else None,
            modification_date=datetime.fromtimestamp(stat.st_mtime),
        )

    def _resolve_and_validate_path(self, raw_path: Path | str) -> Path:
        """Resolve `raw_path` relative to `self._root_dir` and enforce root-directory confinement."""
        if "\x00" in str(raw_path):
            raise ValueError("Path must not contain null bytes")

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = self._root_dir / candidate

        resolved = candidate.resolve()
        if not resolved.is_relative_to(self._root_dir):
            raise PermissionError(
                f"Access denied: path '{raw_path}' resolves outside the allowed root directory."
            )

        self._validate_existence(resolved)
        return resolved

    def _validate_existence(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

    def _auto_list_unignored_stdlib(self, path: Path) -> list[Path]:
        path = path.resolve()

        patterns: list[str] = [".git/"]
        ignore_files = [self._root_dir / ".gitignore"]
        if path != self._root_dir:
            ignore_files.append(path / ".gitignore")

        for ignore_file in ignore_files:
            if ignore_file.is_file():
                for raw_line in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = raw_line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)

        def _is_ignored(rel_posix: str, name: str, is_dir: bool) -> bool:
            ignored = False
            for pat in patterns:
                negated = pat.startswith("!")
                raw_pat = pat[1:] if negated else pat
                if not raw_pat:
                    continue

                dir_only = raw_pat.endswith("/")
                if dir_only and not is_dir:
                    continue

                trimmed = raw_pat.rstrip("/")
                anchored = "/" in trimmed
                clean_pat = trimmed.lstrip("/")

                matched = (
                    fnmatch.fnmatch(rel_posix, clean_pat)
                    if anchored
                    else fnmatch.fnmatch(name, clean_pat) or fnmatch.fnmatch(rel_posix, clean_pat)
                )
                if matched:
                    ignored = not negated
            return ignored

        results: list[Path] = []
        visited_dirs: set[Path] = {path}

        def _walk(current_dir: Path) -> None:
            for entry in current_dir.iterdir():
                resolved_entry = entry.resolve()
                # Block symlinks/junctions that escape outside root_dir
                if not resolved_entry.is_relative_to(self._root_dir):
                    continue

                rel_posix = entry.relative_to(self._root_dir).as_posix()
                is_dir = resolved_entry.is_dir()
                if _is_ignored(rel_posix, entry.name, is_dir):
                    continue
                if is_dir:
                    if resolved_entry in visited_dirs:
                        continue
                    visited_dirs.add(resolved_entry)
                    _walk(resolved_entry)
                elif resolved_entry.is_file():
                    results.append(resolved_entry)

        _walk(path)
        return results
