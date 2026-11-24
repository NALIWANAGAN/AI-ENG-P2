from datetime import datetime
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from core.settings import settings

ReadType = Literal["directory_read", "file_slice", "search_symbol"]
SearchMode = Literal["all", "definitions"]

# =============================================
# Request Models
# =============================================
class ListFolderRequest(BaseModel):
    """Input parameters for listing unignored files in a directory."""

    path: Path | str = Field(
        description="Path to the target directory to inspect."
    )
    batch: int = Field(
        default=1,
        ge=1,
        description="1-indexed batch number to fetch when directory contents are paginated.",
    )


class ReadFileSliceRequest(BaseModel):
    """Input parameters for reading a specific 1-indexed line range from a UTF-8 file."""

    path: Path | str = Field(
        description="Path to the target UTF-8 file to read."
    )
    line_start: int = Field(
        default=1,
        ge=1,
        description="1-indexed starting line number (inclusive).",
    )
    line_end: int | None = Field(
        default=None,
        ge=1,
        description=(
            f"Optional 1-indexed ending line number (inclusive). "
            f"Omit (null) to read the default {settings.DEFAULT_LINES_PER_BATCH}-line batch — do NOT pass -1."
        ),
    )

    @model_validator(mode="after")
    def _validate_line_range(self) -> "ReadFileSliceRequest":
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class SearchSymbolRequest(BaseModel):
    """Input parameters for searching a symbol across a file or directory."""

    path: Path | str = Field(
        description="Path to a single file or a directory to search within."
    )
    symbol: str = Field(
        min_length=1,
        description="Symbol name, table/column identifier, or regex pattern to search for.",
    )
    mode: SearchMode = Field(
        default="all",
        description="'all' searches code, SQL strings, and dict keys; 'definitions' filters to declarations/DDL.",
    )
    whole_word: bool = Field(
        default=True,
        description="Match whole identifiers only so e.g. 'id' does not match 'valid'.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Enable case-sensitive matching when True.",
    )
    use_regex: bool = Field(
        default=False,
        description="Treat `symbol` as a regular expression instead of literal text.",
    )
    file_extensions: list[str] | None = Field(
        default=None,
        description="Optional list of file extensions to include (e.g. ['.py', '.sql'] or ['py', 'sql']).",
    )
    batch: int = Field(
        default=1,
        ge=1,
        description="1-indexed batch number for paginated search matches.",
    )

    @field_validator("symbol")
    @classmethod
    def _validate_non_blank_symbol(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("symbol must be a non-empty string")
        return value


# =============================================
# Response Models
# =============================================
class FileMetadata(BaseModel):
    """For logging and auditing only. Do NOT send this to the LLM."""

    file_name: str
    file_path: str
    creation_date: datetime | None = None
    modification_date: datetime | None = None


class FileContentPublic(BaseModel):
    """LLM-facing file or directory read content (excludes internal audit metadata)."""

    content: str | None = Field(
        default=None,
        description="Raw string file content or newline-delimited directory listing.",
    )
    read_type: ReadType = Field(
        description="Type of read being done e.g. directory_read, file_slice, search_symbol.",
    )
    total_line_count: int = Field(
        default=0,
        description="Total line count in the file or total entries in the directory.",
    )
    batch: int = Field(
        default=1,
        ge=1,
        description="1-indexed batch number per read e.g. 1.",
    )
    line_read: str | None = Field(
        default=None,
        description="1-indexed line range read e.g. l1-200. Capped by line range and token budget.",
    )
    next_line: int | None = Field(
        default=None,
        ge=1,
        description="Next 1-indexed line number to pass as `line_start` when `has_more` is True.",
    )
    token_size: int = Field(
        default=0,
        description="The token size of the returned content.",
    )
    has_more: bool = Field(
        default=False,
        description="True if more lines or entries remain after this read.",
    )


class FileContent(FileContentPublic):
    """File or directory read content and its audit metadata."""

    metadata: FileMetadata


class SymbolMatch(BaseModel):
    """A single line match for a searched symbol."""

    line_number: int = Field(
        ge=1,
        description="1-indexed line number where the symbol matched.",
    )
    snippet: str = Field(
        description="Trimmed source line preview containing the match.",
    )


class SearchSymbol(BaseModel):
    """Per-file symbol matches returned by `search_symbol`."""

    path: str = Field(
        description="Relative POSIX path of the file where the searched symbol exists.",
    )
    lines: list[int] = Field(
        default_factory=list,
        description="1-indexed line numbers where the symbol appears in the file.",
    )
    matches: list[SymbolMatch] = Field(
        default_factory=list,
        description="Matched line numbers paired with trimmed code snippets.",
    )
    truncated_in_file: bool = Field(
        default=False,
        description="True if additional matches in this file were omitted by max_matches_per_file.",
    )


class SearchSymbolPublicResult(BaseModel):
    """LLM-facing paginated, token-capped symbol search response (excludes audit metadata)."""

    symbol: str = Field(description="Searched symbol or pattern.")
    mode: SearchMode = Field(description="Search mode used: 'all' or 'definitions'.")
    read_type: ReadType = Field(
        default="search_symbol",
        description="Type of read being done e.g. directory_read, file_slice, search_symbol.",
    )
    results: list[SearchSymbol] = Field(
        default_factory=list,
        description="Files and line matches included in this batch.",
    )
    total_matches: int = Field(
        default=0,
        description="Total matching lines across all scanned files (after per-file cap).",
    )
    total_files_matched: int = Field(
        default=0,
        description="Total number of files containing at least one match.",
    )
    batch: int = Field(
        default=1,
        ge=1,
        description="1-indexed batch number.",
    )
    line_read: str | None = Field(
        default=None,
        description="1-indexed match range returned in this batch e.g. l1-50.",
    )
    token_size: int = Field(
        default=0,
        description="The token size of the returned matches.",
    )
    has_more: bool = Field(
        default=False,
        description="True if more matches remain after this batch.",
    )


class SearchSymbolResult(SearchSymbolPublicResult):
    """Paginated, token-capped symbol search response and its audit metadata."""

    metadata: FileMetadata
