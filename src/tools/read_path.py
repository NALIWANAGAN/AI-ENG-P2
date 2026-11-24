import asyncio
from pathlib import Path
from typing import Annotated

from fastmcp import Context, FastMCP
from pydantic import Field

from core.settings import settings
from schema.read_path import (
    FileContentPublic,
    ListFolderRequest,
    ReadFileSliceRequest,
    SearchMode,
    SearchSymbolPublicResult,
    SearchSymbolRequest,
)
from services.read_path import ReadPath


def create_read_path_mcp(
    root_dir: Path | str | None = None,
    *,
    max_tokens: int = settings.MAX_READ_TOKENS,
    items_per_batch: int = settings.DEFAULT_ITEMS_PER_BATCH,
    matches_per_batch: int = settings.DEFAULT_SEARCH_MATCHES_PER_BATCH,
    max_matches_per_file: int = settings.DEFAULT_MAX_MATCHES_PER_FILE,
) -> FastMCP:
    """Build a FastMCP sub-server with read/search tools sandboxed to `root_dir`."""
    sub_mcp = FastMCP(
        name="Read Path",
        version="0.1",
    )
    reader = ReadPath(
        root_dir=root_dir,
        max_tokens=max_tokens,
        items_per_batch=items_per_batch,
        matches_per_batch=matches_per_batch,
        max_matches_per_file=max_matches_per_file,
    )

    @sub_mcp.tool(
        title="List Folder Content",
        name="list_folder_content",
        description=(
            "List unignored relative file paths under `path` in paginated batches. "
            "Use this to map project structure or inspect a subfolder when `has_more` is true. "
            "If `has_more` is true, call again with `batch + 1`."
        ),
        output_schema=FileContentPublic.model_json_schema(),
    )
    async def list_folder_content_tool(
        ctx: Context,
        path: Annotated[str, Field(description="Repo-relative path to the directory to inspect (e.g. '.' or 'src').")],
        batch: Annotated[int, Field(ge=1, description="1-indexed batch number for paginated directory listings.")] = 1,
    ) -> dict:
        """List nested unignored relative file paths under `path` in paginated, token-capped batches."""
        request = ListFolderRequest(path=path, batch=batch)
        res = await asyncio.to_thread(reader.list_folder_content, request=request)
        await ctx.info(f"Listed directory: {res.metadata.file_path} ({res.line_read})")
        return res.model_dump(exclude={"metadata"})

    @sub_mcp.tool(
        title="Read File Slice",
        name="read_file_slice",
        description=(
            "Read a 1-indexed inclusive line range from a UTF-8 file, automatically capped to the server's token budget. "
            "Prefer `search_symbol` first to locate line numbers in large code files, or start at `line_start=1` for schema files. "
            "If `has_more` is true, call again with `line_start` set to `next_line`."
        ),
        output_schema=FileContentPublic.model_json_schema(),
    )
    async def read_file_slice_tool(
        ctx: Context,
        path: Annotated[str, Field(description="Repo-relative path to the UTF-8 file to read.")],
        line_start: Annotated[int, Field(ge=1, description="1-indexed starting line number (inclusive).")] = 1,
        line_end: Annotated[
            int | None,
            Field(
                ge=1,
                description=(
                    f"Optional 1-indexed ending line number (inclusive). "
                    f"Omit (null) to read the default {settings.DEFAULT_LINES_PER_BATCH}-line batch — do NOT pass -1."
                ),
            ),
        ] = None,
    ) -> dict:
        """Read a targeted 1-indexed line range from a UTF-8 file, capped by the server's token budget."""
        request = ReadFileSliceRequest(path=path, line_start=line_start, line_end=line_end)
        res = await asyncio.to_thread(reader.read_file_slice, request=request)
        await ctx.info(f"Read slice: {res.metadata.file_path} ({res.line_read})")
        return res.model_dump(exclude={"metadata"})

    @sub_mcp.tool(
        title="Search Symbol",
        name="search_symbol",
        description=(
            "Search for a symbol (table, column, function, class, etc.) across a file or directory; "
            "use `mode='all'` for SQL strings/usages or `mode='definitions'` for declarations/DDL. "
            "Use the returned `path` and `lines` with `read_file_slice` to inspect context. "
            "If `has_more` is true, call `search_symbol` again with `batch + 1`."
        ),
        output_schema=SearchSymbolPublicResult.model_json_schema(),
    )
    async def search_symbol_tool(
        ctx: Context,
        path: Annotated[str, Field(description="Repo-relative path to a single file or directory to search within.")],
        symbol: Annotated[str, Field(min_length=1, description="Symbol name, identifier, or regex pattern to search for.")],
        mode: Annotated[
            SearchMode,
            Field(description="'all' searches code, SQL strings, and dict keys; 'definitions' filters to declarations/DDL."),
        ] = "all",
        whole_word: Annotated[
            bool,
            Field(description="Match whole identifiers only so e.g. 'id' does not match 'valid'."),
        ] = True,
        case_sensitive: Annotated[bool, Field(description="Enable case-sensitive matching when True.")] = False,
        use_regex: Annotated[
            bool,
            Field(description="Treat `symbol` as a regular expression instead of literal text."),
        ] = False,
        file_extensions: Annotated[
            list[str] | None,
            Field(description="Optional list of file extensions to include (e.g. ['.py', '.sql'])."),
        ] = None,
        batch: Annotated[int, Field(ge=1, description="1-indexed batch number for paginated search matches.")] = 1,
    ) -> dict:
        """Search for a symbol across a file or directory in paginated, token-capped batches."""
        request = SearchSymbolRequest(
            path=path,
            symbol=symbol,
            mode=mode,
            whole_word=whole_word,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
            file_extensions=file_extensions,
            batch=batch,
        )
        res = await asyncio.to_thread(reader.search_symbol, request=request)
        await ctx.info(
            f"Searched '{request.symbol}' in {res.metadata.file_path}: {res.total_matches} hits"
        )
        return res.model_dump(exclude={"metadata"})

    return sub_mcp
