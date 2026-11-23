import asyncio
from pathlib import Path

from fastmcp import Context, FastMCP

from core.settings import settings
from schema.read_path import (
    FileContentPublic,
    ListFolderRequest,
    ReadFileSliceRequest,
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
            "List unignored relative file paths under `request.path` in paginated batches. "
            "Use this first when starting a repository audit to map project structure and locate schema/code files. "
            "If `has_more` is true, call again with `batch + 1`."
        ),
        output_schema=FileContentPublic.model_json_schema(),
    )
    async def list_folder_content_tool(
        request: ListFolderRequest, ctx: Context
    ) -> dict:
        """List nested unignored relative file paths under `request.path` in paginated, token-capped batches."""
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
        request: ReadFileSliceRequest, ctx: Context
    ) -> dict:
        """Read a targeted 1-indexed line range from a UTF-8 file, capped by the server's token budget."""
        res = await asyncio.to_thread(reader.read_file_slice, request=request)
        await ctx.info(f"Read slice: {res.metadata.file_path} ({res.line_read})")
        return res.model_dump(exclude={"metadata"})

    @sub_mcp.tool(
        title="Search Symbol",
        name="search_symbol",
        description=(
            "Search for a symbol (table, column, function, class, etc.) across a file or directory; "
            "use `mode='all'` for SQL strings/usages or `mode='definitions'` for declarations/DDL. "
            "Use the returned `path` and `lines` with `read_file_slice` (passing a `line_start`..`line_end` window around each hit) to inspect context. "
            "If `has_more` is true, call `search_symbol` again with `batch + 1`."
        ),
        output_schema=SearchSymbolPublicResult.model_json_schema(),
    )
    async def search_symbol_tool(
        request: SearchSymbolRequest, ctx: Context
    ) -> dict:
        """Search for a symbol across a file or directory in paginated, token-capped batches."""
        res = await asyncio.to_thread(reader.search_symbol, request=request)
        await ctx.info(
            f"Searched '{request.symbol}' in {res.metadata.file_path}: {res.total_matches} hits"
        )
        return res.model_dump(exclude={"metadata"})

    return sub_mcp
