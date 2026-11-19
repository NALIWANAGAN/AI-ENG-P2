import asyncio

from fastmcp import FastMCP, Context
from schema.read_path import (
    FileContentPublic,
    ListFolderRequest,
    ReadFileSliceRequest,
    SearchSymbolPublicResult,
    SearchSymbolRequest,
)
from services.read_path import ReadPath

mcp = FastMCP(
    name="Codebase and Schema Auditor",
    version="0.1"
)

reader = ReadPath()


@mcp.tool(
    title="List Folder Content",
    name="list_folder_content",
    description=(
        "List all unignored files under `request.path`. "
        "This tool is useful to analyze the project directory at a high level."
    ),
    output_schema=FileContentPublic.model_json_schema()
)
async def list_folder_content_tool(
    request: ListFolderRequest, ctx: Context
) -> dict:
    """List nested unignored relative file paths under `request.path` in paginated, token-capped batches."""
    res = await asyncio.to_thread(
        reader.list_folder_content, request=request
    )
    await ctx.info(
        f"Listed directory: {res.metadata.file_path} ({res.line_read})"
    )

    return res.model_dump(exclude={"metadata"})


@mcp.tool(
    title="Read File Slice",
    name="read_file_slice",
    description="Read a targeted 1-indexed line range from a file, capped by `request.max_tokens`.",
    output_schema=FileContentPublic.model_json_schema()
)
async def read_file_slice_tool(
    request: ReadFileSliceRequest, ctx: Context
) -> dict:
    """Read a targeted 1-indexed line range from a UTF-8 file, capped by `request.max_tokens`."""
    res = await asyncio.to_thread(
        reader.read_file_slice, request=request
    )
    await ctx.info(f"Read slice: {res.metadata.file_path} ({res.line_read})")
    return res.model_dump(exclude={"metadata"})


@mcp.tool(
    title="Search Symbol",
    name="search_symbol",
    description="""Search for a symbol (table, column, function, class, etc.) across a file or directory.
    Respects `.gitignore`, skips binary files, supports `'all'` (text/SQL/dict/code) and
    `'definitions'` (Python/SQL/JS/TS/Prisma declarations) modes, and applies batch + token caps.""",
    output_schema=SearchSymbolPublicResult.model_json_schema()
)
async def search_symbol_tool(
    request: SearchSymbolRequest, ctx: Context
) -> dict:
    """Search for a symbol across a file or directory in paginated, token-capped batches."""
    res = await asyncio.to_thread(
        reader.search_symbol, request=request
    )
    await ctx.info(f"Searched '{request.symbol}' in {res.metadata.file_path}: {res.total_matches} hits")
    return res.model_dump(exclude={"metadata"})
