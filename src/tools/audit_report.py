import asyncio
from pathlib import Path

from fastmcp import Context, FastMCP

from core.settings import settings
from schema.audit_report import AuditReportPublicResult, CreateAuditReportRequest
from services.markdown_audit_report import MarkdownAuditReport


def create_audit_report_mcp(
    output_dir: Path | str | None = None,
    *,
    max_report_bytes: int = settings.MAX_AUDIT_REPORT_BYTES,
) -> FastMCP:
    """Build a FastMCP sub-server with the Markdown audit report tool sandboxed to `output_dir`."""
    sub_mcp = FastMCP(
        name="Audit Report",
        version="0.1",
    )
    writer = MarkdownAuditReport(
        output_dir=output_dir,
        max_report_bytes=max_report_bytes,
    )

    @sub_mcp.tool(
        title="Create Audit Report",
        name="create_audit_report",
        description=(
            "Create or incrementally append a Markdown section to the current run's audit report. "
            "All turns automatically write into the same `.md` file; calling with an existing `section` name appends under that heading. "
            "Secrets and PII are automatically redacted. Set `is_final=true` only on the last call once all sections are complete."
        ),
        output_schema=AuditReportPublicResult.model_json_schema(),
    )
    async def create_audit_report_tool(
        request: CreateAuditReportRequest, ctx: Context
    ) -> dict:
        """Create or append a redacted Markdown section to the active audit report."""
        res = await asyncio.to_thread(writer.write_section, request=request)
        await ctx.info(
            f"Saved report section '{res.section_saved}' to {res.metadata.file_path} "
            f"({res.bytes_written} bytes, final={res.is_final})"
        )
        return res.model_dump(exclude={"metadata"})

    return sub_mcp
