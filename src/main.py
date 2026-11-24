import asyncio
from pathlib import Path
from fastmcp import FastMCP
from agent.harness import format_token_io_audit, run_pipeline
from clients.openai_client import client as openai_client
from core.settings import settings
from tools.audit_report import create_audit_report_mcp
from tools.read_path import create_read_path_mcp


def create_mcp_server(
    repo_path: Path | str | None = None,
    audit_path: Path | str | None = None,
) -> FastMCP:
    """Create the root FastMCP server with read and report tools bound to their sandbox directories."""
    server = FastMCP(
        name="Codebase and Schema Auditor",
        version="0.1",
    )
    server.mount(create_read_path_mcp(root_dir=repo_path))
    server.mount(create_audit_report_mcp(output_dir=audit_path))
    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    result = asyncio.run(
        run_pipeline(
            repo_path="C:/Users/imper/Documents/AI-ENG/P1",
            audit_path="C:/Users/imper/Documents/AI-ENG/nested",
            client=openai_client,
            model_name=settings.OLLAMA_MODEL_NAME,
            max_turns=None,
        )
    )
    print(format_token_io_audit(result))
