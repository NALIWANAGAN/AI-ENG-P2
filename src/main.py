import asyncio
from typing import Any

from fastmcp import Client, FastMCP

from clients.openai_client import client as openai_client
from core.settings import settings
from tools.read_path import mcp as read_path_mcp

mcp = FastMCP(
    name="Codebase and Schema Auditor",
    version="0.1",
)

mcp.mount(read_path_mcp)


def _sanitize_schema(node: Any) -> Any:
    """Strip JSON Schema keywords rejected by Gemini's OpenAI-compatible tool validator."""
    if isinstance(node, list):
        return [_sanitize_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"additionalProperties", "title", "format"}:
            continue
        cleaned[key] = _sanitize_schema(value)

    # Collapse `anyOf: [{"type": "string"}, {"type": "string"}]` (from `Path | str`) -> `{"type": "string"}`
    if "anyOf" in cleaned and isinstance(cleaned["anyOf"], list):
        non_null = [item for item in cleaned["anyOf"] if item != {"type": "null"}]
        unique_types = {item.get("type") for item in non_null if isinstance(item, dict)}
        if len(unique_types) == 1 and len(non_null) == len(cleaned["anyOf"]):
            cleaned.pop("anyOf")
            cleaned.update(non_null[0])

    return cleaned


async def get_agent_tools(mcp_client: Client) -> list[dict[str, Any]]:
    """Build agent tool definitions compatible with OpenAI & Gemini chat completions."""
    mcp_tools = await mcp_client.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": _sanitize_schema(tool.input_schema),
            },
        }
        for tool in mcp_tools
    ]


async def run_single_turn(user_prompt: str) -> None:
    async with Client(mcp) as mcp_client:
        tools = await get_agent_tools(mcp_client)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a read-only codebase and schema auditor. "
                    "Always pass repo-relative paths like '.' or 'src' (never guess absolute paths). "
                    "When you need multiple independent lookups, call multiple tools in the same turn."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        response = await openai_client.chat.completions.create(
            model=settings.OLLAMA_MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message

        if assistant_msg.tool_calls:
            print(f"Total parallel tool calls in 1 turn: {len(assistant_msg.tool_calls)}")
            for tc in assistant_msg.tool_calls:
                print(f"ID: {tc.id} | Tool: {tc.function.name} | Args: {tc.function.arguments}")
        else:
            print("Final text:", assistant_msg.content)


if __name__ == "__main__":
    asyncio.run(
        run_single_turn(
            "Search for ReadPath and Settings inside src, and also list files inside tests."
        )
    )
