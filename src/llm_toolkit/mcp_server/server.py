"""MCP server exposing the shared tool registry (inventory tools + ``search_policies``).

Built on the official SDK's low-level ``Server`` so the input schemas are exactly the registry's
pydantic-generated schemas and argument validation happens in one place (the registry). A failed
tool call (bad arguments, backend rejection such as "only 4 in stock, cannot issue 50", backend
down) is returned as a tool result with ``is_error=True`` and the reason as text, so the model can
read it and recover instead of the client seeing a protocol error.
"""

from __future__ import annotations

import json

import anyio
import mcp_types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..tools import ToolRegistry, ToolResult

INSTRUCTIONS = (
    "Inventory tools for a small shop plus a search over its (fictional) policy documents. "
    "Look items up before changing stock, pass the undo_label from issue_stock to undo_last_change, "
    "and quote search_policies excerpts word for word."
)


def to_call_result(result: ToolResult) -> types.CallToolResult:
    if result.ok:
        text = json.dumps(result.data, ensure_ascii=False, default=str)
        structured = result.data if isinstance(result.data, dict) else None
        return types.CallToolResult(content=[types.TextContent(text=text)], structured_content=structured)
    return types.CallToolResult(
        content=[types.TextContent(text=result.error or "tool failed")], is_error=True
    )


def build_server(registry: ToolRegistry, name: str = "llm-toolkit") -> Server:
    async def list_tools(ctx, params) -> types.ListToolsResult:  # type: ignore[no-untyped-def]
        return types.ListToolsResult(
            tools=[
                types.Tool(name=t.name, description=t.description, input_schema=t.mcp_input_schema())
                for t in registry.tools
            ]
        )

    async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:  # type: ignore[no-untyped-def]
        # MCP sends arguments as an already-parsed object; an omitted `arguments` member means "{}"
        # in the protocol. Validation (types, unknown fields, missing fields) is the registry's job.
        arguments = params.arguments if params.arguments is not None else {}
        # Tool handlers do blocking HTTP; keep them off the event loop.
        result = await anyio.to_thread.run_sync(registry.call, params.name, arguments)
        return to_call_result(result)

    return Server(
        name, version="0.1.0", instructions=INSTRUCTIONS, on_list_tools=list_tools, on_call_tool=call_tool
    )


async def serve_stdio(server: Server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
