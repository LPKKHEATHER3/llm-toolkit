"""Start the MCP server as a subprocess over stdio and call its tools with the official client.

    python examples/mcp_stdio_demo.py      # uses --fake-backend: no key, no inventory server

Shows a backend rejection ("only 3 in stock, cannot issue 10") arriving as an is_error tool result.
"""

from __future__ import annotations

import sys

import anyio
from mcp import Client, StdioServerParameters


async def main() -> int:
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "llm_toolkit.mcp_server", "--fake-backend"]
    )
    async with Client(server) as client:
        tools = await client.list_tools()
        print("tools:", ", ".join(t.name for t in tools.tools))

        refused = await client.call_tool(
            "issue_stock", {"sku": "ELE-0002", "quantity": 10, "reason": "order"}
        )
        print(f"issue 10 x ELE-0002 -> is_error={refused.is_error}: {refused.content[0].text}")

        issued = await client.call_tool("issue_stock", {"sku": "ELE-0002", "quantity": 2, "reason": "order"})
        print(f"issue 2 x ELE-0002  -> {issued.structured_content}")

        undo = await client.call_tool(
            "undo_last_change", {"expected_label": issued.structured_content["undo_label"]}
        )
        print(f"undo               -> {undo.structured_content}")

        found = await client.call_tool("search_policies", {"query": "swollen power bank", "k": 1})
        top = found.structured_content["results"][0]
        print(f"search_policies    -> [{top['chunk_id']}] {top['text'][:80]}...")
    return 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))
