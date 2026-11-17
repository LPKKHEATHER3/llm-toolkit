"""Tool definitions shared by the agent, the eval harness and the MCP server.

``build_default_registry`` is the only factory: everything that needs the toolkit's tools calls it,
so there is never a second copy of a tool that could drift.
"""

from __future__ import annotations

from ..rag.bm25 import Retriever
from .inventory import InventoryAPI, inventory_tools, undo_label
from .policies import policy_tools
from .registry import Tool, ToolArgs, ToolError, ToolRegistry, ToolResult, provided

STOCK_AGENT_PROMPT = """You help staff of a small shop manage inventory and answer policy questions.

- Use the tools for every fact about stock: never guess a quantity or a SKU.
- Before issuing stock, make sure you have the right SKU (look it up if the user gave a name).
- If a tool returns an error, tell the user what the error says and what they can do next.
  Never claim that an action succeeded when the tool reported an error.
- To undo, pass the undo_label you got from issue_stock as expected_label.
- For policy questions, call search_policies and quote the policy text you rely on.
- Answer briefly."""


def build_default_registry(inventory: InventoryAPI, retriever: Retriever | None = None) -> ToolRegistry:
    registry = ToolRegistry(inventory_tools(inventory))
    if retriever is None:
        from ..rag import default_index

        retriever = default_index()
    for tool in policy_tools(retriever):
        registry.register(tool)
    return registry


__all__ = [
    "STOCK_AGENT_PROMPT",
    "InventoryAPI",
    "Tool",
    "ToolArgs",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "provided",
    "undo_label",
]
