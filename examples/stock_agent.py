"""An agent that manages stock and answers policy questions with the shared tools.

    python examples/stock_agent.py --offline
        scripted model + in-memory fictional inventory (no key, no server)
    python examples/stock_agent.py --fake-backend "Sell 10 USB-C chargers"
        real model (needs a provider key) + in-memory fictional inventory
    INVENTORY_API_URL=http://127.0.0.1:8000 python examples/stock_agent.py "Which items are low on stock?"
        real model + a running inventory-manager

The same tools are exposed over MCP by ``python -m llm_toolkit.mcp_server``.
"""

from __future__ import annotations

import argparse
import json
import sys

from llm_toolkit.agent import Agent
from llm_toolkit.client import (
    ChatRequest,
    ChatResponse,
    LLMClient,
    Message,
    ToolCall,
    Usage,
    default_providers,
)
from llm_toolkit.rag import default_index
from llm_toolkit.tools import STOCK_AGENT_PROMPT, InventoryAPI, build_default_registry
from llm_toolkit.tools.fake_inventory import FakeInventoryBackend

OFFLINE_QUESTION = "Issue 10 USB-C Charger 20W (ELE-0002) for a school order."


class OfflineModel:
    """Scripted stand-in: tries to issue 10, reads the backend's refusal, issues what is available,
    then answers from the tool result. Shows the error path end to end without a network."""

    def chat(self, req: ChatRequest) -> ChatResponse:
        tool_msgs = [m for m in req.messages if m.role == "tool"]
        if not tool_msgs:
            msg = self._call(
                "issue_stock", {"sku": "ELE-0002", "quantity": 10, "reason": "school order"}, "c1"
            )
        elif len(tool_msgs) == 1:
            error = json.loads(tool_msgs[0].content).get("error", "")
            available = int(error.split()[1]) if error.startswith("only ") else 0
            msg = self._call(
                "issue_stock", {"sku": "ELE-0002", "quantity": available, "reason": "school order"}, "c2"
            )
        else:
            result = json.loads(tool_msgs[-1].content)
            msg = Message.assistant(
                f"Only part of the order could be filled: issued {result['data']['issued']}, "
                f"{result['data']['qty_after']} left. Undo label: {result['data']['undo_label']}."
            )
        return ChatResponse(
            message=msg, usage=Usage(), provider="offline", model="scripted", latency_ms=0.0, attempts=[]
        )

    @staticmethod
    def _call(name: str, args: dict, call_id: str) -> Message:
        return Message(
            role="assistant", tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(args))]
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=OFFLINE_QUESTION)
    parser.add_argument("--offline", action="store_true", help="scripted model + fake inventory")
    parser.add_argument("--fake-backend", action="store_true", help="real model + fake inventory")
    args = parser.parse_args()

    backend = FakeInventoryBackend() if (args.offline or args.fake_backend) else None
    api = InventoryAPI("http://inventory.fake", transport=backend.transport) if backend else InventoryAPI()
    client = OfflineModel() if args.offline else LLMClient(default_providers())
    agent = Agent(client, build_default_registry(api, default_index()), system_prompt=STOCK_AGENT_PROMPT)

    result = agent.run([], args.question)
    print(f"user: {args.question}")
    for call in result.tool_calls:
        status = "ok" if call.ok else f"ERROR: {call.error}"
        print(f"  tool {call.name}({call.arguments}) -> {status}")
    print(f"agent ({result.stop_reason}): {result.final_text or result.error}")
    if backend:
        print(f"backend ELE-0002 qty now: {backend.items['ELE-0002']['qty']}")
    return 0 if result.stop_reason != "llm_error" else 1


if __name__ == "__main__":
    sys.exit(main())
