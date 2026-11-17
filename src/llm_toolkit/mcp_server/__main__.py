"""Run the MCP server over stdio.

    INVENTORY_API_URL=http://127.0.0.1:8000 python -m llm_toolkit.mcp_server
    python -m llm_toolkit.mcp_server --fake-backend      # in-memory fictional inventory, no server needed

Logs go to stderr (stdout carries the MCP protocol).
"""

from __future__ import annotations

import argparse
import logging
import sys

import anyio

from ..rag import default_index
from ..tools import InventoryAPI, build_default_registry
from ..tools.fake_inventory import FakeInventoryBackend
from .server import build_server, serve_stdio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_toolkit.mcp_server")
    parser.add_argument("--fake-backend", action="store_true", help="use the in-memory fictional inventory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    if args.fake_backend:
        api = InventoryAPI("http://inventory.fake", transport=FakeInventoryBackend().transport)
    else:
        api = InventoryAPI()
    registry = build_default_registry(api, default_index())
    anyio.run(serve_stdio, build_server(registry))
    return 0


if __name__ == "__main__":
    sys.exit(main())
