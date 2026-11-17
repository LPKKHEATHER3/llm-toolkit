"""MCP server over the shared tool registry."""

from .server import build_server, serve_stdio, to_call_result

__all__ = ["build_server", "serve_stdio", "to_call_result"]
