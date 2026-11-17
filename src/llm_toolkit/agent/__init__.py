"""Tool-calling agent loop and history trimming."""

from .history import approx_tokens, split_units, trim_history
from .loop import Agent, AgentResult, ToolCallRecord

__all__ = ["Agent", "AgentResult", "ToolCallRecord", "approx_tokens", "split_units", "trim_history"]
