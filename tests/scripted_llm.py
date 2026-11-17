"""A scripted stand-in for ``LLMClient`` used by agent, RAG and eval tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from llm_toolkit.client import ChatRequest, ChatResponse, LLMError, Message, ToolCall, Usage

Reply = Message | LLMError | Callable[[ChatRequest], Message]


def call(name: str, arguments: str | dict[str, Any], call_id: str | None = None) -> ToolCall:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=args)


def calls(*tool_calls: ToolCall, text: str | None = None) -> Message:
    return Message(role="assistant", content=text, tool_calls=list(tool_calls))


def say(text: str) -> Message:
    return Message.assistant(text)


class ScriptedLLM:
    def __init__(
        self, *replies: Reply, provider: str = "fake", model: str = "fake-model", latency_ms: float = 5.0
    ):
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []
        self.provider = provider
        self.model = model
        self.latency_ms = latency_ms

    def chat(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, LLMError):
            raise reply
        msg = reply(req) if callable(reply) and not isinstance(reply, Message) else reply
        return ChatResponse(
            message=msg,
            usage=Usage(prompt_tokens=100, completion_tokens=10),
            provider=self.provider,
            model=self.model,
            latency_ms=self.latency_ms,
            attempts=[],
        )
