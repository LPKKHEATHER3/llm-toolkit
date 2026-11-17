"""Fakes shared by the tests: a controllable clock and scripted HTTP providers."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

import httpx

from llm_toolkit.client import LLMClient, ProviderSpec, RetryPolicy, anthropic, google, openrouter

FAKE_KEYS = {
    "GEMINI_API_KEY": "AIzaFAKEgoogle0000000000000000000000",
    "OPENROUTER_API_KEY": "sk-or-v1-FAKEopenrouter000000000000000",
    "ANTHROPIC_API_KEY": "sk-ant-api03-FAKEanthropic000000000000",
}

HOSTS = {
    "google": "generativelanguage.googleapis.com",
    "openrouter": "openrouter.ai",
    "anthropic": "api.anthropic.com",
}


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


def openai_body(
    content: str | list | None = "ok",
    tool_calls: list[dict] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    cost: float | None = None,
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"message": message, "finish_reason": "stop"}], "usage": usage}


def tool_call(name: str, arguments: str | dict, call_id: str = "call_1") -> dict:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def ok(body: dict | str | None = None, status: int = 200, headers: dict | None = None) -> httpx.Response:
    if body is None:
        body = openai_body()
    if isinstance(body, str):
        return httpx.Response(status, text=body, headers=headers)
    return httpx.Response(status, json=body, headers=headers)


def err(status: int, text: str = "error", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, text=text, headers=headers)


Item = httpx.Response | Exception | Callable[[httpx.Request], httpx.Response]


class ScriptedProviders:
    """``httpx.MockTransport`` handler with a queue of scripted replies per host.

    An item is a response, an exception to raise, or a callable taking the request. ``latency``
    advances the fake clock before each reply so deadline logic can be tested without waiting.
    """

    def __init__(self, clock: FakeClock | None = None, latency: float = 0.0):
        self.queues: dict[str, deque[Item]] = defaultdict(deque)
        self.requests: list[httpx.Request] = []
        self.clock = clock
        self.latency = latency

    def add(self, provider: str, *items: Item) -> ScriptedProviders:
        self.queues[HOSTS.get(provider, provider)].extend(items)
        return self

    def count(self, provider: str) -> int:
        host = HOSTS.get(provider, provider)
        return sum(1 for r in self.requests if r.url.host == host)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.clock is not None and self.latency:
            self.clock.advance(self.latency)
        queue = self.queues[request.url.host]
        if not queue:
            raise AssertionError(f"unexpected request to {request.url}")
        item = queue.popleft()
        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, httpx.Response):
            return item(request)
        return item

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def make_client(
    script: ScriptedProviders,
    providers: list[ProviderSpec] | None = None,
    clock: FakeClock | None = None,
    env: dict[str, str] | None = None,
    **kwargs: Any,
) -> LLMClient:
    clock = clock or script.clock or FakeClock()
    env = FAKE_KEYS if env is None else env
    if providers is None:
        providers = [google(env), openrouter(env)]
    return LLMClient(
        providers,
        transport=script.transport,
        clock=clock,
        sleep=clock.sleep,
        env=env,
        retry=kwargs.pop("retry", RetryPolicy(rand=lambda: 0.0)),
        **kwargs,
    )


__all__ = [
    "FAKE_KEYS",
    "HOSTS",
    "FakeClock",
    "ScriptedProviders",
    "anthropic",
    "err",
    "google",
    "make_client",
    "ok",
    "openai_body",
    "openrouter",
    "tool_call",
]
