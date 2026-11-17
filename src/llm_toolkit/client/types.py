"""Provider-neutral message model and client errors.

Internally every conversation is kept in the OpenAI chat shape (``tool_calls`` on the assistant
message, one ``role="tool"`` message per result). Adapters translate to and from each provider's
wire format, so the agent loop and history trimming only have to reason about one shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: str = Field(description="Raw JSON text exactly as the model produced it")


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    # Opaque provider data that must be replayed verbatim (e.g. Anthropic thinking blocks).
    provider_state: dict[str, Any] | None = None

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", content=text)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=text)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(role="tool", tool_call_id=tool_call_id, content=content)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None


AttemptOutcome = Literal[
    "ok",
    "retryable",  # 429 / 5xx / timeout / transport error
    "gateway",  # 200 that is not a usable answer (non-JSON, empty, disguised gateway error)
    "provider_error",  # 401 / 402 / 403 / 404: specific to this provider, fail over
    "request_error",  # 400 / 422 / other 4xx: the request itself is wrong
    "skipped_no_key",
    "skipped_cooldown",
]


class Attempt(BaseModel):
    provider: str
    model: str
    outcome: AttemptOutcome
    status: int | None = None
    detail: str = ""
    waited_s: float = 0.0
    elapsed_s: float = 0.0


class ChatRequest(BaseModel):
    messages: list[Message]
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = 0.0
    max_tokens: int = 2048
    deadline_s: float = 60.0


class ChatResponse(BaseModel):
    message: Message
    usage: Usage
    provider: str
    model: str
    latency_ms: float
    attempts: list[Attempt]


# ---- errors ------------------------------------------------------------------------------


class LLMError(Exception):
    """Base class. ``attempts`` lists every try that was made, so callers can classify the failure."""

    def __init__(self, message: str, attempts: list[Attempt] | None = None):
        super().__init__(message)
        self.attempts = list(attempts or [])


class ProviderRequestError(LLMError):
    """The request itself was rejected (400, 422, ...). Not retried, not failed over."""

    def __init__(self, message: str, *, status: int, provider: str, attempts: list[Attempt]):
        super().__init__(message, attempts)
        self.status = status
        self.provider = provider


class AllProvidersFailed(LLMError):
    """Every provider was skipped or failed (transient errors, gateway errors, auth/model errors)."""


class DeadlineExceeded(LLMError):
    """The total per-request deadline ran out before any provider answered."""


class RequestBudgetExhausted(LLMError):
    """The client's ``max_requests`` cap was reached; no further HTTP requests are sent."""


class ResponseFormatError(Exception):
    """Raised by an adapter when a 2xx body cannot be turned into a message."""
