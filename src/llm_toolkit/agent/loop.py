"""A bounded tool-calling loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from ..client.types import (
    AllProvidersFailed,
    Attempt,
    ChatRequest,
    ChatResponse,
    DeadlineExceeded,
    LLMError,
    Message,
    Usage,
)
from ..redact import get_logger
from ..tools.registry import ToolRegistry
from .history import approx_tokens, trim_history

log = get_logger(__name__)


class ChatClient(Protocol):
    def chat(self, req: ChatRequest) -> ChatResponse: ...


class ToolCallRecord(BaseModel):
    round: int
    name: str
    arguments: str
    ok: bool
    error: str | None = None
    data: Any = None


class AgentResult(BaseModel):
    messages: list[Message]
    final_text: str
    stop_reason: Literal["answer", "max_rounds", "llm_error"]
    rounds: int
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)
    latency_ms: list[float] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)
    error: str | None = None
    error_type: str | None = None
    infra_failure: bool = False  # True when the LLM call failed for reasons outside the model
    attempts: list[Attempt] = Field(default_factory=list)


class Agent:
    def __init__(
        self,
        client: ChatClient,
        registry: ToolRegistry,
        *,
        system_prompt: str,
        max_rounds: int = 5,
        history_token_budget: int = 6000,
        temperature: float | None = 0.0,
        max_tokens: int = 2048,
        deadline_s: float = 60.0,
    ):
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.history_token_budget = history_token_budget
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.deadline_s = deadline_s

    def run(self, history: Sequence[Message], user_text: str) -> AgentResult:
        messages: list[Message] = [*history, Message.user(user_text)]
        # pydantic copies lists on validation, so ``result.messages`` is re-pointed before returning.
        result = AgentResult(messages=[], final_text="", stop_reason="answer", rounds=0)
        schemas = self.registry.schemas("openai")
        system = Message.system(self.system_prompt)
        budget = max(0, self.history_token_budget - approx_tokens(system))

        for round_no in range(self.max_rounds):
            window = trim_history(messages, budget)
            req = ChatRequest(
                messages=[system, *window],
                tools=schemas or None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                deadline_s=self.deadline_s,
            )
            try:
                resp = self.client.chat(req)
            except LLMError as e:
                log.warning("LLM call failed in round %d: %s", round_no + 1, e)
                result.stop_reason = "llm_error"
                result.error = str(e)
                result.error_type = type(e).__name__
                result.infra_failure = isinstance(e, AllProvidersFailed | DeadlineExceeded)
                result.attempts = e.attempts
                result.messages = messages
                return result
            result.rounds = round_no + 1
            result.usage.append(resp.usage)
            result.latency_ms.append(resp.latency_ms)
            result.providers.append(resp.provider)
            messages.append(resp.message)

            if not resp.message.tool_calls:
                result.final_text = resp.message.content or ""
                result.messages = messages
                return result

            for tc in resp.message.tool_calls:
                out = self.registry.call(tc.name, tc.arguments)
                result.tool_calls.append(
                    ToolCallRecord(
                        round=round_no + 1,
                        name=tc.name,
                        arguments=tc.arguments,
                        ok=out.ok,
                        error=out.error,
                        data=out.data,
                    )
                )
                messages.append(Message.tool(tc.id, out.to_content()))

        # Round limit reached right after tool results. Close the turn with an assistant message so
        # the stored history is valid and the next user turn does not follow dangling tool output.
        note = (
            f"I stopped after {self.max_rounds} tool rounds without reaching an answer. "
            "Please narrow the request or ask again."
        )
        messages.append(Message.assistant(note))
        result.final_text = note
        result.stop_reason = "max_rounds"
        result.messages = messages
        return result
