"""Wire-format adapters: internal message model <-> provider request/response JSON."""

from __future__ import annotations

import json
from typing import Any, Protocol

from .providers import ProviderSpec
from .types import ChatRequest, Message, ResponseFormatError, ToolCall, Usage

ANTHROPIC_VERSION = "2026-10-21"


class Adapter(Protocol):
    def build(self, spec: ProviderSpec, req: ChatRequest, key: str) -> tuple[str, dict[str, str], dict]: ...

    def parse(self, spec: ProviderSpec, data: Any) -> tuple[Message, Usage]: ...


def _text_from_parts(content: Any) -> str | None:
    """Accept a plain string or a list of content parts; join the text parts."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict)
            and p.get("type") in ("text", "output_text")
            and isinstance(p.get("text"), str)
        ]
        return "".join(texts) if texts else None
    raise ResponseFormatError(f"unexpected content type {type(content).__name__}")


def _as_number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


# ---- OpenAI-compatible (OpenRouter, Google AI Studio) -------------------------------------

# Message-level fields that some providers need back within a turn (OpenRouter returns
# reasoning_details for Gemini tool use; Gemini may attach extra_content). Replayed only to the same
# provider that produced them.
REPLAYED_MESSAGE_FIELDS = ("reasoning_details", "extra_content")


class OpenAICompatAdapter:
    def build(self, spec: ProviderSpec, req: ChatRequest, key: str) -> tuple[str, dict[str, str], dict]:
        messages: list[dict[str, Any]] = []
        for m in req.messages:
            if m.role == "assistant":
                out: dict[str, Any] = {"role": "assistant", "content": m.content}
                state = m.provider_state or {}
                same_provider = state.get("kind") == "openai" and state.get("provider") == spec.name
                raw_calls = state.get("tool_calls") if same_provider else None
                if (
                    m.tool_calls
                    and raw_calls
                    and [c.get("id") for c in raw_calls] == [tc.id for tc in m.tool_calls]
                ):
                    # Replay the provider's own tool-call objects verbatim: Gemini attaches a thought
                    # signature to them and rejects the next request (HTTP 400) if it is missing.
                    out["tool_calls"] = raw_calls
                elif m.tool_calls:
                    out["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in m.tool_calls
                    ]
                if same_provider:
                    out.update(state.get("extra") or {})
                messages.append(out)
            elif m.role == "tool":
                messages.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
            else:
                messages.append({"role": m.role, "content": m.content or ""})
        body: dict[str, Any] = {"model": spec.model, "messages": messages, "max_tokens": req.max_tokens}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.tools:
            body["tools"] = req.tools
            body["tool_choice"] = "auto"
        body.update(spec.extra_body)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **spec.extra_headers}
        return f"{spec.base_url}/chat/completions", headers, body

    def parse(self, spec: ProviderSpec, data: Any) -> tuple[Message, Usage]:
        if not isinstance(data, dict):
            raise ResponseFormatError("response body is not a JSON object")
        if "error" in data and not data.get("choices"):
            raise ResponseFormatError(f"error object in a 2xx body: {str(data['error'])[:200]}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ResponseFormatError("response has no choices")
        raw = choices[0].get("message")
        if not isinstance(raw, dict):
            raise ResponseFormatError("response has no message")
        tool_calls: list[ToolCall] = []
        raw_calls: list[dict[str, Any]] = []
        for i, tc in enumerate(raw.get("tool_calls") or []):
            tc = tc if isinstance(tc, dict) else {}
            fn = tc.get("function") or {}
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)  # some OpenAI-compatible endpoints send an object
            call_id = str(tc.get("id") or f"call_{i}")
            tool_calls.append(ToolCall(id=call_id, name=str(fn.get("name", "")), arguments=args))
            # Keep every field the provider sent (e.g. a thought signature), normalised id/arguments.
            raw_calls.append({**tc, "id": call_id, "type": "function", "function": {**fn, "arguments": args}})
        extra = {k: raw[k] for k in REPLAYED_MESSAGE_FIELDS if raw.get(k) is not None}
        state = {"provider": spec.name, "kind": "openai", "tool_calls": raw_calls, "extra": extra}
        msg = Message(
            role="assistant",
            content=_text_from_parts(raw.get("content")),
            tool_calls=tool_calls,
            provider_state=state if (raw_calls or extra) else None,
        )
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            cost_usd=_as_number(u.get("cost")),
        )
        return msg, usage


# ---- Anthropic Messages API ------------------------------------------------------------


def _tool_input(arguments: str) -> dict[str, Any]:
    """Anthropic needs an object. After a failover the arguments may come from another provider and
    be malformed; keep the raw text instead of inventing an empty object."""
    try:
        value = json.loads(arguments) if arguments else {}
    except ValueError:
        return {"_unparsed_arguments": arguments}
    return value if isinstance(value, dict) else {"_value": value}


def _is_error_result(content: str | None) -> bool:
    try:
        data = json.loads(content or "")
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("ok") is False


class AnthropicAdapter:
    def build(self, spec: ProviderSpec, req: ChatRequest, key: str) -> tuple[str, dict[str, str], dict]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if pending_results:
                messages.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in req.messages:
            if m.role == "system":
                system_parts.append(m.content or "")
                continue
            if m.role == "tool":
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content or "",
                }
                if _is_error_result(m.content):
                    block["is_error"] = True
                pending_results.append(block)
                continue
            flush_results()
            if m.role == "user":
                messages.append({"role": "user", "content": m.content or ""})
                continue
            # assistant: replay the provider's own blocks verbatim when we have them (thinking blocks
            # must come back unchanged), otherwise build them from the internal model.
            state = m.provider_state or {}
            if state.get("provider") == "anthropic" and isinstance(state.get("content"), list):
                blocks = state["content"]
            else:
                blocks = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": _tool_input(tc.arguments)}
                    )
            messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        flush_results()

        body: dict[str, Any] = {"model": spec.model, "max_tokens": req.max_tokens, "messages": messages}
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        # API version 2026-10-21 rejects temperature/top_p/top_k on every model, so they are never sent.
        if req.tools:
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {"type": "object"}),
                }
                for t in req.tools
            ]
        body.update(spec.extra_body)
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            **spec.extra_headers,
        }
        return f"{spec.base_url}/messages", headers, body

    def parse(self, spec: ProviderSpec, data: Any) -> tuple[Message, Usage]:
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            raise ResponseFormatError("response has no content list")
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "")),
                        arguments=json.dumps(block.get("input", {}), ensure_ascii=False),
                    )
                )
        msg = Message(
            role="assistant",
            content="".join(texts) or None,
            tool_calls=tool_calls,
            provider_state={"provider": "anthropic", "content": data["content"]},
        )
        u = data.get("usage") or {}
        prompt = sum(
            int(u.get(k) or 0)
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        return msg, Usage(prompt_tokens=prompt, completion_tokens=int(u.get("output_tokens") or 0))


ADAPTERS: dict[str, Adapter] = {"openai": OpenAICompatAdapter(), "anthropic": AnthropicAdapter()}
