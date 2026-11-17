"""Adapters, usage log, token alarm and redaction."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import httpx
import pytest
from fakes import FAKE_KEYS, ScriptedProviders, anthropic, make_client, ok, openai_body, openrouter, tool_call

from llm_toolkit.client import ChatRequest, Message, ProviderRequestError, ToolCall, UsageLog
from llm_toolkit.redact import REDACTED, get_logger, redact

REQ = ChatRequest(messages=[Message.user("hi")])
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0LXVzZXIifQ.c2lnbmF0dXJlLXZhbHVl"


# ---- adapters -------------------------------------------------------------------------------


def test_content_parts_are_joined_as_text():
    parts = [
        {"type": "text", "text": "Hello, "},
        {"type": "image_url", "image_url": {}},
        {"type": "text", "text": "world"},
    ]
    script = ScriptedProviders().add("openrouter", ok(openai_body(parts)))
    res = make_client(script, providers=[openrouter(FAKE_KEYS)]).chat(REQ)
    assert res.message.content == "Hello, world"  # the prototype produced "[object Object]"


def test_openai_tool_calls_and_cost_are_parsed():
    body = openai_body(None, tool_calls=[tool_call("get_item", {"sku": "ELE-0001"})], cost=0.00042)
    script = ScriptedProviders().add("openrouter", ok(body))
    res = make_client(script, providers=[openrouter(FAKE_KEYS)]).chat(REQ)
    assert res.message.tool_calls == [ToolCall(id="call_1", name="get_item", arguments='{"sku": "ELE-0001"}')]
    assert res.usage.cost_usd == 0.00042
    sent = json.loads(script.requests[0].content)
    assert sent["usage"] == {"include": True}
    assert script.requests[0].headers["authorization"] == f"Bearer {FAKE_KEYS['OPENROUTER_API_KEY']}"


def test_anthropic_adapter_round_trip():
    reply = {
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_item", "input": {"sku": "ELE-0001"}},
        ],
        "usage": {"input_tokens": 100, "cache_read_input_tokens": 20, "output_tokens": 7},
        "stop_reason": "tool_use",
    }
    script = ScriptedProviders().add(
        "anthropic", ok(reply), ok({"content": [{"type": "text", "text": "Done"}], "usage": {}})
    )
    client = make_client(script, providers=[anthropic(FAKE_KEYS)])
    tools = [
        {
            "type": "function",
            "function": {"name": "get_item", "description": "d", "parameters": {"type": "object"}},
        }
    ]
    first = client.chat(
        ChatRequest(messages=[Message.system("sys"), Message.user("hi")], tools=tools, temperature=0.0)
    )
    assert first.message.tool_calls[0].name == "get_item"
    assert first.usage.prompt_tokens == 120 and first.usage.completion_tokens == 7

    sent = json.loads(script.requests[0].content)
    headers = script.requests[0].headers
    assert headers["anthropic-version"] == "2026-10-21"
    assert headers["x-api-key"] == FAKE_KEYS["ANTHROPIC_API_KEY"]
    assert "temperature" not in sent, "API version 2026-10-21 rejects sampling parameters"
    assert sent["system"] == "sys"
    assert sent["tools"][0]["input_schema"] == {"type": "object"}

    history = [
        Message.system("sys"),
        Message.user("hi"),
        first.message,
        Message.tool("toolu_1", '{"ok": false, "error": "not found"}'),
    ]
    client.chat(ChatRequest(messages=history, tools=tools))
    second = json.loads(script.requests[1].content)
    assert second["messages"][1]["content"][0]["type"] == "thinking", "assistant blocks replayed verbatim"
    result = second["messages"][2]["content"][0]
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": '{"ok": false, "error": "not found"}',
        "is_error": True,
    }


def test_provider_tool_call_fields_are_replayed_to_the_same_provider_only():
    """Found in a live run: Gemini answered round 2 with HTTP 400 'Function call is missing a
    thought_signature' because extra tool-call fields were dropped. They are now replayed verbatim
    to the provider that produced them, and not sent to any other provider."""
    signed = tool_call("get_item", {"sku": "ELE-0001"})
    signed["extra_content"] = {"google": {"thought_signature": "SIG-A"}}
    body = openai_body(None, tool_calls=[signed])
    body["choices"][0]["message"]["reasoning_details"] = [{"type": "reasoning.encrypted", "data": "R"}]
    script = ScriptedProviders().add("google", ok(body), ok()).add("openrouter", ok())
    client = make_client(script)
    first = client.chat(REQ)
    history = [Message.user("hi"), first.message, Message.tool("call_1", '{"ok": true}')]
    client.chat(ChatRequest(messages=history))
    replay = json.loads(script.requests[1].content)["messages"][1]
    assert replay["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG-A"}}
    assert replay["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "R"}]

    other = make_client(script, providers=[openrouter(FAKE_KEYS)])
    other.chat(ChatRequest(messages=history))
    to_openrouter = json.loads(script.requests[2].content)["messages"][1]
    assert "extra_content" not in to_openrouter["tool_calls"][0]
    assert "reasoning_details" not in to_openrouter


# ---- usage log ------------------------------------------------------------------------------


def test_usage_log_is_valid_jsonl_under_concurrent_writes(tmp_path: Path):
    log = UsageLog(tmp_path / "usage.jsonl")
    threads = [
        threading.Thread(
            target=lambda n=n: [
                log.record(
                    provider=f"p{n}",
                    model="m" * 200,
                    prompt_tokens=i,
                    completion_tokens=1,
                    cost_usd=None,
                    latency_ms=1.0,
                )
                for i in range(200)
            ]
        )
        for n in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1600
    records = [json.loads(line) for line in lines]
    assert {r["provider"] for r in records} == {f"p{n}" for n in range(8)}


def test_successful_call_is_recorded_in_usage_log(tmp_path: Path):
    script = ScriptedProviders().add(
        "openrouter", ok(openai_body("x", prompt_tokens=11, completion_tokens=3, cost=0.001))
    )
    client = make_client(script, providers=[openrouter(FAKE_KEYS)], usage_log=UsageLog(tmp_path / "u.jsonl"))
    client.chat(REQ)
    (rec,) = [json.loads(x) for x in (tmp_path / "u.jsonl").read_text(encoding="utf-8").splitlines()]
    assert (rec["provider"], rec["prompt_tokens"], rec["completion_tokens"], rec["cost_usd"]) == (
        "openrouter",
        11,
        3,
        0.001,
    )


def test_prompt_token_alarm_logs_warning(caplog):
    script = ScriptedProviders().add("openrouter", ok(openai_body("x", prompt_tokens=13_444)))
    client = make_client(script, providers=[openrouter(FAKE_KEYS)])
    with caplog.at_level(logging.WARNING, logger="llm_toolkit"):
        client.chat(REQ)
    assert any("prompt_tokens=13444" in r.getMessage() for r in caplog.records)


# ---- redaction ---------------------------------------------------------------------------------


def test_logs_redact_keys_bearer_and_jwt(caplog, monkeypatch):
    plain_key = "plainkeyvalue-without-known-prefix-123"
    monkeypatch.setenv("ANTHROPIC_API_KEY", plain_key)
    samples = [
        f"key in text {plain_key}",
        "Authorization: Bearer abc.def-123",
        '{"x-api-key": "sk-ant-api03-SECRETSECRET"}',
        f"token={JWT}",
        "google key AIzaSyA1234567890abcdefghijklmnop",
        "openrouter sk-or-v1-0123456789abcdef",
    ]
    logger = get_logger("llm_toolkit.test")
    with caplog.at_level(logging.INFO, logger="llm_toolkit"):
        for s in samples:
            logger.info("upstream said: %s", s)
    text = "\n".join(r.getMessage() for r in caplog.records)
    for secret in (plain_key, "abc.def-123", "SECRETSECRET", JWT, "AIzaSyA1234567890", "0123456789abcdef"):
        assert secret not in text
    assert text.count(REDACTED) >= len(samples)


def test_redact_keeps_ordinary_text():
    assert redact("issue 3 units of ELE-0007, reason: sale") == "issue 3 units of ELE-0007, reason: sale"


def test_usage_log_never_contains_key_values(tmp_path: Path, monkeypatch):
    for k, v in FAKE_KEYS.items():
        monkeypatch.setenv(k, v)
    log = UsageLog(tmp_path / "u.jsonl")
    log.record(
        provider="x",
        model=FAKE_KEYS["GEMINI_API_KEY"],
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=None,
        latency_ms=1,
        extra={"note": f"Bearer {FAKE_KEYS['OPENROUTER_API_KEY']}", "jwt": JWT},
    )
    content = (tmp_path / "u.jsonl").read_text(encoding="utf-8")
    for v in (*FAKE_KEYS.values(), JWT):
        assert v not in content


def test_request_error_message_does_not_echo_the_key():
    echo = f"invalid header: Authorization: Bearer {FAKE_KEYS['OPENROUTER_API_KEY']}"
    script = ScriptedProviders().add("openrouter", httpx.Response(400, text=echo))
    with pytest.raises(ProviderRequestError) as exc:
        make_client(script, providers=[openrouter(FAKE_KEYS)]).chat(REQ)
    assert FAKE_KEYS["OPENROUTER_API_KEY"] not in str(exc.value)
    assert all(FAKE_KEYS["OPENROUTER_API_KEY"] not in a.detail for a in exc.value.attempts)
