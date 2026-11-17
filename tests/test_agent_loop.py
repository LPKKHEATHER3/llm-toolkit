"""The agent loop with a scripted model."""

from __future__ import annotations

import json

from pydantic import Field
from scripted_llm import ScriptedLLM, call, calls, say

from llm_toolkit.agent import Agent, split_units
from llm_toolkit.client import AllProvidersFailed, Attempt, Message, ProviderRequestError
from llm_toolkit.tools import Tool, ToolArgs, ToolRegistry


class GetArgs(ToolArgs):
    sku: str = Field(pattern=r"^[A-Z]{3}-\d{4}$")


def get_item(args: GetArgs) -> dict:
    if args.sku == "BAD-0000":
        raise RuntimeError("backend exploded")
    return {"sku": args.sku, "qty": 7}


REGISTRY = ToolRegistry([Tool("get_item", "Look up one item.", GetArgs, get_item)])


def agent(llm: ScriptedLLM, **kw) -> Agent:
    return Agent(llm, REGISTRY, system_prompt="You manage stock.", **kw)


def tool_payloads(messages: list[Message]) -> list[dict]:
    return [json.loads(m.content) for m in messages if m.role == "tool"]


def assert_history_valid(messages: list[Message]) -> None:
    """Every tool_calls message is followed by all of its replies and there are no orphans."""
    units = split_units(messages)
    assert [m for u in units for m in u] == [m for m in messages if m.role != "system"]


def test_answer_after_one_tool_round():
    llm = ScriptedLLM(calls(call("get_item", {"sku": "ELE-0001"})), say("There are 7."))
    res = agent(llm).run([], "How many ELE-0001?")
    assert res.stop_reason == "answer" and res.final_text == "There are 7."
    assert res.rounds == 2
    assert res.tool_calls[0].ok and res.tool_calls[0].data == {"sku": "ELE-0001", "qty": 7}
    first = llm.requests[0]
    assert first.messages[0].role == "system" and first.tools[0]["function"]["name"] == "get_item"
    assert_history_valid(res.messages)


def test_malformed_tool_json_is_reported_to_model():
    llm = ScriptedLLM(calls(call("get_item", '{"sku": "ELE-0001"')), say("Sorry, retrying."))
    res = agent(llm).run([], "check ELE-0001")
    (payload,) = tool_payloads(res.messages)
    assert payload["ok"] is False and "not valid JSON" in payload["error"]
    # The model sees the error on its next request, not an empty-object result.
    tool_msg = llm.requests[1].messages[-1]
    assert tool_msg.role == "tool" and "not valid JSON" in tool_msg.content


def test_schema_error_reaches_model():
    llm = ScriptedLLM(calls(call("get_item", {"sku": "ELE-0001", "units": 2})), say("ok"))
    res = agent(llm).run([], "x")
    assert "units: unknown field" in tool_payloads(res.messages)[0]["error"]


def test_tool_exception_becomes_error_for_model():
    llm = ScriptedLLM(calls(call("get_item", {"sku": "BAD-0000"})), say("The backend failed."))
    res = agent(llm).run([], "x")
    assert tool_payloads(res.messages)[0] == {"ok": False, "error": "RuntimeError: backend exploded"}
    assert res.stop_reason == "answer"


def test_parallel_tool_calls_each_get_a_reply():
    llm = ScriptedLLM(
        calls(call("get_item", {"sku": "ELE-0001"}, "a"), call("get_item", {"sku": "ELE-0002"}, "b")),
        say("done"),
    )
    res = agent(llm).run([], "x")
    assert [m.tool_call_id for m in res.messages if m.role == "tool"] == ["a", "b"]


def test_round_limit_leaves_history_valid():
    loop_forever = [calls(call("get_item", {"sku": "ELE-0001"}, f"c{i}")) for i in range(3)]
    llm = ScriptedLLM(*loop_forever, say("next turn answer"))
    a = agent(llm, max_rounds=3)
    res = a.run([], "keep checking")
    assert res.stop_reason == "max_rounds" and res.rounds == 3
    assert len(llm.requests) == 3, "no request beyond the limit"
    assert res.messages[-1].role == "assistant" and not res.messages[-1].tool_calls
    assert_history_valid(res.messages)
    # The next turn works from the stored history.
    nxt = a.run(res.messages, "ok, just answer")
    assert nxt.stop_reason == "answer"
    assert llm.requests[-1].messages[-1].content == "ok, just answer"


def test_infra_failure_is_reported_not_raised():
    err = AllProvidersFailed(
        "all down", [Attempt(provider="google", model="m", outcome="retryable", status=503)]
    )
    res = agent(ScriptedLLM(err)).run([], "x")
    assert res.stop_reason == "llm_error" and res.infra_failure
    assert res.attempts[0].status == 503


def test_request_error_is_not_marked_as_infra():
    err = ProviderRequestError("bad", status=400, provider="google", attempts=[])
    res = agent(ScriptedLLM(err)).run([], "x")
    assert res.stop_reason == "llm_error" and not res.infra_failure
    assert res.error_type == "ProviderRequestError"


def test_long_history_is_trimmed_before_sending():
    old = []
    for i in range(50):
        old += [Message.user(f"question {i} " + "x" * 400), Message.assistant("answer " + "y" * 400)]
    llm = ScriptedLLM(say("fine"))
    agent(llm, history_token_budget=1000).run(old, "latest question")
    sent = llm.requests[0].messages
    assert sent[0].role == "system" and sent[1].role == "user"
    assert sent[-1].content == "latest question"
    assert len(sent) < 20
