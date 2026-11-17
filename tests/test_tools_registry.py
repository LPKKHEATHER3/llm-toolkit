"""Argument validation in the shared tool registry."""

from __future__ import annotations

import json

import pytest
from pydantic import Field

from llm_toolkit.tools import Tool, ToolArgs, ToolError, ToolRegistry, provided


class IssueArgs(ToolArgs):
    sku: str = Field(pattern=r"^[A-Z0-9]{1,8}-\d{1,8}$")
    quantity: int = Field(gt=0)
    reason: str = Field(min_length=1, max_length=120)
    note: str | None = None


seen: list[dict] = []


def issue(args: IssueArgs) -> dict:
    seen.append(provided(args))
    return {"issued": args.quantity}


@pytest.fixture
def registry() -> ToolRegistry:
    seen.clear()
    return ToolRegistry([Tool("issue_stock", "Issue stock.", IssueArgs, issue)])


GOOD = {"sku": "ELE-0001", "quantity": 2, "reason": "sale"}


def test_valid_call_returns_data(registry):
    out = registry.call("issue_stock", json.dumps(GOOD))
    assert out.ok and out.data == {"issued": 2}


def test_malformed_json_returns_decoder_message(registry):
    out = registry.call("issue_stock", '{"sku": "ELE-0001", "quantity": 2')
    assert not out.ok
    assert "not valid JSON" in out.error and "position" in out.error
    assert seen == [], "handler never runs on malformed input"


def test_non_object_arguments_are_rejected(registry):
    assert "must be a JSON object" in registry.call("issue_stock", "[1, 2]").error


def test_empty_arguments_are_not_treated_as_empty_object(registry):
    out = registry.call("issue_stock", "")
    assert not out.ok and "not valid JSON" in out.error


def test_strict_types_do_not_coerce_strings(registry):
    out = registry.call("issue_stock", json.dumps({**GOOD, "quantity": "2"}))
    assert not out.ok and "quantity" in out.error


def test_missing_required_field_is_not_defaulted(registry):
    out = registry.call("issue_stock", json.dumps({"sku": "ELE-0001", "reason": "sale"}))
    assert not out.ok and "quantity: field required" in out.error
    assert seen == []


def test_optional_field_not_sent_is_not_passed_to_handler(registry):
    registry.call("issue_stock", json.dumps(GOOD))
    assert seen == [GOOD], "note was not sent, so the handler must not see it"


def test_unknown_argument_is_rejected(registry):
    out = registry.call("issue_stock", json.dumps({**GOOD, "units": 2}))
    assert not out.ok and "units: unknown field" in out.error and "allowed: sku, quantity" in out.error


def test_schema_mismatch_lists_field_errors(registry):
    out = registry.call("issue_stock", json.dumps({"sku": "bad sku", "quantity": -1, "reason": ""}))
    assert not out.ok
    for field in ("sku:", "quantity:", "reason:"):
        assert field in out.error


def test_unknown_tool_lists_available_tools(registry):
    out = registry.call("delete_everything", "{}")
    assert not out.ok and "issue_stock" in out.error


def test_tool_exception_is_returned_as_error():
    def boom(_args: IssueArgs) -> None:
        raise RuntimeError("backend exploded")

    reg = ToolRegistry([Tool("issue_stock", "d", IssueArgs, boom)])
    out = reg.call("issue_stock", json.dumps(GOOD))
    assert not out.ok and out.error == "RuntimeError: backend exploded"


def test_tool_error_message_is_passed_through():
    def refuse(_args: IssueArgs) -> None:
        raise ToolError("only 1 in stock, cannot issue 2")

    out = ToolRegistry([Tool("issue_stock", "d", IssueArgs, refuse)]).call("issue_stock", json.dumps(GOOD))
    assert out.error == "only 1 in stock, cannot issue 2"


def test_keyboard_interrupt_is_not_swallowed():
    def interrupt(_args: IssueArgs) -> None:
        raise KeyboardInterrupt

    reg = ToolRegistry([Tool("issue_stock", "d", IssueArgs, interrupt)])
    with pytest.raises(KeyboardInterrupt):
        reg.call("issue_stock", json.dumps(GOOD))


def test_optional_fields_must_default_to_none():
    class Bad(ToolArgs):
        limit: int = 10

    with pytest.raises(ValueError, match="must default to None"):
        Tool("bad", "d", Bad, lambda a: a)


def test_schemas_are_clean_for_every_format(registry):
    params = registry.get("issue_stock").parameters()
    assert params["required"] == ["sku", "quantity", "reason"]
    assert params["additionalProperties"] is False
    assert params["properties"]["note"] == {"type": "string"}, "optional field shown without null/default"
    assert "title" not in json.dumps(params)
    assert registry.schemas("anthropic")[0]["input_schema"] == params
    assert len(registry.schema_hash()) == 16
