"""MCP server over the shared registry, tested in-process with the official client and a mocked
inventory backend (no network)."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import anyio
import httpx
import pytest
from mcp import Client
from scripted_llm import ScriptedLLM, call, calls, say

from llm_toolkit.agent import Agent
from llm_toolkit.eval import run_cases
from llm_toolkit.eval.cases import Case
from llm_toolkit.eval.runner import RunSettings
from llm_toolkit.mcp_server import build_server
from llm_toolkit.rag import default_index
from llm_toolkit.tools import STOCK_AGENT_PROMPT, InventoryAPI, build_default_registry
from llm_toolkit.tools.fake_inventory import FakeInventoryBackend

URL = "http://inventory.test"


def setup(backend: FakeInventoryBackend | None = None):
    backend = backend or FakeInventoryBackend()
    registry = build_default_registry(InventoryAPI(URL, transport=backend.transport), default_index())
    return backend, registry, build_server(registry)


def mcp_call(server, name: str, arguments: dict | None):
    async def go():
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return anyio.run(go)


def text(result) -> str:
    return "".join(c.text for c in result.content)


# ---- listing ------------------------------------------------------------------------------------


def test_mcp_lists_registry_tools():
    _, registry, server = setup()

    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    listed = anyio.run(go)
    assert [t.name for t in listed.tools] == registry.names
    for t in listed.tools:
        assert t.input_schema == registry.get(t.name).mcp_input_schema()


def test_agent_eval_and_mcp_share_one_tool_definition(tmp_path):
    """The prototype's bench kept its own copy of the tools and drifted (same input, different result). Here the
    agent, the eval runner and the MCP server all get the same schemas from one factory, and no
    other module defines a tool."""
    _, registry, server = setup()

    async def go():
        async with Client(server) as client:
            return await client.list_tools()

    mcp_schemas = {t.name: t.input_schema for t in anyio.run(go).tools}
    llm = ScriptedLLM(say("ok"))
    Agent(llm, registry, system_prompt=STOCK_AGENT_PROMPT).run([], "hi")
    agent_schemas = {t["function"]["name"]: t["function"]["parameters"] for t in llm.requests[0].tools}
    case = Case.model_validate(
        {
            "id": "x",
            "suite": "agent",
            "input": "hi",
            "graders": [{"type": "answer_contains_any", "values": ["ok"]}],
        }
    )
    (line,) = run_cases(
        [case], ScriptedLLM(say("ok")), RunSettings(model_label="fake"), tmp_path, sha="abc"
    ).lines
    assert mcp_schemas == agent_schemas
    assert line["tool_schema_hash"] == registry.schema_hash()

    src = Path(__file__).resolve().parents[1] / "src" / "llm_toolkit"
    definers = sorted(
        p.relative_to(src).as_posix()
        for p in src.rglob("*.py")
        if re.search(r'Tool\(\s*"(issue_stock|search_policies)"', p.read_text(encoding="utf-8"))
    )
    assert definers == ["tools/inventory.py", "tools/policies.py"]


# ---- backend rejections reach the model -----------------------------------------------------------


def test_insufficient_stock_error_reaches_client():
    backend, _, server = setup()
    result = mcp_call(server, "issue_stock", {"sku": "ELE-0002", "quantity": 10, "reason": "school order"})
    assert result.is_error is True
    assert text(result) == "only 3 in stock, cannot issue 10"
    assert backend.items["ELE-0002"]["qty"] == 3 and backend.movements == []


def test_insufficient_stock_error_reaches_the_agents_model():
    _, registry, _ = setup()
    llm = ScriptedLLM(
        calls(call("issue_stock", {"sku": "ELE-0002", "quantity": 10, "reason": "sale"})), say("Only 3.")
    )
    Agent(llm, registry, system_prompt=STOCK_AGENT_PROMPT).run([], "issue 10 ELE-0002")
    tool_msg = llm.requests[1].messages[-1]
    assert tool_msg.role == "tool"
    assert json.loads(tool_msg.content) == {"ok": False, "error": "only 3 in stock, cannot issue 10"}


def test_successful_issue_returns_structured_result():
    backend, _, server = setup()
    result = mcp_call(server, "issue_stock", {"sku": "OSU-0001", "quantity": 2, "reason": "sale"})
    assert not result.is_error
    assert result.structured_content == {
        "sku": "OSU-0001",
        "issued": 2,
        "qty_after": 38,
        "movement_id": 1,
        "undo_label": "-2 OSU-0001",
    }
    assert backend.items["OSU-0001"]["qty"] == 38


def test_unknown_sku_and_validation_errors_are_surfaced():
    _, _, server = setup()
    missing = mcp_call(server, "get_item", {"sku": "ELE-0099"})
    assert missing.is_error and text(missing) == "ELE-0099 not found"
    wrong_type = mcp_call(server, "issue_stock", {"sku": "ELE-0001", "quantity": "2", "reason": "sale"})
    assert wrong_type.is_error and "quantity" in text(wrong_type)
    missing_field = mcp_call(server, "issue_stock", {"sku": "ELE-0001", "reason": "sale"})
    assert "quantity: field required" in text(missing_field)
    bad_sku = mcp_call(server, "get_item", {"sku": "../../etc"})
    assert bad_sku.is_error and "sku" in text(bad_sku)


def test_backend_validation_details_are_passed_on():
    def reject(request: httpx.Request):
        if request.method == "POST":
            return httpx.Response(
                400, json={"error": "validation failed", "details": {"reason": "is required"}}
            )
        return None

    _, _, server = setup(FakeInventoryBackend(fail=reject))
    result = mcp_call(server, "issue_stock", {"sku": "ELE-0001", "quantity": 1, "reason": "x"})
    assert text(result) == "invalid request: validation failed: reason: is required"


# ---- fault tolerance --------------------------------------------------------------------------------


def test_issue_stock_is_not_retried_on_timeout():
    def timeout_on_post(request: httpx.Request):
        return httpx.ReadTimeout("slow") if request.method == "POST" else None

    backend = FakeInventoryBackend(fail=timeout_on_post)
    _, _, server = setup(backend)
    result = mcp_call(server, "issue_stock", {"sku": "ELE-0001", "quantity": 1, "reason": "sale"})
    assert result.is_error and "outcome unknown" in text(result)
    assert sum(1 for r in backend.requests if r.method == "POST") == 1


def test_reads_are_retried_once_on_5xx():
    seen = {"n": 0}

    def flaky(request: httpx.Request):
        seen["n"] += 1
        return httpx.Response(503, json={"error": "busy"}) if seen["n"] == 1 else None

    _, _, server = setup(FakeInventoryBackend(fail=flaky))
    result = mcp_call(server, "get_item", {"sku": "ELE-0001"})
    assert not result.is_error and result.structured_content["qty"] == 12


def test_backend_down_is_reported_as_unavailable():
    _, _, server = setup(FakeInventoryBackend(fail=lambda r: httpx.ConnectError("refused")))
    result = mcp_call(server, "list_items", {})
    assert result.is_error and "inventory service unavailable" in text(result)


def test_undo_refuses_unexpected_label():
    backend, _, server = setup()
    issued = mcp_call(server, "issue_stock", {"sku": "ELE-0001", "quantity": 5, "reason": "sale"})
    label = issued.structured_content["undo_label"]
    backend.push_foreign_change("STA-0001", -1)  # someone else changed stock afterwards
    refused = mcp_call(server, "undo_last_change", {"expected_label": label})
    assert refused.is_error and text(refused).startswith("refused:") and "-1 STA-0001" in text(refused)
    undo_req = backend.requests[-1]
    assert undo_req.url.path == "/api/undo" and json.loads(undo_req.content) == {"expected": label}
    assert not any(r.url.path == "/api/stats" for r in backend.requests), "no separate check-then-act read"
    assert backend.items["STA-0001"]["qty"] == 79 and backend.items["ELE-0001"]["qty"] == 7

    ok = mcp_call(server, "undo_last_change", {"expected_label": "-1 STA-0001"})
    assert not ok.is_error and ok.structured_content["undone"] == "-1 STA-0001"
    assert backend.items["STA-0001"]["qty"] == 80


def test_search_policies_tool_returns_quotable_excerpts():
    _, _, server = setup()
    result = mcp_call(server, "search_policies", {"query": "opened electronics return window", "k": 2})
    data = result.structured_content
    assert len(data["results"]) == 2 and data["results"][0]["chunk_id"].startswith("returns#")


def test_list_items_filters_only_what_was_sent():
    backend, _, server = setup()
    mcp_call(server, "list_items", {"low_stock_only": True})
    assert dict(backend.requests[-1].url.params) == {"low": "1"}
    mcp_call(server, "list_items", {})
    assert dict(backend.requests[-1].url.params) == {}


# ---- optional: against a real inventory-manager checkout ---------------------------------------------


@pytest.mark.skipif(not os.environ.get("INVENTORY_MANAGER_PATH"), reason="set INVENTORY_MANAGER_PATH to run")
def test_tools_against_real_inventory_manager():
    sys.path.insert(0, os.environ["INVENTORY_MANAGER_PATH"])
    from inventory.server import serve_in_thread
    from inventory.store import Store

    store = Store()
    item = store.create(
        {"name": "Test Cable", "category": "Electronics", "price": 10, "qty": 3, "reorder_level": 1}
    )
    srv = serve_in_thread(store)
    try:
        api = InventoryAPI(f"http://127.0.0.1:{srv.server_port}")
        registry = build_default_registry(api, default_index())
        refused = registry.call("issue_stock", {"sku": item["sku"], "quantity": 5, "reason": "sale"})
        assert not refused.ok and refused.error == "only 3 in stock, cannot issue 5"
        done = registry.call("issue_stock", {"sku": item["sku"], "quantity": 2, "reason": "sale"})
        assert done.ok and done.data["qty_after"] == 1
        assert registry.call("undo_last_change", {"expected_label": "-9 XXX-0001"}).ok is False
        undone = registry.call("undo_last_change", {"expected_label": done.data["undo_label"]})
        assert undone.ok and store.get(item["sku"])["qty"] == 3
        assert registry.call("get_item", {"sku": "ELE-9999"}).error == "ELE-9999 not found"
    finally:
        srv.shutdown()
        srv.server_close()
