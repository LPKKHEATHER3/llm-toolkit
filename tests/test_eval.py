"""Eval harness: cases, graders, append-only results, outcome classes, budget and report."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from scripted_llm import ScriptedLLM, call, calls, say

from llm_toolkit.client import (
    AllProvidersFailed,
    Attempt,
    DeadlineExceeded,
    LLMClient,
    ProviderRequestError,
    RetryPolicy,
    google,
)
from llm_toolkit.eval import Case, build_report, load_cases, percentile, run_cases, run_verifier_bench
from llm_toolkit.eval.__main__ import PLANTED
from llm_toolkit.eval.__main__ import main as eval_main
from llm_toolkit.eval.planted import EXPECTED_REASON
from llm_toolkit.eval.runner import RunSettings
from llm_toolkit.rag import default_index

RAG_CASE = Case.model_validate(
    {
        "id": "rag-express",
        "suite": "rag",
        "critical": True,
        "input": "How much does express delivery cost?",
        "graders": [
            {"type": "answer_contains_any", "values": ["120"]},
            {"type": "cites_doc", "doc": "shipping"},
        ],
    }
)
AGENT_CASE = Case.model_validate(
    {
        "id": "issue-clips",
        "suite": "agent",
        "input": "Sold 2 binder clips OSU-0001",
        "fixture": "inventory_default",
        "graders": [
            {"type": "tool_called", "name": "issue_stock", "args": {"sku": "OSU-0001", "quantity": 2}},
            {"type": "stock_equals", "sku": "OSU-0001", "qty": 38},
        ],
    }
)
SETTINGS = RunSettings(model_label="fake:model", temperature=0.0)


def express_answer(req):
    excerpts = req.messages[-1].content
    hit = next(
        h for h in default_index().search("How much does express delivery cost?", 4) if "120" in h.chunk.text
    )
    assert hit.chunk.id in excerpts
    quote = next(s for s in hit.chunk.text.split(". ") if "120" in s)
    return say(json.dumps({"answer": "120 baht.", "citations": [{"chunk_id": hit.chunk.id, "quote": quote}]}))


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


# ---- cases --------------------------------------------------------------------------------------


def test_bundled_cases_load_and_use_known_graders():
    cases = load_cases()
    assert {c.suite for c in cases} == {"agent", "rag"}
    assert any(c.critical for c in cases)
    assert eval_main(["check"]) == 0


def test_case_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Case.model_validate({**RAG_CASE.model_dump(), "measures": "⭐ critical"})


def test_duplicate_case_ids_are_rejected(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps([RAG_CASE.model_dump(), RAG_CASE.model_dump()]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path)


# ---- runner -------------------------------------------------------------------------------------


def test_rag_case_passes_with_verified_citation(tmp_path):
    summary = run_cases([RAG_CASE], ScriptedLLM(express_answer), SETTINGS, tmp_path, sha="abc")
    (line,) = summary.lines
    assert line["outcome"] == "pass", line["notes"]
    assert line["citations_kept"] == 1 and line["citations_dropped"] == 0


def test_agent_case_runs_real_tools_against_fake_backend(tmp_path):
    llm = ScriptedLLM(
        calls(call("issue_stock", {"sku": "OSU-0001", "quantity": 2, "reason": "sale"})),
        say("Done, 38 left."),
    )
    (line,) = run_cases([AGENT_CASE], llm, SETTINGS, tmp_path, sha="abc").lines
    assert line["outcome"] == "pass", line["notes"]
    assert line["tool_calls"] == [{"name": "issue_stock", "ok": True, "error": None}]
    # The model was offered exactly the registry's tools.
    names = {t["function"]["name"] for t in llm.requests[0].tools}
    assert names == {"list_items", "get_item", "issue_stock", "undo_last_change", "search_policies"}


def test_results_are_appended_not_overwritten(tmp_path):
    run_cases([RAG_CASE], ScriptedLLM(express_answer), SETTINGS, tmp_path, run_id="first", sha="abc")
    before = (tmp_path / "rag.jsonl").read_text(encoding="utf-8")
    run_cases([RAG_CASE], ScriptedLLM(say("no idea")), SETTINGS, tmp_path, run_id="second", sha="abc")
    after = (tmp_path / "rag.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before)
    assert [x["run_id"] for x in read(tmp_path / "rag.jsonl")] == ["first", "second"]


def test_result_lines_record_run_settings(tmp_path):
    settings = RunSettings(model_label="fake:model", temperature=0.3, repeats=2)
    summary = run_cases(
        [RAG_CASE], ScriptedLLM(express_answer, express_answer), settings, tmp_path, sha="abc123"
    )
    assert [ln["repeat"] for ln in summary.lines] == [0, 1]
    line = summary.lines[0]
    for key in (
        "run_id",
        "git_sha",
        "model",
        "temperature",
        "prompt_hash",
        "latency_ms",
        "prompt_tokens",
        "case_id",
    ):
        assert key in line
    assert line["temperature"] == 0.3 and line["git_sha"] == "abc123"


def test_infra_errors_are_excluded_from_accuracy(tmp_path):
    down = AllProvidersFailed(
        "down", [Attempt(provider="google", model="m", outcome="retryable", status=429)]
    )
    slow = DeadlineExceeded("slow", [])
    cases = [RAG_CASE.model_copy(update={"id": f"c{i}"}) for i in range(3)]
    summary = run_cases(cases, ScriptedLLM(express_answer, down, slow), SETTINGS, tmp_path, sha="abc")
    assert [ln["outcome"] for ln in summary.lines] == ["pass", "infra_error", "infra_error"]
    report = build_report(tmp_path, tmp_path / "none.json")
    assert "| rag | `fake:model` | 1/1 (100%) | 1/1 | 2 | 0 | 0 |" in report


def test_request_errors_are_reported_as_errors_not_infra(tmp_path):
    bad = ProviderRequestError("400", status=400, provider="google", attempts=[])
    llm = ScriptedLLM(calls(call("get_item", {"sku": "OSU-0001"})), bad)
    (line,) = run_cases([AGENT_CASE], llm, SETTINGS, tmp_path, sha="abc").lines
    assert line["outcome"] == "error" and line["error_type"] == "ProviderRequestError"


def test_runner_uses_agent_max_rounds(tmp_path):
    forever = [calls(call("get_item", {"sku": "OSU-0001"}, f"c{i}")) for i in range(10)]
    llm = ScriptedLLM(*forever)
    settings = RunSettings(model_label="fake", max_rounds=3)
    (line,) = run_cases([AGENT_CASE], llm, settings, tmp_path, sha="abc").lines
    assert len(llm.requests) == 3 and line["stop_reason"] == "max_rounds" and line["max_rounds"] == 3


def test_run_stops_at_request_budget(tmp_path):
    body = {"choices": [{"message": {"role": "assistant", "content": '{"answer": "x", "citations": []}'}}]}
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=body)

    env = {"GEMINI_API_KEY": "AIzaFAKE000000000000000000000000"}
    client = LLMClient(
        [google(env)],
        transport=httpx.MockTransport(handler),
        env=env,
        max_requests=3,
        retry=RetryPolicy(rand=lambda: 0.0),
    )
    cases = [RAG_CASE.model_copy(update={"id": f"c{i}"}) for i in range(5)]
    summary = run_cases(cases, client, SETTINGS, tmp_path, sha="abc")
    assert len(sent) == 3 and client.request_count == 3
    assert [ln["outcome"] for ln in summary.lines].count("not_run") == 2
    assert summary.stopped_early


def test_cli_run_skips_without_keys(capsys, tmp_path, monkeypatch):
    import llm_toolkit.eval.__main__ as cli

    monkeypatch.setattr(cli, "RESULTS", tmp_path)
    assert eval_main(["run"]) == 0
    assert "skipped: no provider key" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_cli_caps_requests_at_30():
    from llm_toolkit.eval import HARD_MAX_REQUESTS

    assert HARD_MAX_REQUESTS == 30


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # Verbatim answer from live run 20261116T233749Z-e646b9 that the first pattern graded as a fail.
        ("I am sorry, but the provided excerpts do not state whether orders can be shipped to Japan.", True),
        ("The policy doesn't mention international shipping.", True),
        ("This is not covered by the documents.", True),
        ("Yes, we ship to Japan within 5 days.", False),
    ],
)
def test_abstains_grader(answer, expected):
    from llm_toolkit.eval.graders import GRADERS, RunOutput
    from llm_toolkit.rag.answer import CitedAnswer

    out = RunOutput(rag=CitedAnswer(question="q", answer=answer, citations=[], dropped=[], retrieved=[]))
    assert GRADERS["abstains"]({}, out).passed is expected


# ---- report -------------------------------------------------------------------------------------


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10.0
    assert percentile(values, 95) == 19.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 50) is None


def test_report_is_generated_from_results_file(tmp_path):
    lines = [
        {
            "run_id": "r1",
            "suite": "rag",
            "model": "m",
            "case_id": f"c{i}",
            "critical": i == 0,
            "outcome": "pass" if i < 3 else "fail",
            "latency_ms": float(100 * (i + 1)),
            "llm_calls": 1,
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "cost_usd": 0.001,
            "citations_kept": 1,
            "citations_dropped": i % 2,
        }
        for i in range(4)
    ]
    (tmp_path / "rag.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    report = build_report(tmp_path, tmp_path / "none.json")
    assert (
        "| rag | `m` | 3/4 (75%) | 1/1 | 0 | 0 | 0 | 200 | 400 | 1000 | 50 | $0.004000 (provider-reported) |"
        in report
    )
    assert "4 quotes verified, 2 dropped" in report


# ---- verifier benchmark -------------------------------------------------------------------------


def test_verifier_bench_detects_every_planted_defect_type():
    result = run_verifier_bench(PLANTED)
    assert set(result["by_label"]) == {"genuine", *EXPECTED_REASON}, "every defect type is planted"
    assert result["defects_dropped"] == result["defects"]
    assert result["genuine_kept"] == result["genuine"]
    assert result["reason_matches"] == result["defects"]


def test_planted_dataset_is_reproducible():
    from llm_toolkit.eval import build_planted_set

    assert build_planted_set() == json.loads(PLANTED.read_text(encoding="utf-8"))
