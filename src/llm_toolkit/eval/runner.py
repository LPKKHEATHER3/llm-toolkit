"""Run eval cases against a model and append one JSON line per (case, repeat).

* Results are **appended** (file mode ``"a"``), never overwritten.
* Each line records the outcome class: ``pass`` / ``fail`` (model behaviour), ``infra_error``
  (429, 5xx, timeouts, deadline, every provider down: not the model's fault), ``error`` (our request
  was rejected, e.g. HTTP 400) or ``not_run`` (the request budget ran out first).
* The agent suite uses the real :class:`~llm_toolkit.agent.Agent` and the tools from
  :func:`~llm_toolkit.tools.build_default_registry` against a fresh fake inventory per case.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent.loop import Agent, ChatClient
from ..client.types import AllProvidersFailed, DeadlineExceeded, LLMError, RequestBudgetExhausted
from ..rag import BM25Index, by_id, load_corpus
from ..rag.answer import SYSTEM_PROMPT as RAG_PROMPT
from ..rag.answer import answer_with_citations
from ..tools import STOCK_AGENT_PROMPT, InventoryAPI, build_default_registry
from ..tools.fake_inventory import DEFAULT_ITEMS, FakeInventoryBackend
from .cases import Case
from .graders import RunOutput, grade

HARD_MAX_REQUESTS = 30
FIXTURES = {"inventory_default": DEFAULT_ITEMS}


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def git_sha(cwd: Path | None = None) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=cwd, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not sha:
        return "uncommitted"
    return f"{sha}-dirty" if dirty else sha


@dataclass
class RunSettings:
    model_label: str
    temperature: float | None = 0.0
    repeats: int = 1
    max_rounds: int = 5
    deadline_s: float = 60.0


@dataclass
class RunSummary:
    run_id: str
    lines: list[dict[str, Any]] = field(default_factory=list)
    stopped_early: bool = False

    @property
    def outcomes(self) -> Counter[str]:
        return Counter(line["outcome"] for line in self.lines)


def _classify_exception(e: LLMError) -> str:
    if isinstance(e, RequestBudgetExhausted):
        return "not_run"
    if isinstance(e, AllProvidersFailed | DeadlineExceeded):
        return "infra_error"
    return "error"


def _agent_case(
    case: Case, client: ChatClient, settings: RunSettings, index: BM25Index
) -> tuple[RunOutput, dict]:
    items = FIXTURES[case.fixture or "inventory_default"]
    backend = FakeInventoryBackend(items)
    api = InventoryAPI("http://inventory.test", transport=backend.transport)
    registry = build_default_registry(api, index)
    agent = Agent(
        client,
        registry,
        system_prompt=STOCK_AGENT_PROMPT,
        max_rounds=settings.max_rounds,
        temperature=settings.temperature,
        deadline_s=settings.deadline_s,
    )
    res = agent.run([], case.input)
    meta = {
        "prompt_hash": short_hash(STOCK_AGENT_PROMPT),
        "tool_schema_hash": registry.schema_hash(),
        "llm_calls": len(res.usage),
        "latency_ms": round(sum(res.latency_ms), 1),
        "prompt_tokens": sum(u.prompt_tokens for u in res.usage),
        "completion_tokens": sum(u.completion_tokens for u in res.usage),
        "cost_usd": _sum_cost([u.cost_usd for u in res.usage]),
        "providers": res.providers,
        "stop_reason": res.stop_reason,
        "tool_calls": [{"name": c.name, "ok": c.ok, "error": c.error} for c in res.tool_calls],
    }
    if res.stop_reason == "llm_error":
        meta["error"] = res.error
        meta["error_type"] = res.error_type
        meta["attempts"] = [a.model_dump() for a in res.attempts]
    return RunOutput(agent=res, backend=backend), meta


def _rag_case(
    case: Case, client: ChatClient, settings: RunSettings, index: BM25Index, corpus: dict
) -> tuple[RunOutput, dict]:
    ans = answer_with_citations(client, index, case.input, corpus=corpus, deadline_s=settings.deadline_s)
    meta = {
        "prompt_hash": short_hash(RAG_PROMPT),
        "tool_schema_hash": None,
        "llm_calls": 1,
        "latency_ms": round(ans.latency_ms, 1),
        "prompt_tokens": ans.usage.prompt_tokens,
        "completion_tokens": ans.usage.completion_tokens,
        "cost_usd": ans.usage.cost_usd,
        "providers": [ans.provider],
        "citations_kept": len(ans.citations),
        "citations_dropped": ans.dropped_count,
        "drop_reasons": dict(Counter(d.reason for d in ans.dropped)),
        "parse_error": ans.parse_error,
        "retrieved": ans.retrieved,
    }
    return RunOutput(rag=ans), meta


def _sum_cost(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(sum(known), 8) if known and len(known) == len(values) else None


def run_cases(
    cases: list[Case],
    client: ChatClient,
    settings: RunSettings,
    results_dir: str | Path,
    *,
    run_id: str | None = None,
    sha: str | None = None,
) -> RunSummary:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = RunSummary(run_id=run_id or f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}")
    sha = sha or git_sha()
    corpus_chunks = load_corpus()
    index = BM25Index.build(corpus_chunks)
    corpus = by_id(corpus_chunks)

    for case in cases:
        for repeat in range(settings.repeats):
            base = {
                "run_id": summary.run_id,
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_sha": sha,
                "suite": case.suite,
                "case_id": case.id,
                "critical": case.critical,
                "model": settings.model_label,
                "repeat": repeat,
                "temperature": settings.temperature,
                "max_rounds": settings.max_rounds if case.suite == "agent" else None,
            }
            if summary.stopped_early:
                line = {**base, "outcome": "not_run", "notes": ["request budget used up before this case"]}
            else:
                line = _run_one(case, client, settings, index, corpus, base)
                if line["outcome"] == "not_run":
                    summary.stopped_early = True
            summary.lines.append(line)
            with (results_dir / f"{case.suite}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return summary


def _run_one(
    case: Case, client: ChatClient, settings: RunSettings, index: BM25Index, corpus: dict, base: dict
) -> dict:
    try:
        if case.suite == "agent":
            out, meta = _agent_case(case, client, settings, index)
        else:
            out, meta = _rag_case(case, client, settings, index, corpus)
    except LLMError as e:
        return {
            **base,
            "outcome": _classify_exception(e),
            "error": str(e)[:300],
            "error_type": type(e).__name__,
            "attempts": [a.model_dump() for a in e.attempts],
            "notes": [],
        }

    if meta.get("stop_reason") == "llm_error":
        kind = meta.get("error_type")
        if kind == "RequestBudgetExhausted":
            outcome = "not_run"
        elif out.agent is not None and out.agent.infra_failure:
            outcome = "infra_error"
        else:
            outcome = "error"
        return {**base, **meta, "outcome": outcome, "notes": []}

    graded = grade(case, out)
    passed = all(r.passed for _, r in graded)
    notes = [f"{'PASS' if r.passed else 'FAIL'} {spec.type}: {r.note}" for spec, r in graded]
    return {**base, **meta, "outcome": "pass" if passed else "fail", "notes": notes, "answer": out.text[:500]}
