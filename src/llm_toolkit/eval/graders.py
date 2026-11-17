"""Code graders. Each takes the case's grader parameters and the run output and returns
``(passed, note)``. Structural checks (which tool, which arguments, what the backend state is) are
preferred over matching the model's wording."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agent.loop import AgentResult
from ..rag.answer import CitedAnswer
from ..tools.fake_inventory import FakeInventoryBackend
from .cases import Case, GraderSpec


@dataclass
class RunOutput:
    agent: AgentResult | None = None
    rag: CitedAnswer | None = None
    backend: FakeInventoryBackend | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.agent is not None:
            return self.agent.final_text
        if self.rag is not None:
            return self.rag.answer
        return ""


@dataclass
class GradeResult:
    passed: bool
    note: str


Grader = Callable[[dict[str, Any], RunOutput], GradeResult]
GRADERS: dict[str, Grader] = {}


def grader(name: str) -> Callable[[Grader], Grader]:
    def deco(fn: Grader) -> Grader:
        GRADERS[name] = fn
        return fn

    return deco


def _args(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


# ---- agent graders ------------------------------------------------------------------------------


@grader("tool_called")
def tool_called(p: dict[str, Any], out: RunOutput) -> GradeResult:
    """A successful call to ``name`` (or any of ``names``) whose arguments include ``args``."""
    names = p.get("names") or [p["name"]]
    want = p.get("args", {})
    calls = [c for c in (out.agent.tool_calls if out.agent else []) if c.name in names]
    for c in calls:
        got = _args(c.arguments)
        if all(got.get(k) == v for k, v in want.items()) and (c.ok or p.get("allow_error", False)):
            return GradeResult(True, f"{c.name} called with {c.arguments}")
    seen = [f"{c.name}({c.arguments}){'' if c.ok else ' -> error'}" for c in calls]
    return GradeResult(False, f"no matching {'/'.join(names)} call with {want}; saw {seen or 'none'}")


@grader("tool_not_called")
def tool_not_called(p: dict[str, Any], out: RunOutput) -> GradeResult:
    hits = [c for c in (out.agent.tool_calls if out.agent else []) if c.name == p["name"] and c.ok]
    return GradeResult(not hits, f"{p['name']} succeeded {len(hits)} time(s)")


@grader("stock_equals")
def stock_equals(p: dict[str, Any], out: RunOutput) -> GradeResult:
    if out.backend is None:
        return GradeResult(False, "no backend state")
    qty = out.backend.items[p["sku"]]["qty"]
    return GradeResult(qty == p["qty"], f"{p['sku']} qty is {qty}, expected {p['qty']}")


@grader("no_movements")
def no_movements(p: dict[str, Any], out: RunOutput) -> GradeResult:
    n = len(out.backend.movements) if out.backend else 0
    return GradeResult(n == 0, f"{n} stock movement(s) recorded")


# ---- text graders (both suites) --------------------------------------------------------------------


@grader("answer_contains_any")
def answer_contains_any(p: dict[str, Any], out: RunOutput) -> GradeResult:
    text = _norm(out.text)
    found = [v for v in p["values"] if _norm(v) in text]
    return GradeResult(bool(found), f"found {found}" if found else f"none of {p['values']} in answer")


@grader("answer_mentions_all")
def answer_mentions_all(p: dict[str, Any], out: RunOutput) -> GradeResult:
    """``groups`` is a list of alternatives lists; every group needs one hit."""
    text = _norm(out.text)
    missing = [g for g in p["groups"] if not any(_norm(v) in text for v in g)]
    return GradeResult(not missing, f"missing {missing}" if missing else "all mentioned")


@grader("answer_matches")
def answer_matches(p: dict[str, Any], out: RunOutput) -> GradeResult:
    ok = re.search(p["pattern"], out.text, flags=re.IGNORECASE) is not None
    return GradeResult(ok, f"pattern {p['pattern']!r} {'found' if ok else 'not found'}")


# ---- RAG graders ----------------------------------------------------------------------------------


@grader("grounded")
def grounded(p: dict[str, Any], out: RunOutput) -> GradeResult:
    n = len(out.rag.citations) if out.rag else 0
    return GradeResult(n >= p.get("min", 1), f"{n} verified citation(s)")


@grader("cites_doc")
def cites_doc(p: dict[str, Any], out: RunOutput) -> GradeResult:
    docs = sorted({c.chunk_id.split("#")[0] for c in (out.rag.citations if out.rag else [])})
    return GradeResult(p["doc"] in docs, f"verified citations from {docs or 'no documents'}")


@grader("abstains")
def abstains(p: dict[str, Any], out: RunOutput) -> GradeResult:
    """For questions the corpus does not answer: the answer says the information is not there.

    Wording-based, so it is kept broad. A live run showed the first, narrower pattern missing
    "the provided excerpts do not state whether ..." (a correct abstention graded as a fail)."""
    verbs = r"(?:say|state|mention|cover|specify|include|address|contain|indicate|provide|answer)"
    pattern = p.get(
        "pattern",
        rf"\b(not (?:mentioned|covered|specified|stated|available|included|found|addressed)|"
        rf"no (?:information|mention|details?)|(?:do|does|did)(?: not|n't) {verbs}|cannot find|"
        rf"can't find|unable to (?:find|answer|determine))\b",
    )
    said = re.search(pattern, out.text, flags=re.IGNORECASE) is not None
    return GradeResult(said, "abstained" if said else "answered without saying the policy is silent")


def grade(case: Case, out: RunOutput) -> list[tuple[GraderSpec, GradeResult]]:
    results = []
    for spec in case.graders:
        fn = GRADERS.get(spec.type)
        if fn is None:
            raise KeyError(f"case {case.id}: unknown grader type {spec.type!r}")
        try:
            results.append((spec, fn(spec.params, out)))
        except KeyError as e:
            results.append((spec, GradeResult(False, f"grader parameter missing: {e}")))
    return results


def check_case_graders(case: Case) -> None:
    for spec in case.graders:
        if spec.type not in GRADERS:
            raise KeyError(f"case {case.id}: unknown grader type {spec.type!r}")
