"""Eval cases are data (JSON), checked by code graders."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GraderSpec(BaseModel):
    """``type`` selects a grader; the other keys are that grader's parameters."""

    model_config = ConfigDict(extra="allow")

    type: str

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    suite: Literal["agent", "rag"]
    critical: bool = False
    input: str = Field(min_length=1)
    fixture: str | None = None
    graders: list[GraderSpec] = Field(min_length=1)


def default_cases_dir() -> Path:
    env = os.environ.get("LLM_TOOLKIT_CASES_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "data" / "eval" / "cases"


def load_cases(path: str | os.PathLike[str] | None = None, suite: str | None = None) -> list[Case]:
    """Load one JSON file or every ``*.json`` in a directory. Ids must be unique."""
    root = Path(path) if path else default_cases_dir()
    files = sorted(root.glob("*.json")) if root.is_dir() else [root]
    cases: list[Case] = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{f}: expected a JSON list of cases")
        cases.extend(Case.model_validate(c) for c in data)
    ids = [c.id for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
    return [c for c in cases if suite in (None, "all", c.suite)]
