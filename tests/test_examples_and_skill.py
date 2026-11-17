"""The examples run offline, and the Skill only names tools that exist."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from llm_toolkit.redact import KEY_ENV_VARS
from llm_toolkit.tools import InventoryAPI, build_default_registry

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "inventory-stock" / "SKILL.md"


def run_example(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in KEY_ENV_VARS}
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_policy_example_offline_drops_invented_quote():
    out = run_example("examples/policy_agent.py", "--offline", "Can I return opened electronics?")
    assert "verified  [returns#" in out and "(quote_not_found)" in out


def test_stock_example_offline_surfaces_backend_refusal():
    out = run_example("examples/stock_agent.py", "--offline")
    assert "ERROR: only 3 in stock, cannot issue 10" in out
    assert "backend ELE-0002 qty now: 0" in out


def test_mcp_stdio_demo_runs_the_server_as_a_subprocess():
    out = run_example("examples/mcp_stdio_demo.py")
    assert "is_error=True: only 3 in stock, cannot issue 10" in out
    assert "'undone': '-2 ELE-0002'" in out


def test_every_test_named_in_the_docs_exists():
    """README and the design notes cite tests as proof; a renamed or deleted test must break this."""
    defined = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        defined.update(re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), flags=re.M))
    for doc in ("README.md", "docs/design.md"):
        cited = set(re.findall(r"`(test_[a-z0-9_]+)`", (ROOT / doc).read_text(encoding="utf-8")))
        assert cited, doc
        assert cited - defined == set(), f"{doc} cites tests that do not exist"


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "SKILL.md must start with YAML frontmatter"
    out = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def test_skill_frontmatter_and_tool_names():
    text = SKILL.read_text(encoding="utf-8")
    meta = _frontmatter(text)
    assert meta["name"] == SKILL.parent.name
    assert 0 < len(meta["description"]) <= 1024
    registry = build_default_registry(InventoryAPI("http://x.invalid"))
    table = text.split("## Tools", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"^\| `([a-z_]+)` \|", table, flags=re.M)
    assert listed == registry.names, "the Skill's tool table matches the registry exactly"
    for arg in ("expected_label", "undo_label", "qty_after", "chunk_id"):
        assert arg in text
