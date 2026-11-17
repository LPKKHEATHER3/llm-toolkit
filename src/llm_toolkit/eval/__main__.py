"""Command line: ``python -m llm_toolkit.eval {run,report,verifier-bench,build-planted,check}``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ..client import LLMClient, UsageLog, default_providers
from ..redact import KEY_ENV_VARS
from .cases import load_cases
from .graders import check_case_graders
from .planted import build_planted_set, run_verifier_bench
from .report import write_report
from .runner import HARD_MAX_REQUESTS, RunSettings, run_cases

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results" / "eval"
PLANTED = ROOT / "data" / "eval" / "planted_citations.json"
PRICES = ROOT / "data" / "prices.json"


def cmd_check(_: argparse.Namespace) -> int:
    cases = load_cases()
    for c in cases:
        check_case_graders(c)
    print(f"ok: {len(cases)} cases, all grader types known")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    provider_keys = [k for k in KEY_ENV_VARS if k != "LLM_TOOLKIT_PROXY_TOKEN"]
    if not any(os.environ.get(k) for k in provider_keys):
        print(f"skipped: no provider key in the environment (set one of {', '.join(provider_keys)})")
        return 0
    max_requests = min(args.max_requests, HARD_MAX_REQUESTS)
    providers = [p for p in default_providers() if os.environ.get(p.key_env)]
    label = ">".join(f"{p.name}:{p.model}" for p in providers)
    client = LLMClient(
        providers,
        usage_log=UsageLog(ROOT / "logs" / "usage.jsonl"),
        max_requests=max_requests,
        min_interval_s=args.min_interval,
    )
    # Cheap single-request RAG cases first, so a small request budget still covers both suites.
    cases = sorted(load_cases(suite=args.suite), key=lambda c: c.suite != "rag")
    settings = RunSettings(model_label=label, temperature=args.temperature, repeats=args.repeats)
    summary = run_cases(cases, client, settings, RESULTS)
    print(
        f"run {summary.run_id}: {dict(summary.outcomes)}; HTTP requests sent: {client.request_count}/{max_requests}"
    )
    if summary.stopped_early:
        print("stopped early: request budget used up (remaining cases recorded as not_run)")
    write_report(RESULTS, PRICES)
    return 0


def cmd_report(_: argparse.Namespace) -> int:
    print(f"wrote {write_report(RESULTS, PRICES)}")
    return 0


def cmd_build_planted(_: argparse.Namespace) -> int:
    data = build_planted_set()
    PLANTED.parent.mkdir(parents=True, exist_ok=True)
    PLANTED.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n = sum(len(a["citations"]) for a in data["answers"])
    print(f"wrote {PLANTED} ({len(data['answers'])} answers, {n} citations)")
    return 0


def cmd_verifier_bench(args: argparse.Namespace) -> int:
    result = run_verifier_bench(PLANTED)
    print(
        f"planted defects dropped {result['defects_dropped']}/{result['defects']}, "
        f"genuine quotes kept {result['genuine_kept']}/{result['genuine']}, "
        f"expected drop reason {result['reason_matches']}/{result['defects']}"
    )
    if not args.dry_run:
        RESULTS.mkdir(parents=True, exist_ok=True)
        with (RESULTS / "verifier.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        write_report(RESULTS, PRICES)
    ok = result["defects_dropped"] == result["defects"] and result["genuine_kept"] == result["genuine"]
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="python -m llm_toolkit.eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run cases against live models (needs a key; max 30 requests)")
    run.add_argument("--suite", choices=["agent", "rag", "all"], default="all")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-requests", type=int, default=HARD_MAX_REQUESTS)
    run.add_argument("--min-interval", type=float, default=0.0, help="seconds between HTTP requests")
    run.set_defaults(fn=cmd_run)
    sub.add_parser("report", help="regenerate results/eval/REPORT.md").set_defaults(fn=cmd_report)
    vb = sub.add_parser("verifier-bench", help="offline benchmark of the citation verifier")
    vb.add_argument("--dry-run", action="store_true", help="print only, append nothing")
    vb.set_defaults(fn=cmd_verifier_bench)
    sub.add_parser("build-planted", help="regenerate the planted-defect dataset").set_defaults(
        fn=cmd_build_planted
    )
    sub.add_parser("check", help="validate case files").set_defaults(fn=cmd_check)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
