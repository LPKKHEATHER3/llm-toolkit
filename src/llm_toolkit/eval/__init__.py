"""Evaluation harness: JSON cases, code graders, append-only results, generated report."""

from .cases import Case, GraderSpec, load_cases
from .graders import GRADERS, GradeResult, RunOutput, grade
from .planted import build_planted_set, run_verifier_bench
from .report import build_report, percentile, write_report
from .runner import HARD_MAX_REQUESTS, RunSettings, RunSummary, run_cases

__all__ = [
    "GRADERS",
    "HARD_MAX_REQUESTS",
    "Case",
    "GradeResult",
    "GraderSpec",
    "RunOutput",
    "RunSettings",
    "RunSummary",
    "build_planted_set",
    "build_report",
    "grade",
    "load_cases",
    "percentile",
    "run_cases",
    "run_verifier_bench",
    "write_report",
]
