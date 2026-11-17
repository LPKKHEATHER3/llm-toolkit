"""Thread-safe JSONL usage log and the prompt-token alarm."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..redact import get_logger, redact_obj

log = get_logger(__name__)


class UsageLog:
    """Appends one JSON object per line. Each line is written with a single ``write`` call under a
    lock, so concurrent threads never interleave partial lines."""

    def __init__(self, path: str | os.PathLike[str] | None, *, alarm_prompt_tokens: int = 8000):
        self.path = Path(path) if path else None
        self.alarm_prompt_tokens = alarm_prompt_tokens
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None,
        latency_ms: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "latency_ms": round(latency_ms, 1),
        }
        if extra:
            entry.update(extra)
        entry = redact_obj(entry)
        if prompt_tokens > self.alarm_prompt_tokens:
            log.warning(
                "prompt_tokens=%d from %s/%s exceeds the alarm threshold %d; "
                "something may be injecting context",
                prompt_tokens,
                provider,
                model,
                self.alarm_prompt_tokens,
            )
        if self.path:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line)
        return entry
