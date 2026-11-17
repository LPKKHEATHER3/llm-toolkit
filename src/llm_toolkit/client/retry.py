"""Backoff policy and Retry-After parsing."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter: ``uniform(0, min(cap, base * 2**attempt))``.

    If the server sent ``Retry-After``, the wait is at least that long.
    """

    base_s: float = 0.5
    cap_s: float = 8.0
    rand: Callable[[], float] = field(default=random.random, compare=False, repr=False)

    def backoff(self, attempt: int, retry_after_s: float | None = None) -> float:
        ceiling = min(self.cap_s, self.base_s * (2**attempt))
        wait = ceiling * self.rand()
        if retry_after_s is not None:
            wait = max(wait, retry_after_s)
        return wait


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date). Returns seconds, or None."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds) if seconds == seconds else None  # reject NaN
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    return max(0.0, (when - now).total_seconds())
