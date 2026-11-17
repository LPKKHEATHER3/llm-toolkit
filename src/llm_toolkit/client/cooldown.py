"""Provider cooldowns after a quota error.

In-process checks use ``time.monotonic`` (immune to wall-clock jumps). The optional JSON file
stores wall-clock expiry so a cooldown survives a process restart.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path


class CooldownStore:
    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ):
        self._path = Path(path) if path else None
        self._monotonic = monotonic
        self._wall = wall
        self._lock = threading.Lock()
        self._until: dict[str, float] = {}  # provider -> monotonic expiry
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        assert self._path is not None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # a corrupt cooldown file must never stop the client
        now_wall, now_mono = self._wall(), self._monotonic()
        for name, wall_expiry in data.items() if isinstance(data, dict) else ():
            if isinstance(wall_expiry, int | float) and wall_expiry > now_wall:
                self._until[name] = now_mono + (wall_expiry - now_wall)

    def _save(self) -> None:
        if not self._path:
            return
        now_wall, now_mono = self._wall(), self._monotonic()
        data = {k: now_wall + (v - now_mono) for k, v in self._until.items() if v > now_mono}
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self._path)

    def set(self, provider: str, seconds: float) -> None:
        with self._lock:
            self._until[provider] = self._monotonic() + max(0.0, seconds)
            self._save()

    def remaining(self, provider: str) -> float:
        with self._lock:
            left = self._until.get(provider, 0.0) - self._monotonic()
            return left if left > 0 else 0.0

    def active(self, provider: str) -> bool:
        return self.remaining(provider) > 0
