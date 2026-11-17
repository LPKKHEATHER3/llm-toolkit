"""Secret redaction for every log line and usage record.

Anything that may end up in a log goes through :func:`redact` first. It removes:

* the current values of the known API-key environment variables,
* ``Bearer <token>`` and ``x-api-key: <token>`` style header values,
* strings shaped like provider keys (``sk-...``, ``sk-ant-...``, ``sk-or-...``, ``AIza...``),
* JSON Web Tokens (``eyJ....eyJ....<signature>``).
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "[REDACTED]"

KEY_ENV_VARS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_TOOLKIT_PROXY_TOKEN",
)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Header-style values: "Authorization: Bearer x", "x-api-key: x", "api_key=x", '"x-goog-api-key": "x"'
    (
        re.compile(
            r"(?i)((?:authorization|proxy-authorization|x-api-key|x-goog-api-key|api[_-]?key)"
            r"[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?[^\s\"',;}]+"
        ),
        r"\1" + REDACTED,
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer " + REDACTED),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*"), REDACTED),
    (re.compile(r"\bsk-(?:ant-|or-|proj-)?[A-Za-z0-9_-]{8,}"), REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"), REDACTED),
)


def _secret_values(env: Mapping[str, str] | None, extra: Iterable[str]) -> list[str]:
    env = os.environ if env is None else env
    values = [env.get(name, "") for name in KEY_ENV_VARS]
    values.extend(extra)
    # Very short values would redact ordinary words; real keys are long.
    return sorted({v for v in values if v and len(v) >= 8}, key=len, reverse=True)


def redact(text: str, *, env: Mapping[str, str] | None = None, extra_secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with every secret replaced by ``[REDACTED]``."""
    if not text:
        return text
    for value in _secret_values(env, extra_secrets):
        text = text.replace(value, REDACTED)
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact_obj(obj: Any, **kwargs: Any) -> Any:
    """Recursively redact strings inside dicts/lists (used for JSONL records)."""
    if isinstance(obj, str):
        return redact(obj, **kwargs)
    if isinstance(obj, Mapping):
        return {k: redact_obj(v, **kwargs) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [redact_obj(v, **kwargs) for v in obj]
    return obj


class RedactingFilter(logging.Filter):
    """Rewrites the fully formatted message so no handler ever sees a secret."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed %-args: still redact what we have
            message = str(record.msg)
        record.msg = redact(message)
        record.args = None
        if record.exc_info and record.exc_info[1] is not None:
            # Format the traceback now, redacted, and drop the live exception object.
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True


_FILTER = RedactingFilter()


def get_logger(name: str) -> logging.Logger:
    """Logger with the redacting filter attached.

    Logger-level filters are not inherited by child loggers, so every module asks for its
    logger through this function instead of ``logging.getLogger``.
    """
    logger = logging.getLogger(name)
    if _FILTER not in logger.filters:
        logger.addFilter(_FILTER)
    return logger
