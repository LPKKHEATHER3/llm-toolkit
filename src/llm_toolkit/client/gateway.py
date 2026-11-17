"""The single definition of "a gateway error disguised as model output".

An HTTP proxy or load balancer in front of a provider can return an error *inside*
``message.content`` with HTTP 200, so it looks like an answer. The earlier prototype had two copies
of this check with different patterns; this module is the only one. Everything that needs the check
imports :func:`looks_like_gateway_error`.
"""

from __future__ import annotations

import re

# Kept deliberately narrow: a false positive makes the client fail over and costs a request,
# a false negative shows an error string to the user as if the model had said it.
GATEWAY_ERROR_PATTERNS: tuple[str, ...] = (
    r"\brate[ _-]?limit(?:ed)? exceeded\b",
    r"\bupstream (?:connect )?error\b",
    r"\bbad gateway\b",
)

_GATEWAY_RE = re.compile("|".join(GATEWAY_ERROR_PATTERNS), re.IGNORECASE)

# Only short contents are suspicious: a real answer that happens to mention "bad gateway" in a
# paragraph about HTTP should not be discarded.
_MAX_SUSPICIOUS_LEN = 400


def looks_like_gateway_error(content: str | None) -> bool:
    if not content:
        return False
    text = content.strip()
    return len(text) <= _MAX_SUSPICIOUS_LEN and bool(_GATEWAY_RE.search(text))
