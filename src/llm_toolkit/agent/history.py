"""Token-budgeted history trimming that never splits a tool call from its results.

The conversation is cut into atomic *units*:

* a ``user`` message,
* a plain ``assistant`` message,
* an ``assistant`` message with ``tool_calls`` **plus every** ``tool`` reply to it.

Units are kept newest-first while they fit the budget. The current turn (the last ``user`` message
and everything after it) is always kept, and the window always starts at a ``user`` message. Orphan
``tool`` messages and tool-call units missing a reply are dropped, because providers reject both.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from ..client.types import Message

TokenCounter = Callable[[Message], int]


def approx_tokens(message: Message) -> int:
    """Cheap estimate (~4 characters per token plus per-message overhead). Good enough for a budget;
    swap in a real tokenizer through ``count_tokens`` if you need precision."""
    chars = len(message.content or "")
    for tc in message.tool_calls:
        chars += len(tc.name) + len(tc.arguments)
    return 4 + math.ceil(chars / 4)


def split_units(messages: Sequence[Message]) -> list[list[Message]]:
    """Group messages into atomic units, dropping orphan/incomplete tool exchanges."""
    units: list[list[Message]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            wanted = {tc.id for tc in m.tool_calls}
            unit = [m]
            j = i + 1
            while j < n and messages[j].role == "tool":
                unit.append(messages[j])
                j += 1
            got = {t.tool_call_id for t in unit[1:]}
            if got == wanted and len(unit) - 1 == len(wanted):
                units.append(unit)
            i = j
        elif m.role == "tool":
            i += 1  # orphan tool result: its parent is gone
        elif m.role == "system":
            i += 1  # the agent supplies the system prompt separately
        else:
            units.append([m])
            i += 1
    return units


def trim_history(
    messages: Sequence[Message],
    budget_tokens: int,
    count_tokens: TokenCounter = approx_tokens,
) -> list[Message]:
    units = split_units(messages)
    if not units:
        return []
    last_user = max((k for k, u in enumerate(units) if u[0].role == "user"), default=None)
    if last_user is None:
        return []

    def cost(unit: list[Message]) -> int:
        return sum(count_tokens(m) for m in unit)

    kept = units[last_user:]  # the current turn is never cut
    used = sum(cost(u) for u in kept)
    k = last_user - 1
    while k >= 0:
        c = cost(units[k])
        if used + c > budget_tokens:
            break
        kept.insert(0, units[k])
        used += c
        k -= 1
    while kept and kept[0][0].role != "user":
        kept.pop(0)
    return [m for unit in kept for m in unit]
