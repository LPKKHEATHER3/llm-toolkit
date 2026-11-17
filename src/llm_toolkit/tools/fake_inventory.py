"""An in-memory stand-in for the inventory-manager REST API, served through ``httpx.MockTransport``.

It mirrors the routes the tools use, with the same status codes and error bodies
(``{"error": ..., "details": {...}}``), so the tools, the eval harness, the examples' ``--offline``
mode and the tests can run without the real server or a network. All items are fictional.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from typing import Any

import httpx

from .inventory import undo_label

SKU = r"(?P<sku>[A-Z0-9]{1,8}-\d{1,8})"

DEFAULT_ITEMS: list[dict[str, Any]] = [
    {
        "sku": "ELE-0001",
        "name": "Desk Lamp LED",
        "category": "Electronics",
        "price": 450.0,
        "qty": 12,
        "reorder_level": 4,
        "location": "Aisle 1",
    },
    {
        "sku": "ELE-0002",
        "name": "USB-C Charger 20W",
        "category": "Electronics",
        "price": 290.0,
        "qty": 3,
        "reorder_level": 5,
        "location": "Aisle 1",
    },
    {
        "sku": "ELE-0003",
        "name": "Wired Keyboard",
        "category": "Electronics",
        "price": 520.0,
        "qty": 9,
        "reorder_level": 3,
        "location": "Aisle 2",
    },
    {
        "sku": "ELE-0004",
        "name": "Bluetooth Speaker Mini",
        "category": "Electronics",
        "price": 790.0,
        "qty": 0,
        "reorder_level": 2,
        "location": "Aisle 2",
    },
    {
        "sku": "STA-0001",
        "name": "Graph Paper Pad A4",
        "category": "Stationery",
        "price": 55.0,
        "qty": 80,
        "reorder_level": 20,
        "location": "Shelf 4",
    },
    {
        "sku": "STA-0002",
        "name": "Fineliner Pen Green",
        "category": "Stationery",
        "price": 22.0,
        "qty": 15,
        "reorder_level": 25,
        "location": "Shelf 4",
    },
    {
        "sku": "OSU-0001",
        "name": "Binder Clips (24)",
        "category": "Office Supplies",
        "price": 48.0,
        "qty": 40,
        "reorder_level": 10,
        "location": "Shelf 6",
    },
    {
        "sku": "OSU-0002",
        "name": "Label Tape 12mm",
        "category": "Office Supplies",
        "price": 160.0,
        "qty": 4,
        "reorder_level": 6,
        "location": "Shelf 6",
    },
]


class FakeInventoryBackend:
    """Implements GET /api/items, GET /api/items/<sku>, POST /api/items/<sku>/move, GET /api/stats
    and POST /api/undo (with the optional ``{"expected": label}`` guard)."""

    def __init__(
        self,
        items: list[dict[str, Any]] | None = None,
        *,
        fail: Callable[[httpx.Request], httpx.Response | Exception | None] | None = None,
    ):
        self.items: dict[str, dict[str, Any]] = {i["sku"]: copy.deepcopy(i) for i in (items or DEFAULT_ITEMS)}
        self.movements: list[dict[str, Any]] = []
        self.undo_stack: list[
            tuple[str, dict[str, Any] | None, int]
        ] = []  # (label, before-image, movements_len)
        self.requests: list[httpx.Request] = []
        self.fail = fail  # optional hook: return a response or exception to inject a failure

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def push_foreign_change(self, sku: str, delta: int, reason: str = "another user") -> None:
        """Simulate a change made by someone else (it goes on top of the global undo stack)."""
        self._move(sku, delta, reason)

    # ---- helpers -------------------------------------------------------------------------------

    @staticmethod
    def _json(status: int, body: Any) -> httpx.Response:
        return httpx.Response(status, json=body)

    def _move(self, sku: str, delta: int, reason: str) -> dict[str, Any]:
        item = self.items[sku]
        before = copy.deepcopy(item)
        item["qty"] += delta
        movement = {
            "id": len(self.movements) + 1,
            "sku": sku,
            "delta": delta,
            "reason": reason,
            "qty_after": item["qty"],
            "at": "2026-01-01T00:00:00+00:00",
        }
        self.undo_stack.append((undo_label(delta, sku), before, len(self.movements)))
        self.movements.append(movement)
        return movement

    # ---- routing -------------------------------------------------------------------------------

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail is not None:
            injected = self.fail(request)
            if isinstance(injected, Exception):
                raise injected
            if injected is not None:
                return injected
        path, method = request.url.path, request.method

        if path == "/api/items" and method == "GET":
            q = request.url.params.get("q", "").casefold().strip()
            cat = request.url.params.get("category", "")
            low = request.url.params.get("low") == "1"
            out = []
            for it in self.items.values():
                if cat and it["category"] != cat:
                    continue
                if low and it["qty"] > it["reorder_level"]:
                    continue
                if q and q not in " ".join((it["sku"], it["name"], it.get("location", ""))).casefold():
                    continue
                out.append(copy.deepcopy(it))
            return self._json(200, sorted(out, key=lambda i: i["sku"]))

        if m := re.fullmatch(rf"/api/items/{SKU}", path):
            if method != "GET":
                return self._json(405, {"error": "use GET"})
            sku = m["sku"]
            if sku not in self.items:
                return self._json(404, {"error": f"{sku} not found"})
            return self._json(200, copy.deepcopy(self.items[sku]))

        if m := re.fullmatch(rf"/api/items/{SKU}/move", path):
            if method != "POST":
                return self._json(405, {"error": "use POST"})
            try:
                body = json.loads(request.content or b"{}")
            except ValueError:
                return self._json(400, {"error": "body is not valid JSON"})
            delta, reason = body.get("delta"), str(body.get("reason") or "").strip()
            errors = {}
            if isinstance(delta, bool) or not isinstance(delta, int) or delta == 0:
                errors["delta"] = "must be a non-zero whole number"
            if not reason:
                errors["reason"] = "is required"
            if errors:
                return self._json(400, {"error": "validation failed", "details": errors})
            sku = m["sku"]
            if sku not in self.items:
                return self._json(404, {"error": f"{sku} not found"})
            qty = self.items[sku]["qty"]
            if qty + delta < 0:
                return self._json(409, {"error": f"only {qty} in stock, cannot issue {-delta}"})
            return self._json(201, self._move(sku, delta, reason))

        if path == "/api/stats" and method == "GET":
            items = list(self.items.values())
            return self._json(
                200,
                {
                    "products": len(items),
                    "units": sum(i["qty"] for i in items),
                    "low_stock": sum(1 for i in items if i["qty"] <= i["reorder_level"]),
                    "undo": self.undo_stack[-1][0] if self.undo_stack else None,
                },
            )

        if path == "/api/undo" and method == "POST":
            try:
                body = json.loads(request.content or b"{}")
            except ValueError:
                return self._json(400, {"error": "body is not valid JSON"})
            if not self.undo_stack:
                return self._json(409, {"error": "nothing to undo"})
            current = self.undo_stack[-1][0]
            expected = body.get("expected") if isinstance(body, dict) else None
            if expected is not None and expected != current:
                return self._json(
                    409,
                    {
                        "error": f"the most recent change is {current!r}, not {expected!r}; nothing was undone",
                        "current": current,
                    },
                )
            label, before, mlen = self.undo_stack.pop()
            if before is not None:
                self.items[before["sku"]] = before
            del self.movements[mlen:]
            return self._json(
                200, {"undone": label, "next": self.undo_stack[-1][0] if self.undo_stack else None}
            )

        return self._json(404, {"error": "no such endpoint"})
