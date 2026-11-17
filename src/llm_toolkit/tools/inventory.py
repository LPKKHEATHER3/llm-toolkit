"""Inventory tools over the inventory-manager REST API.

Backend errors are turned into messages the model can act on:

* 400 -> ``invalid request: field: message; ...`` (the backend's per-field details)
* 404 -> the backend's message (e.g. ``ELE-0099 not found``)
* 409 -> the backend's message verbatim (e.g. ``only 4 in stock, cannot issue 50``)
* 5xx / unreachable -> "inventory service unavailable"

Reads (GET) are retried once on a transport error or 5xx. Writes (POST) are **never** retried: a
stock movement is not idempotent, so a timeout is reported as "outcome unknown, check before
retrying" instead of risking a double issue.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import Field

from ..redact import get_logger
from .registry import Tool, ToolArgs, ToolError, provided

log = get_logger(__name__)

SKU_PATTERN = r"^[A-Z0-9]{1,8}-\d{1,8}$"
DEFAULT_URL = "http://127.0.0.1:8000"
LIST_LIMIT = 50


def undo_label(delta: int, sku: str) -> str:
    """The label inventory-manager puts on the undo stack for a stock movement."""
    return f"{'+' if delta > 0 else ''}{delta} {sku}"


class InventoryAPI:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: httpx.Timeout | None = None,
        get_retries: int = 1,
    ):
        self.base_url = (base_url or os.environ.get("INVENTORY_API_URL") or DEFAULT_URL).rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            transport=transport,
            timeout=timeout or httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0),
        )
        self.get_retries = get_retries

    def close(self) -> None:
        self._http.close()

    # ---- transport -------------------------------------------------------------------------

    @staticmethod
    def _error_text(resp: httpx.Response) -> str:
        try:
            body = resp.json()
        except ValueError:
            return resp.text[:200] or f"HTTP {resp.status_code}"
        if not isinstance(body, dict):
            return str(body)[:200]
        message = str(body.get("error") or f"HTTP {resp.status_code}")
        details = body.get("details")
        if isinstance(details, dict) and details:
            message += ": " + "; ".join(f"{k}: {v}" for k, v in details.items())
        return message

    def _raise_for(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        text = self._error_text(resp)
        if resp.status_code == 400:
            raise ToolError(f"invalid request: {text}")
        if resp.status_code in (404, 409):
            raise ToolError(text)
        if resp.status_code >= 500:
            raise ToolError(f"inventory service unavailable (HTTP {resp.status_code}); try again later")
        raise ToolError(f"inventory service rejected the request (HTTP {resp.status_code}): {text}")

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        last: Exception | None = None
        for attempt in range(1 + self.get_retries):
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as e:
                last = e
                log.warning("inventory GET %s failed (%s), attempt %d", path, type(e).__name__, attempt + 1)
                continue
            if resp.status_code >= 500 and attempt < self.get_retries:
                continue
            self._raise_for(resp)
            return resp.json()
        raise ToolError(f"inventory service unavailable ({type(last).__name__}); try again later")

    def _post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        try:
            resp = self._http.post(path, json=body or {})
        except httpx.ConnectError as e:
            raise ToolError(
                f"could not reach the inventory service ({type(e).__name__}); nothing was changed"
            ) from e
        except httpx.TransportError as e:
            raise ToolError(
                "outcome unknown: the inventory service did not answer in time. "
                "Check the item with get_item before trying again; do not repeat the call blindly."
            ) from e
        self._raise_for(resp)
        return resp.json()

    # ---- API ---------------------------------------------------------------------------------

    def list_items(
        self, query: str | None = None, category: str | None = None, low_only: bool = False
    ) -> list:
        params: dict[str, str] = {}
        if query:
            params["q"] = query
        if category:
            params["category"] = category
        if low_only:
            params["low"] = "1"
        return self._get("/api/items", params)

    def get_item(self, sku: str) -> dict:
        return self._get(f"/api/items/{sku}")

    def move(self, sku: str, delta: int, reason: str) -> dict:
        return self._post(f"/api/items/{sku}/move", {"delta": delta, "reason": reason})

    def stats(self) -> dict:
        return self._get("/api/stats")

    def undo(self, expected: str | None = None) -> dict:
        return self._post("/api/undo", {"expected": expected} if expected is not None else None)


# ---- tool argument models -----------------------------------------------------------------------


class ListItemsArgs(ToolArgs):
    query: str | None = Field(
        default=None, max_length=80, description="Text to search in SKU, name, barcode or location"
    )
    category: str | None = Field(default=None, max_length=120, description="Exact category name")
    low_stock_only: bool | None = Field(
        default=None, description="Only items at or below their reorder level"
    )


class GetItemArgs(ToolArgs):
    sku: str = Field(pattern=SKU_PATTERN, description="Item SKU, e.g. ELE-0003")


class IssueStockArgs(ToolArgs):
    sku: str = Field(pattern=SKU_PATTERN, description="Item SKU, e.g. ELE-0003")
    quantity: int = Field(gt=0, le=100_000, description="Units to take out of stock (positive)")
    reason: str = Field(min_length=1, max_length=120, description='Why, e.g. "sale" or "damaged"')


class UndoArgs(ToolArgs):
    expected_label: str = Field(
        min_length=3,
        max_length=60,
        description='Label of the change to undo, as returned by issue_stock (e.g. "-3 ELE-0003")',
    )


def _compact(item: dict) -> dict:
    keys = ("sku", "name", "category", "qty", "reorder_level", "price", "location")
    return {k: item.get(k) for k in keys}


def inventory_tools(api: InventoryAPI) -> list[Tool]:
    def list_items(args: ListItemsArgs) -> dict:
        sent = provided(args)
        items = api.list_items(sent.get("query"), sent.get("category"), bool(sent.get("low_stock_only")))
        return {
            "total": len(items),
            "items": [_compact(i) for i in items[:LIST_LIMIT]],
            "truncated": len(items) > LIST_LIMIT,
        }

    def get_item(args: GetItemArgs) -> dict:
        return _compact(api.get_item(args.sku))

    def issue_stock(args: IssueStockArgs) -> dict:
        movement = api.move(args.sku, -args.quantity, args.reason)
        return {
            "sku": args.sku,
            "issued": args.quantity,
            "qty_after": movement.get("qty_after"),
            "movement_id": movement.get("id"),
            "undo_label": undo_label(-args.quantity, args.sku),
        }

    def undo_last_change(args: UndoArgs) -> dict:
        # The backend's undo stack is global. inventory-manager compares `expected` with the top of
        # the stack under the same lock that performs the undo, so there is no check-then-act race.
        try:
            result = api.undo(expected=args.expected_label)
        except ToolError as e:
            if "most recent change is" in str(e):
                raise ToolError(f"refused: {e}. Ask a person to resolve it.") from None
            raise
        return {"undone": result.get("undone"), "next_undo": result.get("next")}

    return [
        Tool(
            "list_items",
            "List or search inventory items. All filters are optional.",
            ListItemsArgs,
            list_items,
        ),
        Tool("get_item", "Get one item by SKU, including its current quantity.", GetItemArgs, get_item),
        Tool(
            "issue_stock",
            "Take units out of stock (sale, damage, transfer). Refused if there is not enough stock. "
            "Returns the undo label for this change.",
            IssueStockArgs,
            issue_stock,
        ),
        Tool(
            "undo_last_change",
            "Undo the most recent inventory change, only if it is the change named by expected_label. "
            "Refused when the most recent change is a different one.",
            UndoArgs,
            undo_last_change,
        ),
    ]
