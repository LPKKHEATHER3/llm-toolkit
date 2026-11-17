---
name: inventory-stock
description: Manage shop inventory through the llm-toolkit MCP tools (list_items, get_item, issue_stock, undo_last_change, search_policies). Use when the user asks how much stock is left, wants to record a sale, write-off or transfer, wants to undo a stock change, or asks about store policy on returns, shipping, warranty or reservations.
---

# Managing stock with the llm-toolkit MCP tools

These tools talk to an inventory-manager backend. Every stock change is a recorded movement, stock
can never go below zero, and the undo stack is shared by everyone using the backend. Work so that a
mistake is visible and reversible.

## Tools

| Tool | Use it to | Notes |
|---|---|---|
| `list_items` | find items by text, category, or low stock | all filters optional; send only the ones you need |
| `get_item` | read one item by SKU | SKU looks like `ELE-0003` |
| `issue_stock` | take units out (sale, damaged, transfer) | `quantity` is positive; `reason` is required; returns `undo_label` |
| `undo_last_change` | reverse the most recent change | needs `expected_label`; refused if the top change is someone else's |
| `search_policies` | look up store policy text | returns excerpts with `chunk_id`s to quote |

## Workflow for a stock change

1. **Identify the item.** If the user gave a name, call `list_items` with a short `query` and pick
   the matching SKU. If several items match, ask which one; do not guess.
2. **Check before you change.** Call `get_item` when the quantity matters to the decision (for
   example a large issue).
3. **Issue.** Call `issue_stock` with the SKU, a positive `quantity`, and a short `reason` in the
   user's words ("sale", "damaged", "transfer to branch").
4. **Report from the tool result.** Quote `qty_after` from the result. Keep the `undo_label`; you
   need it to undo.

## Reading errors

Tool errors are written for you to act on. Never tell the user an action succeeded when the tool
returned an error.

| Error text | Meaning | What to do |
|---|---|---|
| `only N in stock, cannot issue M` | not enough stock; nothing changed | tell the user N is available; offer to issue N or to check the shelf |
| `XXX-0000 not found` | wrong SKU | search with `list_items` and confirm the item |
| `invalid arguments: ...` / `invalid request: ...` | a field is missing or has the wrong type | fix the named field and call again |
| `outcome unknown: ...` | the backend did not answer in time | call `get_item` to see whether the change happened **before** retrying; never repeat `issue_stock` blindly |
| `inventory service unavailable` | backend down | tell the user; do not retry in a loop |
| `refused: the most recent change is ...` | the top undo entry is not yours | do not undo; tell the user someone else changed stock since, and suggest a supervisor |

## Undo

Only undo a change you made in this conversation. Pass the `undo_label` from that `issue_stock`
result as `expected_label` (for example `-3 ELE-0003`). If the tool refuses, stop: the change on
top of the shared undo stack belongs to someone else.

## Policy questions

Call `search_policies` with the user's question. Answer only from the returned excerpts, quote the
sentence you rely on word for word, and mention its `chunk_id`. If no excerpt answers the question,
say the policy does not cover it.

## Example

User: "We sold two of the USB-C chargers."

1. `list_items` `{"query": "USB-C"}` → `ELE-0002`, qty 3
2. `issue_stock` `{"sku": "ELE-0002", "quantity": 2, "reason": "sale"}` → `qty_after: 1`, `undo_label: "-2 ELE-0002"`
3. Reply: "Recorded the sale of 2 USB-C Charger 20W (ELE-0002). 1 left, which is below its reorder
   level of 5."
