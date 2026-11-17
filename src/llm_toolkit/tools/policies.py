"""``search_policies``: the RAG retriever exposed as a tool."""

from __future__ import annotations

from pydantic import Field

from ..rag.bm25 import Retriever
from .registry import Tool, ToolArgs, provided

DEFAULT_K = 4


class SearchPoliciesArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=200, description="What to look up in the store policies")
    k: int | None = Field(default=None, ge=1, le=8, description="How many excerpts to return (default 4)")


def policy_tools(retriever: Retriever) -> list[Tool]:
    def search_policies(args: SearchPoliciesArgs) -> dict:
        k = provided(args).get("k") or DEFAULT_K
        hits = retriever.search(args.query, k=k)
        return {
            "results": [
                {"chunk_id": h.chunk.id, "section": h.chunk.section, "text": h.chunk.text, "score": h.score}
                for h in hits
            ],
            "note": "Quote excerpts word for word and cite their chunk_id."
            if hits
            else "No matching policy text.",
        }

    return [
        Tool(
            "search_policies",
            "Search the store's policy documents (returns, shipping, warranty, stock, payments). "
            "Returns excerpts with ids to quote.",
            SearchPoliciesArgs,
            search_policies,
        )
    ]
