"""Citation verification: every quote must exist verbatim in a retrieved chunk.

A citation is kept only if

* its quote is at least ``min_quote_chars`` long after trimming outer whitespace (a one-word quote
  proves nothing),
* it names a chunk that exists,
* that chunk was among the hits retrieved for this question, and
* the quote is an exact substring of that chunk's text (no case folding, no whitespace or
  punctuation normalisation).

Everything else is dropped with a reason and counted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from .bm25 import Hit
from .chunking import Chunk

DropReason = Literal["empty", "too_short", "unknown_chunk", "not_retrieved", "quote_not_found"]


class Citation(BaseModel):
    chunk_id: str
    quote: str


class DroppedCitation(BaseModel):
    citation: Citation
    reason: DropReason


class RawAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    parse_error: str | None = None


class Verification(BaseModel):
    kept: list[Citation]
    dropped: list[DroppedCitation]

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    @property
    def grounded(self) -> bool:
        return bool(self.kept)


def check_citation(
    c: Citation,
    retrieved: Mapping[str, Chunk],
    corpus: Mapping[str, Chunk],
    min_quote_chars: int = 8,
) -> DropReason | None:
    quote = c.quote.strip()
    if not quote:
        return "empty"
    if len(quote) < min_quote_chars:
        return "too_short"
    if c.chunk_id not in corpus and c.chunk_id not in retrieved:
        return "unknown_chunk"
    chunk = retrieved.get(c.chunk_id)
    if chunk is None:
        return "not_retrieved"
    if quote not in chunk.text:
        return "quote_not_found"
    return None


def verify_citations(
    answer: RawAnswer,
    hits: Sequence[Hit],
    corpus: Mapping[str, Chunk] | None = None,
    *,
    min_quote_chars: int = 8,
) -> Verification:
    retrieved = {h.chunk.id: h.chunk for h in hits}
    corpus = corpus or {}
    kept: list[Citation] = []
    dropped: list[DroppedCitation] = []
    for c in answer.citations:
        reason = check_citation(c, retrieved, corpus, min_quote_chars)
        if reason is None:
            kept.append(c)
        else:
            dropped.append(DroppedCitation(citation=c, reason=reason))
    return Verification(kept=kept, dropped=dropped)
