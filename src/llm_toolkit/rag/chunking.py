"""Split documents into chunks whose text is an exact slice of the source.

Keeping ``chunk.text == document[start:end]`` matters for the citation verifier: a quote that is
found in a chunk is guaranteed to be found, character for character, in the original document.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    section: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Segment:
    start: int
    end: int
    heading: bool


_BLOCK_RE = re.compile(r"(?:[^\n]*\S[^\n]*(?:\n|$))+")
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?=\s|$)|[^.!?]+$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s")


def _segments(text: str) -> list[_Segment]:
    """Headings, list items and sentences, each with exact offsets."""
    out: list[_Segment] = []
    for block in _BLOCK_RE.finditer(text):
        b_start, b_text = block.start(), block.group()
        lines = b_text.splitlines(keepends=True)
        if lines and (lines[0].lstrip().startswith("#") or _LIST_ITEM_RE.match(lines[0])):
            pos = b_start
            for line in lines:
                stripped = line.strip()
                if stripped:
                    s = pos + (len(line) - len(line.lstrip()))
                    out.append(_Segment(s, s + len(stripped), stripped.startswith("#")))
                pos += len(line)
            continue
        for m in _SENTENCE_RE.finditer(b_text):
            raw = m.group()
            if not raw.strip():
                continue
            lead = len(raw) - len(raw.lstrip())
            s = b_start + m.start() + lead
            out.append(_Segment(s, s + len(raw.strip()), False))
    return out


def chunk_document(doc_id: str, text: str, *, max_chars: int = 600, overlap: int = 1) -> list[Chunk]:
    """Greedy packing of segments up to ``max_chars``. A heading always starts a new chunk and is
    remembered as the chunk's section. ``overlap`` sentences are repeated at the start of the next
    chunk within the same section so a fact split across a boundary is still retrievable."""
    segs = _segments(text)
    chunks: list[Chunk] = []
    section = ""
    i = 0
    while i < len(segs):
        if segs[i].heading:
            section = text[segs[i].start : segs[i].end].lstrip("#").strip()
            i += 1
            continue
        j = i
        while j + 1 < len(segs) and not segs[j + 1].heading and segs[j + 1].end - segs[i].start <= max_chars:
            j += 1
        start, end = segs[i].start, segs[j].end
        chunks.append(
            Chunk(
                id=f"{doc_id}#{len(chunks) + 1}",
                doc_id=doc_id,
                section=section,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
        nxt = j + 1
        if overlap and nxt < len(segs) and not segs[nxt].heading and j - i + 1 > overlap:
            nxt -= overlap
        i = nxt
    return chunks


def default_corpus_dir() -> Path:
    env = os.environ.get("LLM_TOOLKIT_POLICIES_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data" / "policies"


def load_corpus(directory: str | os.PathLike[str] | None = None, **chunk_kwargs: int) -> list[Chunk]:
    """Chunk every ``*.md`` file except README.md, in a stable (sorted) order."""
    root = Path(directory) if directory else default_corpus_dir()
    chunks: list[Chunk] = []
    for path in sorted(root.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), **chunk_kwargs))
    return chunks


def by_id(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    return {c.id: c for c in chunks}
