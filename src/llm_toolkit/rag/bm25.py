"""Pure-Python BM25 and the retriever interface.

``Retriever`` is the seam for swapping retrieval backends: ``BM25Index`` is the default, and
``EmbeddingRetriever`` accepts any ``embed(texts) -> vectors`` function, so an embedding model can be
plugged in later without touching the answer or verification code.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .chunking import Chunk

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "must",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "within",
        "without",
        "you",
        "your",
    ]
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Tiny suffix stripper: enough to match 'returns'/'return' and 'items'/'item'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int = 4) -> list[Hit]: ...


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        for idx, chunk in enumerate(self.chunks):
            # The section heading is indexed with the text: "Return window" helps "how long to return".
            tf = Counter(tokenize(f"{chunk.section} {chunk.text}"))
            self._tf.append(tf)
            self._len.append(sum(tf.values()))
            for term in tf:
                self._postings[term].append(idx)
        n = len(self.chunks)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        self._idf = {
            term: math.log((n - len(docs) + 0.5) / (len(docs) + 0.5) + 1.0)
            for term, docs in self._postings.items()
        }

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> BM25Index:
        return cls(list(chunks))

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for idx in self._postings[term]:
                tf = self._tf[idx][term]
                norm = 1 - self.b + self.b * (self._len[idx] / self._avgdl if self._avgdl else 0)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.chunks[kv[0]].id))
        return [Hit(self.chunks[i], round(s, 4)) for i, s in ranked[: max(0, k)] if s > 0]


Embed = Callable[[list[str]], list[list[float]]]


class EmbeddingRetriever:
    """Cosine similarity over vectors from any embedding function (no dependency on a vendor)."""

    def __init__(self, chunks: Sequence[Chunk], embed: Embed):
        self.chunks = list(chunks)
        self._embed = embed
        self._vectors = [self._normalise(v) for v in embed([f"{c.section} {c.text}" for c in self.chunks])]

    @staticmethod
    def _normalise(v: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else list(v)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        (q,) = self._embed([query])
        q = self._normalise(q)
        scored = [
            (sum(a * b for a, b in zip(q, v, strict=True)), c)
            for v, c in zip(self._vectors, self.chunks, strict=True)
        ]
        scored.sort(key=lambda sc: (-sc[0], sc[1].id))
        return [Hit(c, round(s, 4)) for s, c in scored[: max(0, k)] if s > 0]
