"""Chunking, BM25 retrieval, cited answers and quote verification."""

from __future__ import annotations

from functools import lru_cache

from .answer import CitedAnswer, answer_with_citations, build_messages, parse_answer
from .bm25 import BM25Index, EmbeddingRetriever, Hit, Retriever, tokenize
from .chunking import Chunk, by_id, chunk_document, default_corpus_dir, load_corpus
from .verify import Citation, DroppedCitation, RawAnswer, Verification, verify_citations


@lru_cache(maxsize=1)
def default_index() -> BM25Index:
    """BM25 index over the bundled fictional policy documents."""
    return BM25Index.build(load_corpus())


def search(query: str, k: int = 4) -> list[Hit]:
    """``rag.search``: top-k chunks from the default policy index."""
    return default_index().search(query, k=k)


__all__ = [
    "BM25Index",
    "Chunk",
    "Citation",
    "CitedAnswer",
    "DroppedCitation",
    "EmbeddingRetriever",
    "Hit",
    "RawAnswer",
    "Retriever",
    "Verification",
    "answer_with_citations",
    "build_messages",
    "by_id",
    "chunk_document",
    "default_corpus_dir",
    "default_index",
    "load_corpus",
    "parse_answer",
    "search",
    "tokenize",
    "verify_citations",
]
