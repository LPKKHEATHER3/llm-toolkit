"""Retrieve, ask the model for a JSON answer with quotes, verify the quotes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..client.types import ChatRequest, ChatResponse, Message, Usage
from .bm25 import Hit, Retriever
from .chunking import Chunk
from .verify import Citation, DroppedCitation, RawAnswer, verify_citations

SYSTEM_PROMPT = """You answer questions about a store's policies.
Use ONLY the excerpts provided in the user message. Each excerpt starts with its id in square brackets.

Reply with one JSON object and nothing else:
{"answer": "<short answer>", "citations": [{"chunk_id": "<excerpt id>", "quote": "<exact text>"}]}

Rules:
- Every quote must be copied character for character from the excerpt named by chunk_id. Do not
  paraphrase, shorten words, or change punctuation inside a quote.
- Cite at least one quote for every claim in the answer.
- If the excerpts do not answer the question, say so in "answer" and return an empty citations list."""


class CitedAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    dropped: list[DroppedCitation]
    retrieved: list[str]
    parse_error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    provider: str = ""
    latency_ms: float = 0.0

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    @property
    def grounded(self) -> bool:
        return bool(self.citations)


def format_excerpts(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk.id}] ({h.chunk.section})\n{h.chunk.text}" for h in hits)


def build_messages(question: str, hits: Sequence[Hit]) -> list[Message]:
    body = (
        f"Excerpts:\n\n{format_excerpts(hits)}\n\nQuestion: {question}"
        if hits
        else (f"Excerpts: (none found)\n\nQuestion: {question}")
    )
    return [Message.system(SYSTEM_PROMPT), Message.user(body)]


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_answer(text: str | None) -> RawAnswer:
    """Parse the model's JSON. On failure the raw text becomes the answer with no citations, and the
    parse error is recorded (it is never silently treated as a grounded answer)."""
    raw = (text or "").strip()
    candidate = _FENCE_RE.sub("", raw).strip()
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        candidate = candidate[start : end + 1] if 0 <= start < end else candidate
    try:
        data: Any = json.loads(candidate)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
        return RawAnswer.model_validate(data)
    except (ValueError, ValidationError) as e:
        return RawAnswer(answer=raw, citations=[], parse_error=f"{type(e).__name__}: {str(e)[:200]}")


def answer_with_citations(
    client: Any,
    retriever: Retriever,
    question: str,
    *,
    k: int = 4,
    corpus: Mapping[str, Chunk] | None = None,
    deadline_s: float = 60.0,
    max_tokens: int = 1024,
) -> CitedAnswer:
    hits = retriever.search(question, k=k)
    resp: ChatResponse = client.chat(
        ChatRequest(messages=build_messages(question, hits), deadline_s=deadline_s, max_tokens=max_tokens)
    )
    raw = parse_answer(resp.message.content)
    check = verify_citations(raw, hits, corpus)
    return CitedAnswer(
        question=question,
        answer=raw.answer,
        citations=check.kept,
        dropped=check.dropped,
        retrieved=[h.chunk.id for h in hits],
        parse_error=raw.parse_error,
        usage=resp.usage,
        provider=resp.provider,
        latency_ms=resp.latency_ms,
    )
