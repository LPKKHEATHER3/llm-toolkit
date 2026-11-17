"""Answer a question about the fictional store's policies with BM25 retrieval and verified quotes.

    python examples/policy_agent.py "Can I return opened electronics?"            # needs a provider key
    python examples/policy_agent.py --offline "Can I return opened electronics?"  # scripted model, no key

In --offline mode a scripted stand-in plays the model: it quotes the first sentence of the top
excerpt (a genuine quote) and adds one invented quote, so you can watch the verifier drop it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from llm_toolkit.client import (
    ChatRequest,
    ChatResponse,
    LLMClient,
    LLMError,
    Message,
    Usage,
    default_providers,
)
from llm_toolkit.rag import answer_with_citations, by_id, default_index, load_corpus


class OfflineModel:
    """Scripted stand-in for a model (no network)."""

    def chat(self, req: ChatRequest) -> ChatResponse:
        prompt = req.messages[-1].content or ""
        found = re.search(r"\[([^\]]+)\] \([^)]*\)\n([^\n]+)", prompt)
        cites = []
        if found:
            chunk_id, first_line = found.groups()
            sentence = re.split(r"(?<=[.!?])\s", first_line)[0]
            cites = [
                {"chunk_id": chunk_id, "quote": sentence},
                {"chunk_id": chunk_id, "quote": "All items can be returned at any time, no questions asked."},
            ]
        answer = {
            "answer": "See the quoted policy." if cites else "The policy does not say.",
            "citations": cites,
        }
        return ChatResponse(
            message=Message.assistant(json.dumps(answer)),
            usage=Usage(),
            provider="offline",
            model="scripted",
            latency_ms=0.0,
            attempts=[],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--offline", action="store_true", help="use a scripted model instead of a provider")
    parser.add_argument("-k", type=int, default=4)
    args = parser.parse_args()

    client = OfflineModel() if args.offline else LLMClient(default_providers())
    try:
        out = answer_with_citations(
            client, default_index(), args.question, k=args.k, corpus=by_id(load_corpus())
        )
    except LLMError as e:
        print(f"model call failed: {e}", file=sys.stderr)
        return 1
    print(f"provider:  {out.provider}")
    print(f"retrieved: {', '.join(out.retrieved)}")
    print(f"answer:    {out.answer}")
    for c in out.citations:
        print(f"  verified  [{c.chunk_id}] {c.quote!r}")
    for d in out.dropped:
        print(f"  DROPPED   [{d.citation.chunk_id}] {d.citation.quote!r} ({d.reason})")
    print(f"grounded:  {out.grounded}  (dropped {out.dropped_count})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
