"""Planted-defect dataset for the citation verifier.

``build_planted_set`` turns a fixed list of questions into answers whose citations are either
genuine (copied verbatim from a retrieved chunk) or carry one deliberate defect. The output is
written to ``data/eval/planted_citations.json`` so the dataset is reviewable data, and
``run_verifier_bench`` measures how many defects the verifier drops and how many genuine quotes it
keeps.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..rag import BM25Index, Chunk, Hit, by_id, load_corpus
from ..rag.verify import Citation, RawAnswer, check_citation

# question -> (paraphrase of a real policy sentence, fabricated policy sentence)
QUESTIONS: list[tuple[str, str, str]] = [
    (
        "How long do I have to return opened electronics?",
        "Opened electronics may be sent back within two weeks.",
        "Opened electronics can be returned within 60 days of delivery.",
    ),
    (
        "Is there a fee for returning an opened item?",
        "Opened items get a refund minus a fifteen percent fee.",
        "Every return is charged a flat fee of 50 baht.",
    ),
    (
        "When will my refund arrive?",
        "Money goes back to the card within five working days.",
        "Refunds are paid in cash at the store on the same day.",
    ),
    (
        "How much does express delivery cost?",
        "Next-day shipping is 120 baht and never free.",
        "Express delivery is free for members.",
    ),
    (
        "What is the minimum order for free standard delivery?",
        "Orders above 800 baht ship for free.",
        "Standard delivery is free for every order over 300 baht.",
    ),
    (
        "How long is the warranty on a keyboard?",
        "Keyboards are guaranteed for two years.",
        "Keyboards carry a lifetime warranty.",
    ),
    (
        "What should I do with a swollen power bank?",
        "A swollen power bank should be brought to the shop, not mailed.",
        "Swollen power banks must be mailed back in the original box.",
    ),
    (
        "How long is a reservation held?",
        "Reserved items are kept for two days.",
        "Reservations are held for 14 days.",
    ),
    (
        "How much is a loyalty point worth?",
        "One point is worth a quarter of a baht.",
        "Each point is worth 1 baht at checkout.",
    ),
    (
        "What happens if an issue would take stock below zero?",
        "Movements that would make stock negative are rejected.",
        "Stock below zero is allowed and corrected at the next count.",
    ),
]

EXPECTED_REASON = {
    "paraphrase": "quote_not_found",
    "number_changed": "quote_not_found",
    "negation_flipped": "quote_not_found",
    "case_changed": "quote_not_found",
    "whitespace_changed": "quote_not_found",
    "fabricated": "quote_not_found",
    "wrong_chunk": "quote_not_found",
    "not_retrieved": "not_retrieved",
    "unknown_chunk": "unknown_chunk",
    "empty": "empty",
    "too_short": "too_short",
}

_SENTENCE = re.compile(r"[^.!?\n]*[.!?]")


def corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c.id}\x00{c.text}\x00".encode())
    return h.hexdigest()[:16]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if len(s.strip()) >= 20]


def _flip_negation(s: str) -> str | None:
    for a, b in ((" cannot ", " can "), (" not ", " "), (" never ", " always ")):
        if a in s:
            return s.replace(a, b, 1)
    for a, b in ((" can ", " cannot "), (" is ", " is not "), (" are ", " are not ")):
        if a in s:
            return s.replace(a, b, 1)
    return None


def _change_number(s: str) -> str | None:
    m = re.search(r"\d+", s)
    if not m:
        return None
    return s[: m.start()] + str(int(m.group()) + 1) + s[m.end() :]


def build_planted_set(chunks: list[Chunk] | None = None, k: int = 3) -> dict[str, Any]:
    chunks = chunks or load_corpus()
    index = BM25Index.build(chunks)
    answers = []
    for n, (question, paraphrase, fabricated) in enumerate(QUESTIONS, start=1):
        hits = index.search(question, k=k)
        retrieved = [h.chunk.id for h in hits]
        top = hits[0].chunk
        s = max(_sentences(top.text), key=len)
        cites: list[dict[str, Any]] = [{"chunk_id": top.id, "quote": s, "label": "genuine"}]
        for h in hits[1:2]:
            extra = _sentences(h.chunk.text)
            if extra:
                cites.append({"chunk_id": h.chunk.id, "quote": extra[0], "label": "genuine"})
        cites.append({"chunk_id": top.id, "quote": paraphrase, "label": "paraphrase"})
        cites.append({"chunk_id": top.id, "quote": fabricated, "label": "fabricated"})
        if (changed := _change_number(s)) is not None:
            cites.append({"chunk_id": top.id, "quote": changed, "label": "number_changed"})
        if (flipped := _flip_negation(s)) is not None and flipped not in top.text:
            cites.append({"chunk_id": top.id, "quote": flipped, "label": "negation_flipped"})
        cites.append(
            {"chunk_id": top.id, "quote": s.upper() if s.upper() != s else s.lower(), "label": "case_changed"}
        )
        cites.append({"chunk_id": top.id, "quote": s.replace(" ", "  ", 1), "label": "whitespace_changed"})
        others = [h.chunk for h in hits[1:] if s not in h.chunk.text]
        if others:
            cites.append({"chunk_id": others[0].id, "quote": s, "label": "wrong_chunk"})
        outside = next(c for c in chunks if c.id not in retrieved and c.doc_id != top.doc_id)
        cites.append({"chunk_id": outside.id, "quote": _sentences(outside.text)[0], "label": "not_retrieved"})
        cites.append({"chunk_id": f"{top.doc_id}#99", "quote": s, "label": "unknown_chunk"})
        cites.append({"chunk_id": top.id, "quote": "", "label": "empty"})
        # Verbatim but shorter than the verifier's 8-character minimum: proves nothing.
        cites.append({"chunk_id": top.id, "quote": s[:5], "label": "too_short"})
        for c in cites:
            c["expected"] = "keep" if c["label"] == "genuine" else "drop"
        answers.append(
            {"id": f"planted-{n:02d}", "question": question, "retrieved": retrieved, "citations": cites}
        )
    return {
        "description": "Answers with genuine and deliberately defective citations, generated by "
        "llm_toolkit.eval.planted.build_planted_set from the fictional policy corpus.",
        "corpus_hash": corpus_hash(chunks),
        "answers": answers,
    }


def run_verifier_bench(dataset_path: Path, chunks: list[Chunk] | None = None) -> dict[str, Any]:
    chunks = chunks or load_corpus()
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if data["corpus_hash"] != corpus_hash(chunks):
        raise ValueError("the policy corpus changed since the dataset was built; run `build-planted` again")
    corpus = by_id(chunks)
    by_label: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0, "expected": ""})
    reason_matches = 0
    totals: Counter[str] = Counter()
    for ans in data["answers"]:
        hits = [Hit(corpus[cid], 1.0) for cid in ans["retrieved"]]
        retrieved = {h.chunk.id: h.chunk for h in hits}
        raw = RawAnswer(
            answer="",
            citations=[Citation(chunk_id=c["chunk_id"], quote=c["quote"]) for c in ans["citations"]],
        )
        for c, cite in zip(ans["citations"], raw.citations, strict=True):
            reason = check_citation(cite, retrieved, corpus)
            outcome = "keep" if reason is None else "drop"
            stats = by_label[c["label"]]
            stats["total"] += 1
            stats["expected"] = c["expected"]
            stats["correct"] += outcome == c["expected"]
            totals[c["expected"]] += 1
            if c["expected"] == "drop" and reason == EXPECTED_REASON.get(c["label"]):
                reason_matches += 1
    defects = totals["drop"]
    genuine = totals["keep"]
    defects_dropped = sum(s["correct"] for s in by_label.values() if s["expected"] == "drop")
    genuine_kept = sum(s["correct"] for s in by_label.values() if s["expected"] == "keep")
    return {
        "run_id": f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}",
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset_path.name,
        "corpus_hash": data["corpus_hash"],
        "answers": len(data["answers"]),
        "citations": defects + genuine,
        "defects": defects,
        "defects_dropped": defects_dropped,
        "genuine": genuine,
        "genuine_kept": genuine_kept,
        "reason_matches": reason_matches,
        "by_label": dict(by_label),
    }
