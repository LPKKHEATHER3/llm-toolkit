"""Chunking, BM25, the embedding seam, cited answers and the quote verifier."""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import pytest
from scripted_llm import ScriptedLLM, say

from llm_toolkit.rag import (
    BM25Index,
    Chunk,
    Citation,
    EmbeddingRetriever,
    RawAnswer,
    Retriever,
    answer_with_citations,
    by_id,
    chunk_document,
    default_corpus_dir,
    load_corpus,
    parse_answer,
    tokenize,
    verify_citations,
)

CORPUS = load_corpus()
INDEX = BM25Index.build(CORPUS)
CHUNKS = by_id(CORPUS)


# ---- chunking ---------------------------------------------------------------------------------


def test_chunks_are_exact_slices_of_the_source():
    for path in default_corpus_dir().glob("*.md"):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        for c in chunk_document(path.stem, text):
            assert text[c.start : c.end] == c.text


def test_headings_start_chunks_and_name_the_section():
    doc = "# Title\n\nIntro sentence here.\n\n## Part A\n\nAlpha one. Alpha two.\n\n## Part B\n\nBeta one."
    chunks = chunk_document("d", doc)
    assert [(c.section, c.text) for c in chunks] == [
        ("Title", "Intro sentence here."),
        ("Part A", "Alpha one. Alpha two."),
        ("Part B", "Beta one."),
    ]
    assert [c.id for c in chunks] == ["d#1", "d#2", "d#3"]


def test_long_section_is_split_with_overlap():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(12)]
    doc = "## Long\n\n" + " ".join(sentences)
    chunks = chunk_document("d", doc, max_chars=120, overlap=1)
    assert len(chunks) > 2
    assert all(len(c.text) <= 120 for c in chunks)
    for a, b in itertools.pairwise(chunks):
        last_sentence = re.findall(r"[^.]+\.", a.text)[-1].strip()
        assert b.text.startswith(last_sentence), "one sentence of overlap"
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences), "nothing is lost"


def test_list_items_are_separate_segments():
    doc = "## Steps\n\n- First do this.\n- Then do that.\n"
    (chunk,) = chunk_document("d", doc)
    assert chunk.text == "- First do this.\n- Then do that."


def test_loader_skips_readme_and_is_ordered():
    assert all(c.doc_id != "README" for c in CORPUS)
    assert [c.doc_id for c in CORPUS] == sorted(c.doc_id for c in CORPUS)


# ---- BM25 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "doc_id"),
    [
        ("Can I return opened electronics?", "returns"),
        ("how long is the warranty on a keyboard", "warranty"),
        ("what happens if an issue would take stock below zero", "stock-and-reservations"),
        ("minimum order for free standard delivery", "shipping"),
        ("how much is one loyalty point worth", "payments-and-loyalty"),
    ],
)
def test_bm25_ranks_the_relevant_document_first(query, doc_id):
    assert INDEX.search(query, k=1)[0].chunk.doc_id == doc_id


def test_bm25_score_matches_hand_computation():
    a = Chunk("a#1", "a", "", "cable cable mouse", 0, 17)
    b = Chunk("b#1", "b", "", "keyboard mouse", 0, 14)
    idx = BM25Index([a, b], k1=1.5, b=0.75)
    # N=2, df(cable)=1 -> idf = ln((2-1+0.5)/(1+0.5)+1) = ln 2; |a|=3, avgdl=2.5, tf=2
    idf = math.log(2.0)
    norm = 1 - 0.75 + 0.75 * 3 / 2.5
    expected = idf * 2 * 2.5 / (2 + 1.5 * norm)
    (hit,) = idx.search("cable", k=5)
    assert hit.chunk.id == "a#1" and hit.score == pytest.approx(expected, abs=1e-4)


def test_bm25_returns_nothing_for_unknown_terms_and_respects_k():
    assert INDEX.search("zzzz qqqq") == []
    assert len(INDEX.search("return refund delivery warranty points", k=3)) == 3
    assert INDEX.search("return", k=0) == []


def test_tokenizer_drops_stopwords_and_stems():
    assert tokenize("The returns of items") == ["return", "item"]


def test_retrievers_share_the_protocol():
    vocab = sorted({t for c in CORPUS for t in tokenize(c.text)})

    def bag_of_words(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            toks = tokenize(t)
            out.append([float(toks.count(v)) for v in vocab])
        return out

    emb = EmbeddingRetriever(CORPUS, bag_of_words)
    assert isinstance(INDEX, Retriever) and isinstance(emb, Retriever)
    assert emb.search("warranty keyboard mouse months", k=1)[0].chunk.doc_id == "warranty"


# ---- verifier ---------------------------------------------------------------------------------

HITS = INDEX.search("Can I return opened electronics?", k=3)
TOP = HITS[0].chunk


def raw(*cites: tuple[str, str]) -> RawAnswer:
    return RawAnswer(answer="x", citations=[Citation(chunk_id=c, quote=q) for c, q in cites])


def test_verbatim_quote_is_kept():
    quote = TOP.text.split(". ")[0]
    v = verify_citations(raw((TOP.id, quote)), HITS, CHUNKS)
    assert v.kept and v.dropped_count == 0 and v.grounded


def test_fabricated_quote_is_dropped_and_counted():
    v = verify_citations(
        raw((TOP.id, "Opened electronics can be returned within 60 days."), (TOP.id, TOP.text[:40])),
        HITS,
        CHUNKS,
    )
    assert len(v.kept) == 1
    assert v.dropped_count == 1 and v.dropped[0].reason == "quote_not_found"


@pytest.mark.parametrize(
    "mutate",
    [
        str.upper,
        lambda s: s.replace(" ", "  ", 1),
        lambda s: s.rstrip(".") + "!",
        lambda s: s.replace("14", "fourteen"),
    ],
    ids=["case", "spacing", "punctuation", "number-words"],
)
def test_near_miss_quotes_are_dropped(mutate):
    sentence = next(s for s in re.findall(r"[^.]+\.", TOP.text) if "14" in s).strip()
    changed = mutate(sentence)
    assert changed != sentence, "the mutation must actually change the quote"
    v = verify_citations(raw((TOP.id, changed)), HITS, CHUNKS)
    assert v.dropped_count == 1


def test_quote_from_unretrieved_chunk_is_dropped():
    other = CHUNKS["shipping#2"]
    assert other.id not in {h.chunk.id for h in HITS}
    v = verify_citations(raw((other.id, other.text[:50])), HITS, CHUNKS)
    assert v.dropped[0].reason == "not_retrieved"


@pytest.mark.parametrize(
    ("chunk_id", "quote", "reason"),
    [
        ("returns#99", "Most items can be returned", "unknown_chunk"),
        (TOP.id, "   ", "empty"),
        (TOP.id, "days", "too_short"),
    ],
)
def test_other_drop_reasons(chunk_id, quote, reason):
    assert verify_citations(raw((chunk_id, quote)), HITS, CHUNKS).dropped[0].reason == reason


# ---- end to end with a scripted model -----------------------------------------------------------


def test_answer_with_citations_keeps_real_quotes_and_drops_fake_ones():
    def reply(req):
        excerpts = req.messages[-1].content
        first_id = re.search(r"\[([^\]]+)\]", excerpts).group(1)
        real = CHUNKS[first_id].text[:45]
        return say(
            json.dumps(
                {
                    "answer": "Within 14 days.",
                    "citations": [
                        {"chunk_id": first_id, "quote": real},
                        {"chunk_id": first_id, "quote": "Opened electronics are returnable for 90 days."},
                    ],
                }
            )
        )

    llm = ScriptedLLM(reply)
    out = answer_with_citations(llm, INDEX, "Can I return opened electronics?", k=3, corpus=CHUNKS)
    assert out.answer == "Within 14 days."
    assert len(out.citations) == 1 and out.dropped_count == 1 and out.grounded
    assert out.retrieved == [h.chunk.id for h in HITS]
    assert llm.requests[0].messages[0].role == "system"


def test_unparseable_answer_is_not_grounded():
    out = answer_with_citations(
        ScriptedLLM(say("Sure! 14 days.")), INDEX, "return electronics", corpus=CHUNKS
    )
    assert out.parse_error and not out.grounded and out.answer == "Sure! 14 days."


def test_fenced_json_is_parsed():
    parsed = parse_answer('```json\n{"answer": "a", "citations": []}\n```')
    assert parsed.answer == "a" and parsed.parse_error is None


def test_corpus_is_marked_fictional():
    for path in Path(default_corpus_dir()).glob("*.md"):
        assert "fictional" in path.read_text(encoding="utf-8").lower()
