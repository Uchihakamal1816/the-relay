import asyncio, time
import numpy as np
from relay.ingest.parse_html import Block
from relay.ingest.chunk import chunk_blocks, count_tokens
from relay.retrieve.index import Index
from relay.retrieve.tool import PaperSearch
from relay.types import RetrievalSetting


def B(p, sec, kind, text, label=None, o=0):
    return Block(p, sec, f"T{sec}", kind, label, text, o)


BLOCKS = [B("attention", "1", "text", "The encoder has six layers. " * 3), B("attention", "1", "text", "Dropout is 0.1."),
          B("attention", "2", "text", "Training used eight GPUs."),
          B("attention", "2", "table", "Table 1: BLEU\nModel | BLEU\nBig | 28.4", "Table 1"),
          B("moshi", "3", "text", "Mimi codec runs at 12.5 Hz frame rate."), B("hnet", "2", "text", "Dynamic chunking routes bytes.")]


class FakeRR:
    def __init__(self, sleep=0): self.sleep = sleep
    def score(self, q, texts):
        time.sleep(self.sleep); return [len(t) for t in texts]


def ps(rr=None):
    ch = chunk_blocks(BLOCKS, 200)
    return PaperSearch(indexes={200: Index(ch, None)}, reranker=rr or FakeRR()), ch


def test_no_cross_section_and_table_atomic():
    ch = chunk_blocks(BLOCKS, 200)
    assert len({(c.paper, c.section_num) for c in ch if c.kind == "text"}) == len([c for c in ch if c.kind == "text"])
    t = [c for c in ch if c.kind == "table"]
    assert len(t) == 1 and "28.4" in t[0].text and t[0].label == "Table 1"


def test_sizes_respected():
    long = [B("moshi", "4", "text", "This is a sentence about speech. " * 200)]
    ch = chunk_blocks(long, 120)
    assert len(ch) > 1 and all(c.n_tokens <= 140 for c in ch)


def test_paper_filter_and_deterministic():
    s, _ = ps()
    st = RetrievalSetting("X", 200, 3, False)
    r1 = asyncio.run(s.search_papers(["frame rate"], ["moshi"], st, question="frame rate"))
    assert all(h.chunk.paper == "moshi" for h in r1.hits)
    a = asyncio.run(s.search_papers(["layers"], None, st, question="layers"))
    b = asyncio.run(s.search_papers(["layers"], None, st, question="layers"))
    assert [h.chunk.id for h in a.hits] == [h.chunk.id for h in b.hits]


def test_rerank_timeout():
    s, _ = ps(FakeRR(sleep=1.0))
    st = RetrievalSetting("X", 200, 3, True)
    r = asyncio.run(s.search_papers(["layers"], None, st, question="layers", timeout_s=0.05))
    assert r.timed_out is True


def test_per_paper_quota():
    s, _ = ps()
    st = RetrievalSetting("X", 200, 3, False)
    r = asyncio.run(s.search_papers(["the"], ["attention", "moshi", "hnet"], st, question="the", per_paper=True))
    assert {h.chunk.paper for h in r.hits} == {"attention", "moshi", "hnet"}
