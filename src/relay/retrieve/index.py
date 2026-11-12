from __future__ import annotations
import json, re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from relay.types import Chunk, Hit

TOK = re.compile(r"[a-z0-9_]+(?:[.\-][a-z0-9_]+)*")
RRF_K = 60
EMB_MODEL = "BAAI/bge-small-en-v1.5"


def tokenize(s: str) -> list[str]:
    return TOK.findall(s.lower())


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(EMB_MODEL)


def embed(texts: list[str], query: bool = False) -> np.ndarray:
    m = embedder()
    fn = m.query_embed if (query and hasattr(m, "query_embed")) else m.embed
    v = np.array(list(fn(texts)), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def doc_text(c: Chunk) -> str:
    return c.header + "\n" + c.text


class Index:
    def __init__(self, chunks: list[Chunk], emb: np.ndarray | None):
        self.chunks = chunks
        self.emb = emb
        self.bm25 = BM25Okapi([tokenize(doc_text(c)) for c in chunks])
        self.by_id = {c.id: c for c in chunks}
        self.paper = np.array([c.paper for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], dense: bool = True) -> "Index":
        return cls(chunks, embed([doc_text(c) for c in chunks]) if dense else None)

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks.jsonl").write_text("\n".join(json.dumps(asdict(c)) for c in self.chunks))
        if self.emb is not None:
            np.save(d / "emb.npy", self.emb)

    @classmethod
    def load(cls, d: Path) -> "Index":
        chunks = [Chunk(**json.loads(l)) for l in (d / "chunks.jsonl").read_text().splitlines()]
        e = d / "emb.npy"
        return cls(chunks, np.load(e) if e.exists() else None)

    def search(self, query: str, papers=None, n: int = 20, qvec: np.ndarray | None = None,
               bm25: bool = True, dense: bool = True) -> list[Hit]:
        mask = np.ones(len(self.chunks), bool) if not papers else np.isin(self.paper, list(papers))
        idx = np.flatnonzero(mask)
        fused: dict[int, float] = {}
        if bm25:
            s = self.bm25.get_scores(tokenize(query))[idx]
            for r, i in enumerate(idx[np.argsort(-s)[:n]]):
                fused[i] = fused.get(i, 0) + 1 / (RRF_K + r + 1)
        if dense and self.emb is not None:
            q = qvec if qvec is not None else embed([query], query=True)[0]
            s = self.emb[idx] @ q
            for r, i in enumerate(idx[np.argsort(-s)[:n]]):
                fused[i] = fused.get(i, 0) + 1 / (RRF_K + r + 1)
        order = sorted(fused, key=lambda i: (-fused[i], i))[:n]
        return [Hit(self.chunks[i], fused[i]) for i in order]


SIZES = (120, 200, 400)


def cli(argv):
    from relay.ingest.parse_html import load_blocks
    from relay.ingest.chunk import chunk_blocks
    blocks = load_blocks()
    for size in SIZES:
        ch = chunk_blocks(blocks, size)
        Index.build(ch).save(Path(f"data/corpus/c{size}"))
        print(f"c{size}: {len(ch)} chunks, {sum(c.n_tokens for c in ch)} tokens, max {max(c.n_tokens for c in ch)}")
