from __future__ import annotations
import asyncio, time
from dataclasses import dataclass
from pathlib import Path
from relay.types import Hit, RetrievalSetting
from .index import Index, doc_text, embed, SIZES
from .rerank import Reranker


@dataclass
class ToolResult:
    hits: list[Hit]
    timed_out: bool
    ms: float
    rerank_skipped: bool = False


class PaperSearch:
    def __init__(self, corpus_dir: Path = Path("data/corpus"), reranker: Reranker | None = None,
                 indexes: dict[int, Index] | None = None, max_rerank: int = 4):
        self.indexes = indexes or {s: Index.load(corpus_dir / f"c{s}") for s in SIZES if (corpus_dir / f"c{s}").exists()}
        self.reranker = reranker or Reranker()
        # Re-ranking is CPU work the provider limits never see. A timed-out re-rank keeps running in its
        # thread, so under overload they pile up. Cap concurrent re-ranks; beyond the cap, skip re-ranking.
        self.max_rerank, self.rerank_inflight = max_rerank, 0

    def warmup(self) -> None:
        """Load ONNX models once; a cold re-ranker takes seconds and would time out the first requests."""
        embed(["warmup"], query=True)
        self.reranker.score("warmup", ["warmup"])

    def chunks_by_id(self, chunk_tokens: int):
        return self.indexes[chunk_tokens].by_id

    def _fuse(self, queries, papers, setting) -> list[Hit]:
        ix = self.indexes[setting.chunk_tokens]
        best: dict[str, Hit] = {}
        for q in queries:
            for h in ix.search(q, papers, setting.candidates):
                if h.chunk.id not in best or h.score > best[h.chunk.id].score:
                    best[h.chunk.id] = h
        return sorted(best.values(), key=lambda h: (-h.score, h.chunk.id))[:setting.candidates]

    async def search_papers(self, queries, papers, setting: RetrievalSetting, *, question: str,
                            timeout_s: float = 1.5, per_paper: bool = False, k: int | None = None) -> ToolResult:
        t0 = time.perf_counter()
        k = k or setting.k
        cands = await asyncio.to_thread(self._fuse, queries or [question], papers, setting)
        timed_out = skipped = False
        if setting.rerank and cands and self.rerank_inflight >= self.max_rerank:
            skipped = True
        elif setting.rerank and cands:
            self.rerank_inflight += 1
            try:
                scores = await asyncio.wait_for(asyncio.to_thread(
                    self.reranker.score, question, [doc_text(h.chunk) for h in cands]), timeout_s)
                for h, s in zip(cands, scores):
                    h.rerank_score = s
                cands.sort(key=lambda h: -h.rerank_score)
            except asyncio.TimeoutError:
                timed_out = True
            finally:
                self.rerank_inflight -= 1
        if per_paper:
            ps = list(papers) if papers else sorted({h.chunk.paper for h in cands})
            quota = max(1, k // max(1, len(ps)))
            out, used = [], {p: 0 for p in ps}
            for h in cands:
                if used.get(h.chunk.paper, quota) < quota:
                    out.append(h); used[h.chunk.paper] += 1
            for h in cands:
                if len(out) >= k:
                    break
                if h not in out:
                    out.append(h)
            hits = out[:max(k, len(ps))]
        else:
            hits = cands[:k]
        return ToolResult(hits, timed_out, (time.perf_counter() - t0) * 1000, skipped)


def cli(argv):
    import argparse
    from relay.config import load_config
    ap = argparse.ArgumentParser(); ap.add_argument("query"); ap.add_argument("--setting", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(); s = cfg.retrieval.settings[a.setting or cfg.retrieval.default]
    ps = PaperSearch(); ps.warmup()
    r = asyncio.run(ps.search_papers([a.query], None, s, question=a.query))
    for h in r.hits:
        print(f"{h.chunk.id:6} {h.score:.4f} {h.rerank_score} {h.chunk.header} {h.chunk.text[:90]!r}")
