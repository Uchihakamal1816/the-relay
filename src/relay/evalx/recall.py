from __future__ import annotations
import asyncio
import numpy as np
import pandas as pd
from relay.types import RetrievalSetting
from relay.pipeline.understand import heuristic_understand


def _hit(chunks, q) -> bool:
    return any(c.paper == g.paper and ((g.section and c.section_num.startswith(g.section)) or (g.label and c.label == g.label))
               for c in chunks for g in q.gold)


def recall_table(qs, ps, papermap=None, sizes=(120, 200, 400), ks=(2, 4, 6, 8), maps=(False, True)) -> pd.DataFrame:
    rows = []
    ans = [q for q in qs if q.answerable]
    for size in sizes:
        for k in ks:
            for rr in (False, True):
                for use_map in maps:
                    s = RetrievalSetting(f"c{size}k{k}{'rr' if rr else ''}{'+map' if use_map else ''}", size, k, rr)
                    hit, toks = 0, []
                    for q in ans:
                        u = heuristic_understand(q.question)
                        r = asyncio.run(ps.search_papers([q.question], u.papers, s, question=q.question, timeout_s=30, per_paper=u.ambiguous))
                        hits = r.hits
                        if use_map and papermap:
                            mh = papermap.resolve(papermap.lookup(q.question, u.papers), ps.chunks_by_id(size), 2)
                            ids = {h.chunk.id for h in mh}
                            hits = (mh + [h for h in hits if h.chunk.id not in ids])[:max(len(hits), len(mh))]
                        hit += _hit([h.chunk for h in hits], q)
                        toks.append(sum(h.chunk.n_tokens for h in hits))
                    rows.append({"setting": s.name, "size": size, "k": k, "rerank": rr, "map": use_map,
                                 "recall": round(hit / max(1, len(ans)), 3), "passage_tokens": int(np.mean(toks)) if toks else 0})
    return pd.DataFrame(rows)


def naive_baseline(qs, index_400) -> list[dict]:
    """N0: 400-token pieces, dense only, k=4, no routing. Returns per-question top hits for the 'mislead' section."""
    out = []
    for q in qs:
        if not q.answerable:
            continue
        hits = index_400.search(q.question, None, 4, bm25=False)
        out.append({"qid": q.id, "hit": _hit([h.chunk for h in hits], q), "top": [(h.chunk.id, h.chunk.header) for h in hits]})
    return out


def score_split(qs, ps, setting) -> dict:
    """Top fused score for answerable vs unanswerable questions (to decide retrieval.refuse_below)."""
    res = {"answerable": [], "unanswerable": []}
    for q in qs:
        r = asyncio.run(ps.search_papers([q.question], None, setting, question=q.question, timeout_s=30))
        top = r.hits[0].rerank_score if r.hits and r.hits[0].rerank_score is not None else (r.hits[0].score if r.hits else 0)
        res["answerable" if q.answerable else "unanswerable"].append(round(float(top), 4))
    return res
