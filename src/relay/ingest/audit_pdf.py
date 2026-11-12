from __future__ import annotations
import re
from collections import Counter
from pathlib import Path


def pdf_pages(path: Path) -> list[str]:
    import pymupdf
    with pymupdf.open(path) as doc:
        return [pg.get_text("text", sort=True) for pg in doc]


def _toks(s: str) -> list[str]:
    s = re.sub(r"-\s*\n\s*", "", s.lower())
    return re.findall(r"[a-z0-9]+", s)


def _grams(t: list[str], n: int) -> list[tuple]:
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


def coverage_by_page(pages: list[str], corpus_text: str, n: int = 5) -> list[float]:
    have = set(_grams(_toks(corpus_text), n))
    out = []
    for p in pages:
        g = _grams(_toks(p), n)
        out.append(sum(x in have for x in g) / len(g) if g else 1.0)
    return out


def map_pages(blocks, pages: list[str], n: int = 5) -> None:
    page_grams = [set(_grams(_toks(p), n)) for p in pages]
    for b in blocks:
        clean = re.sub(r"\$[^$]*\$", " ", b.text)
        g = _grams(_toks(clean), n)
        if not g:
            continue
        picks = [g[0], g[len(g) // 2], g[-1]]
        votes = Counter(i for x in picks for i, s in enumerate(page_grams) if x in s)
        if votes:
            b.page = votes.most_common(1)[0][0] + 1
