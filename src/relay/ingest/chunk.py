from __future__ import annotations
import re
from functools import lru_cache
from relay.types import Chunk, LETTER


@lru_cache(maxsize=1)
def _enc():
    import tiktoken
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_enc().encode(text, disallowed_special=()))


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$(\[])")


def _split_long(text: str, limit: int) -> list[str]:
    out, cur = [], ""
    for s in _SENT.split(text):
        if cur and count_tokens(cur + " " + s) > limit:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return out


def _split_table(text: str, limit: int) -> list[str]:
    lines = text.split("\n")
    head = lines[:2]  # caption + header row
    out, cur = [], list(head)
    for l in lines[2:]:
        if len(cur) > len(head) and count_tokens("\n".join(cur + [l])) > limit:
            out.append("\n".join(cur)); cur = list(head)
        cur.append(l)
    out.append("\n".join(cur))
    return out


def chunk_blocks(blocks, chunk_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}

    def emit(b, text, label=None, kind=None):
        p = b.paper
        counters[p] = counters.get(p, 0) + 1
        chunks.append(Chunk(f"{LETTER[p]}{counters[p]}", p, b.section_num, b.section_title, kind or b.kind,
                            label if label is not None else b.label, b.page, text, count_tokens(text)))

    buf, buf_b = "", None

    def flush():
        nonlocal buf, buf_b
        if buf:
            emit(buf_b, buf, kind="text" if buf_b.kind != "abstract" else "abstract")
        buf, buf_b = "", None

    for b in blocks:
        if b.kind in ("table", "figure_caption"):
            flush()
            parts = _split_table(b.text, 2 * chunk_tokens) if b.kind == "table" and count_tokens(b.text) > 2 * chunk_tokens else [b.text]
            for t in parts:
                emit(b, t)
            continue
        same = buf_b is not None and (buf_b.paper, buf_b.section_num, buf_b.section_title) == (b.paper, b.section_num, b.section_title)
        if not same:
            flush()
        for piece in ([b.text] if count_tokens(b.text) <= chunk_tokens else _split_long(b.text, chunk_tokens)):
            if buf and count_tokens(buf + "\n" + piece) > chunk_tokens:
                cur_b = buf_b; flush(); buf_b = cur_b
            if not buf:
                buf_b = b
            buf = (buf + "\n" + piece).strip()
    flush()
    return chunks
