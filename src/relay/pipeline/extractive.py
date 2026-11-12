from __future__ import annotations
from . import messages as M


def extractive_reply(question: str, hits, reason: str, resume_in_s: float | None = None) -> str:
    if reason == "day":
        s = int(resume_in_s or 0)
        head = M.EXTRACTIVE_DAY.format(h=s // 3600, m=(s % 3600) // 60)
    elif reason == "unverified":
        head = M.EXTRACTIVE_UNVERIFIED
    else:
        head = M.EXTRACTIVE_MINUTE
    quotes = []
    for h in hits[:2]:
        t = h.chunk.text.replace("\n", " ")
        quotes.append(f'"{t[:320]}{"…" if len(t) > 320 else ""}" {h.chunk.cite}')
    return head + "\n" + "\n".join(quotes)
