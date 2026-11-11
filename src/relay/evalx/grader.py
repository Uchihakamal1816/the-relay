from __future__ import annotations
import re
from relay.types import Level


def _norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", s.lower())


def grade(q, reply, chunks_by_id: dict) -> str | None:
    if reply is None or reply.level == Level.EXTRACTIVE or not reply.text:
        return None
    if not q.answerable:
        return "refused_correctly" if reply.refused else "wrong"
    if reply.refused:
        return "refused_wrongly"
    t = _norm(reply.text)
    if not all(any(_norm(str(a)) in t for a in grp) for grp in q.must_include):
        return "wrong"
    for cid in reply.cited_ids:
        c = chunks_by_id.get(cid)
        if c and any(c.paper == g.paper and ((g.section and c.section_num.startswith(g.section)) or (g.label and c.label == g.label))
                     for g in q.gold):
            return "correct"
    return "wrong"
