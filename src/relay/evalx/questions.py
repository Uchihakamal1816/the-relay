from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Gold:
    paper: str
    section: str = ""
    label: str | None = None


@dataclass
class Question:
    id: str
    question: str
    answerable: bool
    category: str = ""
    gold: list = field(default_factory=list)
    must_include: list = field(default_factory=list)
    note: str = ""


def load_questions(path: str | Path = "eval/questions.yaml") -> list[Question]:
    raw = yaml.safe_load(Path(path).read_text()) or []
    return [Question(**{**d, "gold": [Gold(**g) for g in d.get("gold") or []]}) for d in raw]


def _norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", s.lower())


def validate_questions(qs, blocks) -> list[str]:
    probs, ids = [], set()
    for q in qs:
        if q.id in ids:
            probs.append(f"{q.id}: duplicate id")
        ids.add(q.id)
        if not q.answerable:
            if q.gold or q.must_include:
                probs.append(f"{q.id}: unanswerable question has gold or must_include")
            continue
        if not q.gold:
            probs.append(f"{q.id}: answerable but no gold")
        text = " ".join(_norm(b.text) for b in blocks for g in q.gold
                        if b.paper == g.paper and (b.section_num.startswith(g.section) if g.section else True) and (g.label is None or b.label == g.label))
        if not text:
            probs.append(f"{q.id}: gold section not found")
        for grp in q.must_include:
            if not any(_norm(str(a)) in text for a in grp):
                probs.append(f"{q.id}: none of {grp} appears in the gold section text")
    return probs
