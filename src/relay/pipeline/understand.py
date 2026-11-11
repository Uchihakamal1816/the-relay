from __future__ import annotations
import json, re
from dataclasses import dataclass, field

ALIASES = {
    "moshi": r"moshi|mimi|helium|inner monologue|kyutai|speech-text|full-duplex|spoken dialogue",
    "hnet": r"h-net|hnet|dynamic chunking|byte-level|hierarchical sequence|routing module|smoothing module",
    "attention": r"transformer paper|attention is all you need|vaswani|\bbleu\b|\bwmt\b|english-to-german|english-to-french|constituency parsing",
}


@dataclass
class Understanding:
    in_scope: bool = True
    papers: list = field(default_factory=lambda: ["attention", "moshi", "hnet"])
    ambiguous: bool = True
    queries: list = field(default_factory=list)
    source: str = "heuristic"


def heuristic_understand(question: str) -> Understanding:
    q = question.lower()
    hit = [p for p, rx in ALIASES.items() if re.search(rx, q)]
    return Understanding(True, hit or ["attention", "moshi", "hnet"], not hit, [question], "heuristic")


PROMPT = """Classify a question about three papers: "attention" (Attention Is All You Need, the Transformer), "moshi" (Moshi speech-text dialogue model, Mimi codec, Helium), "hnet" (H-Net, dynamic chunking, byte-level hierarchical modeling).
Return JSON only: {"in_scope": bool, "papers": [subset of "attention","moshi","hnet"], "ambiguous": bool, "queries": [up to 3 short search queries]}.
in_scope=false only if the question is clearly unrelated to these papers. ambiguous=true if the question names no paper and could apply to several.
QUESTION: """


def parse_understanding(text: str) -> Understanding:
    d = json.loads(text[text.index("{"): text.rindex("}") + 1])
    papers = [p for p in d.get("papers") or [] if p in ALIASES] or ["attention", "moshi", "hnet"]
    qs = [q for q in (d.get("queries") or []) if isinstance(q, str) and q.strip()][:3]
    return Understanding(bool(d.get("in_scope", True)), papers, bool(d.get("ambiguous", len(papers) > 1)), qs, "llm")
