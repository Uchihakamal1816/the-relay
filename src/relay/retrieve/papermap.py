from __future__ import annotations
import json, re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from relay.types import Hit

GREEK = "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
DEF_AFTER = re.compile(r"^\s*,?\s*(?:denotes?|is the|are the|is defined as|represents?|refers? to|is a|indicates?)\b", re.I)
DEF_BEFORE = re.compile(r"\b(?:where|let)\s*$", re.I)
GENERIC = {"related work", "experimental setup", "model architecture", "training data", "main results", "additional results", "ablation studies", "additional ablation studies", "results", "training"}
TERM = re.compile(r"\bwe (?:call|term|name|dub|refer to (?:this|it|them|these)(?: \w+){0,3} as)\s+(?:(?:this|it|them|these|the|a|an|our)\s+)*[\"“']?([A-Za-z][\w\- ]{2,40}?)[\"”']?(?=[,.;:)]|\s\(|$)", re.I)
MATH = re.compile(r"\$([^$]{1,40})\$")


@dataclass
class MapEntry:
    paper: str
    kind: str
    key: str
    display: str
    section_num: str
    snippet: str
    label: str | None = None


def normalise_symbol(s: str) -> str:
    s = s.strip().strip("$")
    s = re.sub(r"\\(?:text|mathrm|mathit|mathbf|operatorname|boldsymbol|mathcal)\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"[{}\s\\]", "", s)
    return s.lower()


class PaperMap:
    def __init__(self, entries: list[MapEntry]):
        self.entries = entries

    @classmethod
    def build(cls, blocks) -> "PaperMap":
        out: list[MapEntry] = []
        seen: set[tuple] = set()

        def add(e: MapEntry):
            k = (e.paper, e.kind, e.key)
            if k not in seen:
                seen.add(k); out.append(e)

        for b in blocks:
            if b.section_num or b.section_title:
                key = b.section_num.lower() if b.section_num else b.section_title.lower()
                if re.fullmatch(r"[a-f]", key):
                    key = f"appendix {key}"
                add(MapEntry(b.paper, "section", key, f"§{b.section_num} {b.section_title}".strip(), b.section_num, b.text[:120]))
                t = b.section_title.lower().strip().rstrip(".")
                if 2 <= len(t.split()) <= 5 and t not in GENERIC:  # section titles name the paper's own concepts
                    add(MapEntry(b.paper, "term", t, b.section_title, b.section_num, b.text[:120]))
            if b.kind in ("table", "figure_caption") and b.label:
                add(MapEntry(b.paper, "table" if b.kind == "table" else "figure", b.label.lower(), b.label, b.section_num,
                             b.text[:120], b.label))
            if b.kind not in ("text", "abstract"):
                continue
            for m in MATH.finditer(b.text):
                sym = normalise_symbol(m.group(1))
                if not sym or len(sym) > 20 or "=" in sym or re.fullmatch(r"[\d.,]+", sym):
                    continue
                if DEF_AFTER.match(b.text[m.end():m.end() + 60]) or DEF_BEFORE.search(b.text[max(0, m.start() - 20):m.start()]):
                    s0 = max(0, m.start() - 30)
                    add(MapEntry(b.paper, "symbol", sym, f"${m.group(1)}$", b.section_num, b.text[s0:s0 + 120]))
            for m in TERM.finditer(b.text):
                term = m.group(1).strip().lower()
                if len(term.split()) <= 5:
                    s0 = max(0, m.start() - 30)
                    add(MapEntry(b.paper, "term", term, term, b.section_num, b.text[s0:s0 + 120]))
        return cls(out)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps([asdict(e) for e in self.entries], indent=0))

    @classmethod
    def load(cls, path: Path) -> "PaperMap":
        return cls([MapEntry(**d) for d in json.loads(path.read_text())])

    def print_counts(self) -> None:
        c = Counter((e.paper, e.kind) for e in self.entries)
        print("papermap:", ", ".join(f"{p}/{k}={n}" for (p, k), n in sorted(c.items())))

    def lookup(self, question: str, papers=None) -> list[MapEntry]:
        q = question.lower()
        wanted: list[tuple[str, str]] = []
        for kind, n in re.findall(r"\b(table|tab\.|figure|fig\.)\s*(\d+)\b", q):
            wanted.append(("table" if kind.startswith("tab") else "figure", f"{'table' if kind.startswith('tab') else 'figure'} {n}"))
        for _, n in re.findall(r"\b(section|sec\.|§)\s*(\d+(?:\.\d+)*)\b", q):
            wanted.append(("section", n))
        for n in re.findall(r"\bappendix\s+([a-f])\b", q):
            wanted.append(("section", f"appendix {n}"))
        syms = set()
        for m in re.finditer(r"\$([^$]+)\$|`([^`]+)`", question):
            syms.add(normalise_symbol(m.group(1) or m.group(2)))
        for tok in re.findall(r"[^\s,?;:()]+", question):
            if any(ch in tok for ch in "_^\\") or any(ch in tok for ch in GREEK):
                syms.add(normalise_symbol(tok.rstrip(".")))
        res = []
        for e in self.entries:
            if papers and e.paper not in papers:
                continue
            if (e.kind, e.key) in wanted or (e.kind == "symbol" and e.key in syms) or \
               (e.kind == "term" and re.search(r"\b" + re.escape(e.key) + r"\b", q)):
                res.append(e)
        return res

    def resolve(self, entries: list[MapEntry], chunks: dict, max_hits: int) -> list[Hit]:
        hits, used = [], set()
        for e in entries:
            if len(hits) >= max_hits:
                break
            for c in chunks.values():
                if c.paper != e.paper or c.id in used:
                    continue
                if e.label and c.label == e.label and c.kind in ("table", "figure_caption") or \
                   (not e.label and c.section_num == e.section_num and e.snippet[:60] in c.text):
                    hits.append(Hit(c, 1.0, from_map=True)); used.add(c.id); break
        return hits
