from __future__ import annotations
import re
from dataclasses import dataclass

CITE = re.compile(r"\[\s*([A-Z]\d+(?:\s*,\s*[A-Z]\d+)*)\s*\]\s*[.]?\s*$")
NUM = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*")
LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


@dataclass
class LineVerdict:
    line: str
    ok: bool
    reason: str
    ids: list[str]
    body: str = ""


def _norm_num(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", s).rstrip(".,")


def _nums_in(text: str) -> set[str]:
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return {m.rstrip(".,") for m in re.findall(r"\d+(?:\.\d+)*", t)}


class LineGate:
    def __init__(self, hits, question: str):
        self.by_id = {h.chunk.id: h.chunk for h in hits}
        self.qnums = _nums_in(question)
        self.buf = ""
        self.n = 0

    def _judge(self, line: str) -> LineVerdict:
        self.n += 1
        if self.n == 1 and line.strip().strip("`*.").strip() == "NOT_IN_PAPERS":
            return LineVerdict(line, True, "sentinel", [])
        m = CITE.search(line)
        if not m:
            return LineVerdict(line, False, "no_citation", [])
        ids = [x.strip() for x in m.group(1).split(",")]
        if any(i not in self.by_id for i in ids):
            return LineVerdict(line, False, "unknown_id", ids)
        body = LIST.sub("", line[:m.start()]).strip()
        src = set(self.qnums)
        for i in ids:
            c = self.by_id[i]
            src |= _nums_in(c.header + " " + c.text)
        for n in NUM.findall(body):
            if _norm_num(n) not in src:
                return LineVerdict(line, False, "number_not_in_source", ids, body)
        return LineVerdict(line, True, "ok", ids, body)

    def feed(self, text: str) -> list[LineVerdict]:
        self.buf += text
        out = []
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                out.append(self._judge(line))
        return out

    def flush(self) -> list[LineVerdict]:
        line, self.buf = self.buf, ""
        return [self._judge(line)] if line.strip() else []
