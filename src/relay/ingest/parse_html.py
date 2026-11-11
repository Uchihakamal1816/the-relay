from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag

WS = re.compile(r"\s+")


@dataclass
class Block:
    paper: str
    section_num: str
    section_title: str
    kind: str  # abstract | text | table | figure_caption
    label: str | None
    text: str
    order: int
    page: int | None = None


def _clean(s: str) -> str:
    return WS.sub(" ", s).strip()


def _cls(t: Tag) -> list[str]:
    return t.get("class") or []


def _prep(soup: BeautifulSoup) -> None:
    for m in soup.find_all("math"):
        m.replace_with(NavigableString(f" ${m.get('alttext', '').strip()}$ "))
    for n in soup.select(".ltx_note"):
        c = n.select_one(".ltx_note_content")
        if c is None:
            n.decompose(); continue
        for x in c.select(".ltx_note_mark, .ltx_tag, .ltx_note_type"):
            x.decompose()
        n.replace_with(NavigableString(f" (footnote: {_clean(c.get_text(' '))})"))
    for x in soup.select("section.ltx_bibliography, nav, footer, div.ltx_authors, .ltx_page_logo, div.ltx_pagination"):
        x.decompose()


def _heading(sec: Tag) -> tuple[str, str]:
    h = sec.find(re.compile(r"^h[1-6]$"), recursive=False)
    if h is None:
        return "", ""
    tag = h.select_one("span.ltx_tag")
    num = _clean(tag.get_text()) if tag else ""
    if tag:
        tag.extract()
    title = _clean(h.get_text(" "))
    num = re.sub(r"^Appendix\s+", "", num).rstrip(".")
    return num, title


def _label(fig: Tag) -> str | None:
    cap = fig.find("figcaption")
    tag = cap.select_one("span.ltx_tag") if cap else None
    if not tag:
        return None
    return _clean(tag.get_text()).rstrip(":.").strip() or None


def _caption(fig: Tag) -> str:
    cap = fig.find("figcaption")
    return _clean(cap.get_text(" ")) if cap else ""


class _Walker:
    def __init__(self, paper: str, patches: dict[str, str]):
        self.paper, self.patches = paper, patches
        self.blocks: list[Block] = []
        self.gaps: list[str] = []

    def add(self, num, title, kind, text, label=None):
        text = _clean(text) if kind != "table" else text.strip()
        if text:
            self.blocks.append(Block(self.paper, num, title, kind, label, text, len(self.blocks)))

    def append_prev(self, num, title, text):
        text = _clean(text)
        if not text:
            return
        if self.blocks and self.blocks[-1].kind == "text" and self.blocks[-1].section_num == num and self.blocks[-1].section_title == title:
            self.blocks[-1].text += " " + text
        else:
            self.add(num, title, "text", text)

    def table(self, fig: Tag, num, title):
        label = _label(fig) or "Table"
        cap = _caption(fig)
        rows = []
        for tr in fig.find_all("tr"):
            cells = [_clean(c.get_text(" ")) for c in tr.find_all(["td", "th"])]
            if any(cells):
                rows.append(" | ".join(cells))
        if not rows:
            key = f"{self.paper}_{label}"
            if key in self.patches:
                rows = [self.patches[key]]
            else:
                rows = ["[TABLE BODY MISSING]"]
                self.gaps.append(label)
        self.add(num, title, "table", cap + "\n" + "\n".join(rows), label)

    def figure(self, fig: Tag, num, title):
        for t in fig.select("figure.ltx_table"):
            self.table(t, num, title)
        cap = _caption(fig)
        if cap:
            self.add(num, title, "figure_caption", cap, _label(fig))

    def node(self, c: Tag, num, title):
        cls = _cls(c)
        if c.name == "section":
            self.section(c); return
        if c.name == "figure":
            (self.table if "ltx_table" in cls else self.figure)(c, num, title); return
        if c.name == "table" and any(x.startswith("ltx_equation") for x in cls):
            self.append_prev(num, title, "$$" + _clean(c.get_text(" ")).replace("$", "") + "$$"); return
        if c.name in ("ul", "ol"):
            items = [_clean(li.get_text(" ")) for li in c.find_all("li", recursive=False)]
            self.add(num, title, "text", "\n".join(f"- {i}" for i in items)); return
        if c.name == "p":
            self.add(num, title, "text", c.get_text(" ")); return
        if c.name and re.match(r"^h[1-6]$", c.name):
            return
        kids = [k for k in c.children if isinstance(k, Tag)]
        if kids and (c.name == "div" or c.name == "span"):
            for k in kids:
                self.node(k, num, title)
        else:
            self.add(num, title, "text", c.get_text(" "))

    def section(self, sec: Tag):
        num, title = _heading(sec)
        self.last = (num, title)
        for c in sec.children:
            if isinstance(c, Tag):
                self.node(c, num, title)
        self.last = (num, title)


def parse_paper(html: str, paper: str, patches: dict[str, str] | None = None) -> tuple[list[Block], list[str]]:
    soup = BeautifulSoup(html, "lxml")
    _prep(soup)
    w = _Walker(paper, patches or {})
    ab = soup.select_one("div.ltx_abstract")
    if ab:
        for h in ab.find_all("h6"):
            h.decompose()
        w.add("0", "Abstract", "abstract", ab.get_text(" "))
    art = soup.select_one("article") or soup.body
    w.last = None
    for c in art.find_all(recursive=False):
        if c.name == "section":
            w.section(c)
        elif c.name == "figure" and w.last:  # floats LaTeXML left outside their section
            w.node(c, *w.last)
    return w.blocks, w.gaps


def load_patches(d: Path = Path("data/patches")) -> tuple[dict[str, str], list[str]]:
    out, unverified = {}, []
    for f in sorted(d.glob("*_table*.md")):
        paper, n = re.match(r"(\w+?)_table(\d+)\.md", f.name).groups()
        lines = f.read_text().splitlines()
        if not lines[0].strip().lower().startswith("verified: true"):
            unverified.append(f.name)
        out[f"{paper}_Table {n}"] = "\n".join(l for l in lines[1:] if l.strip())
    return out, unverified


def cli(argv):
    from .audit_pdf import pdf_pages, coverage_by_page, map_pages
    from .fetch import SOURCES
    patches, unverified = load_patches()
    out = Path("data/corpus"); out.mkdir(parents=True, exist_ok=True)
    allb, audit = [], {}
    for p in SOURCES:
        blocks, gaps = parse_paper(Path(f"data/papers/{p}.html").read_text(), p, patches)
        pages = pdf_pages(Path(f"data/papers/{p}.pdf"))
        map_pages(blocks, pages)
        cov = coverage_by_page(pages, " ".join(b.text for b in blocks))
        tables = sorted({b.label for b in blocks if b.kind == "table"}, key=lambda s: int(re.sub(r"\D", "", s) or 0))
        low = [(i + 1, round(c, 2)) for i, c in enumerate(cov) if c < 0.5]
        audit[p] = {"blocks": len(blocks), "tables": tables, "gaps": gaps, "low_coverage_pages": low,
                    "unmapped_blocks": sum(b.page is None for b in blocks)}
        print(f"{p}: {len(blocks)} blocks, {len(tables)} tables, gaps={gaps}, unmapped={audit[p]['unmapped_blocks']}, low-coverage pages={low}")
        allb += blocks
    (out / "blocks.jsonl").write_text("\n".join(json.dumps(asdict(b)) for b in allb))
    (out / "audit.json").write_text(json.dumps(audit, indent=1))
    try:
        from relay.retrieve.papermap import PaperMap
        pm = PaperMap.build(allb); pm.save(out / "papermap.json"); pm.print_counts()
    except ImportError:
        pass
    if unverified:
        print("WARNING unverified patches:", unverified)


def load_blocks(path: Path = Path("data/corpus/blocks.jsonl")) -> list[Block]:
    return [Block(**json.loads(l)) for l in path.read_text().splitlines() if l.strip()]
