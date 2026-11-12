from pathlib import Path
from relay.ingest.parse_html import parse_paper, Block
from relay.ingest.chunk import chunk_blocks
from relay.retrieve.papermap import PaperMap, normalise_symbol

B, _ = parse_paper(Path("tests/fixtures/mini_paper.html").read_text(), "attention", {"attention_Table 2": "x | y"})
PM = PaperMap.build(B)


def test_table_entry():
    r = PM.lookup("What is in Table 1?")
    assert [(e.kind, e.key) for e in r] == [("table", "table 1")]


def test_symbol_definition():
    assert normalise_symbol(r"$d_{\text{model}}$") == "d_model"
    r = PM.lookup("What does d_model mean?")
    assert r and r[0].kind == "symbol" and r[0].section_num == "1"


def test_term_definition():
    r = PM.lookup("what is the relay layer used for")
    assert any(e.kind == "term" and e.key == "relay layer" for e in r)


def test_section_and_appendix():
    assert PM.lookup("summarise section 1.1")[0].section_num == "1.1"
    assert PM.lookup("What is in Appendix A?")[0].section_num == "A"


def test_no_false_hits():
    assert PM.lookup("How fast is training?") == []


def test_ambiguous_table_number():
    bl = [Block(p, "3", "R", "table", "Table 3", f"Table 3: {p} results\nx | y", 0) for p in ("attention", "moshi", "hnet")]
    r = PaperMap.build(bl).lookup("What does Table 3 show?")
    assert sorted(e.paper for e in r) == ["attention", "hnet", "moshi"]


def test_resolve_finds_chunk():
    for size in (120, 400):
        ch = {c.id: c for c in chunk_blocks(B, size)}
        h = PM.resolve(PM.lookup("Table 1 values?") + PM.lookup("d_model meaning"), ch, 2)
        assert len(h) == 2 and h[0].chunk.label == "Table 1" and h[0].from_map and "d_{model}" in h[1].chunk.text
