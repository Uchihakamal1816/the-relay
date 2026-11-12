from pathlib import Path
from relay.ingest.parse_html import parse_paper
from relay.ingest.audit_pdf import coverage_by_page, map_pages

HTML = Path("tests/fixtures/mini_paper.html").read_text()


def test_sections():
    b, _ = parse_paper(HTML, "attention")
    nums = list(dict.fromkeys(x.section_num for x in b))
    assert nums == ["0", "1", "1.1", "A"]
    assert [x.section_title for x in b if x.section_num == "A"][0] == "Extra"


def test_math_alttext_and_footnote():
    b, _ = parse_paper(HTML, "attention")
    t = [x.text for x in b if x.section_num == "1"][0]
    assert "$d_{model}=512$" in t and "(footnote: A footnote here.)" in t


def test_table_rows_and_gap():
    b, gaps = parse_paper(HTML, "attention")
    t1 = [x for x in b if x.label == "Table 1"][0]
    assert t1.kind == "table" and "a | b | c" in t1.text
    assert gaps == ["Table 2"]
    b2, gaps2 = parse_paper(HTML, "attention", {"attention_Table 2": "x | y\n5 | 6"})
    assert gaps2 == [] and "5 | 6" in [x for x in b2 if x.label == "Table 2"][0].text


def test_bibliography_dropped():
    b, _ = parse_paper(HTML, "attention")
    assert not any("Zebra" in x.text for x in b)
    assert not any("Alice" in x.text for x in b)


def test_coverage_and_page_map():
    b, _ = parse_paper(HTML, "attention")
    pages = ["unrelated words about cooking pasta in water with salt and oil today", "Appendix text and more appendix text here"]
    tgt = [x for x in b if x.section_num == "A"]
    tgt[0].text = "Appendix text and more appendix text here"
    map_pages(tgt, pages)
    assert tgt[0].page == 2
    cov = coverage_by_page(pages, " ".join(x.text for x in b))
    assert cov[0] < 0.1


def test_paragraph_heading_inherits_number():
    b, _ = parse_paper(HTML, "attention")
    x = [x for x in b if x.text == "Paragraph body."][0]
    assert x.section_num == "1.1" and x.section_title == "Results / Encoder"
