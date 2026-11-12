from relay.evalx.questions import Question, Gold, validate_questions
from relay.evalx.grader import grade
from relay.ingest.parse_html import Block
from relay.types import Reply, Chunk, Level

C = {"A1": Chunk("A1", "attention", "6.1", "MT", "table", "Table 2", 8, "BLEU 28.4", 5),
     "A2": Chunk("A2", "attention", "3.1", "Enc", "text", None, 3, "N = 6", 5)}
QA = Question("q1", "BLEU?", True, gold=[Gold("attention", "6.1", "Table 2")], must_include=[["28.4"]])
QU = Question("q2", "Optimizer of Moshi codec?", False)


def R(text, ids=(), refused=False, level=Level.FULL):
    return Reply("r", text=text, cited_ids=list(ids), refused=refused, level=level)


def test_grader_matrix():
    assert grade(QA, R("It is 28.4", ["A1"]), C) == "correct"
    assert grade(QA, R("It is 28.4", ["A2"]), C) == "wrong"          # right number, wrong place
    assert grade(QA, R("It is 27.3", ["A1"]), C) == "wrong"
    assert grade(QA, R("no", refused=True), C) == "refused_wrongly"
    assert grade(QU, R("no", refused=True), C) == "refused_correctly"
    assert grade(QU, R("Adam [A2]", ["A2"]), C) == "wrong"
    assert grade(QA, R("quotes", level=Level.EXTRACTIVE), C) is None
    assert grade(QA, None, C) is None


def test_validate():
    bl = [Block("attention", "6.1", "MT", "table", "Table 2", "BLEU 28.4", 0)]
    assert validate_questions([QA, QU], bl) == []
    bad = Question("q3", "x", True, gold=[Gold("attention", "6.1")], must_include=[["99.9"]])
    assert any("99.9" in p for p in validate_questions([bad], bl))
    assert any("gold" in p for p in validate_questions([Question("q4", "x", False, gold=[Gold("attention")])], bl))
