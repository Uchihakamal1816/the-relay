import pytest
from relay.results_csv import ResultsWriter, COLUMNS
from relay.config import load_config
from relay.trace import split

ROW = dict(request_id="r1", source="stand-in", start_time="2026-01-01T00:00:00.000Z", first_token_ms=10,
           total_ms=20, status="ok", retries=0, tokens_in=1, tokens_out=2, retrieval_setting="S3", answer_grade="")


def test_header_exact(tmp_path):
    w = ResultsWriter(tmp_path / "r.csv"); w.replace_scenario("a", [ROW])
    assert (tmp_path / "r.csv").read_text().splitlines()[0] == ",".join(COLUMNS)


def test_replace_scenario(tmp_path):
    w = ResultsWriter(tmp_path / "r.csv")
    w.replace_scenario("a", [ROW]); w.replace_scenario("a", [ROW, {**ROW, "request_id": "r2"}]); w.replace_scenario("b", [ROW])
    rows = w.read()
    assert [r["scenario"] for r in rows] == ["a", "a", "b"]


@pytest.mark.parametrize("k,v", [("status", "done"), ("source", "standin"), ("answer_grade", "ok")])
def test_rejects_bad_values(tmp_path, k, v):
    with pytest.raises(ValueError):
        ResultsWriter(tmp_path / "r.csv").replace_scenario("a", [{**ROW, k: v}])


def test_accepts_empty(tmp_path):
    ResultsWriter(tmp_path / "r.csv").replace_scenario("a", [{**ROW, "first_token_ms": None, "answer_grade": None}])


def test_config_loads():
    cfg = load_config()
    s = cfg.retrieval.settings["S3"]
    assert s.k == 4 and s.rerank is True and s.tag == "S3:c200-k4-rr"


def test_split():
    t = {"first_ms": 120, "end_ms": 200, "spans": [
        {"kind": "wait", "start_ms": 0, "end_ms": 100}, {"kind": "work", "start_ms": 100, "end_ms": 150},
        {"kind": "retry", "start_ms": 150, "end_ms": 180}]}
    s = split(t)
    assert s["wait_ms"] == 100 and s["retry_ms"] == 30 and s["before_first_ms"] == 120 and s["after_first_ms"] == 80
