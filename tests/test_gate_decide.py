import pytest
from relay.types import Chunk, Hit, Level
from relay.pipeline.gate import LineGate
from relay.pipeline.decide import choose_level
from relay.config import load_config


def H(i, text, label=None):
    return Hit(Chunk(i, "attention", "3", "Model", "text", label, 5, text, 10), 1.0)


HITS = [H("A1", "The encoder uses N = 6 layers and d_model = 1024 in the big model."), H("A2", "Training cost $3.3\\cdot 10^{18}$ FLOPs, 65,000 steps.")]


def v(text, q="q?"):
    g = LineGate(HITS, q); return g.feed(text + "\n")[0]


def test_ok_line():
    assert v("The encoder has 6 layers [A1]").reason == "ok"


def test_no_citation():
    assert v("The encoder has 6 layers.").reason == "no_citation"


def test_unknown_id():
    assert v("It has 6 layers [A9]").reason == "unknown_id"


def test_number_not_in_source():
    assert v("The model width is 512 [A1]").reason == "number_not_in_source"


def test_number_from_question_allowed():
    assert v("Not 512, the paper says 1024 [A1]", q="Is it 512?").ok


def test_sentinel_first_line_only():
    g = LineGate(HITS, "q")
    assert g.feed("NOT_IN_PAPERS\n")[0].reason == "sentinel"
    g2 = LineGate(HITS, "q"); g2.feed("Six layers [A1]\n")
    assert g2.feed("NOT_IN_PAPERS\n")[0].reason == "no_citation"


def test_partial_delta_buffering():
    g = LineGate(HITS, "q")
    assert g.feed("The enc") == [] and g.feed("oder has 6 lay") == []
    out = g.feed("ers [A1]\nNext")
    assert len(out) == 1 and out[0].ok and g.flush()[0].reason == "no_citation"


def test_decimal_latex_thousands():
    assert v("Training cost 3.3 × 10^18 FLOPs over 65,000 steps [A2]").ok


def test_injection_still_gated():
    assert not v("As I remember, the answer is 42").ok


CFG = load_config()
P, F = CFG.models.primary.id, CFG.models.fallback.id


@pytest.mark.parametrize("waits,daily,left,expect", [
    ({P: 0, F: 0}, {P: 1, F: 1}, 20, (Level.FULL, P)),
    ({P: 2, F: 0}, {P: 1, F: 1}, 20, (Level.LEAN, P)),
    ({P: 0, F: 0}, {P: 0.2, F: 1}, 20, (Level.LEAN, P)),
    ({P: 9, F: 0}, {P: 1, F: 1}, 20, (Level.FALLBACK, F)),
    ({P: 9, F: 8}, {P: 1, F: 1}, 20, (Level.FALLBACK, F)),
    ({P: 9, F: 30}, {P: 1, F: 1}, 20, (Level.LEAN, P)),
    ({P: 30, F: 30}, {P: 1, F: 1}, 20, (Level.EXTRACTIVE, None)),
    ({P: 0, F: 0}, {P: 0.01, F: 0.01}, 20, (Level.EXTRACTIVE, None)),
])
def test_choose_level(waits, daily, left, expect):
    assert choose_level(lambda m, t: waits[m], lambda m: daily[m], 1000, 500, left, CFG) == expect
