import asyncio, json, math
import httpx, pytest
from relay.config import load_config
from relay.ingest.parse_html import Block
from relay.ingest.chunk import chunk_blocks
from relay.retrieve.index import Index
from relay.retrieve.tool import PaperSearch
from relay.retrieve.papermap import PaperMap
from relay.standin.server import create_app
from relay.llm.client import LLMClient
from relay.llm.governor import Governor
from relay.pipeline.orchestrator import Relay
from relay.types import Level, Limits

BLOCKS = [Block("attention", "3", "Model", "text", None, "The encoder is composed of a stack of N = 6 identical layers.", 0),
          Block("attention", "5", "Training", "text", None, "We trained on 8 P100 GPUs for 12 hours.", 1),
          Block("moshi", "3", "Model", "text", None, "Mimi runs at a frame rate of 12.5 Hz with 8 codebooks.", 2),
          Block("hnet", "2", "Method", "text", None, "The routing module predicts boundaries between bytes.", 3),
          Block("hnet", "3", "Results", "table", "Table 5", "Table 5: HG38 results\nModel | ppl\nH-Net | 2.705", 4)]
FAST = {"latency": {"*": {"ttft_base_ms": 30, "ttft_per_1k_prompt_ms": 0, "ttft_sigma": 0, "tps_mu": math.log(20000), "tps_sigma": 0, "p_slow": 0, "reasoning_tokens_mu": 1}},
        "time_scale": 1.0, "load": {"c0": 1, "alpha": 0}}


class RR:
    def score(self, q, texts): return [0.0] * len(texts)


def make(tmp_path, scripts: dict, standin=None, cfg_over=None, limits=None, papermap=False):
    rec = tmp_path / "rec"
    for key, content in scripts.items():
        stage, rest = key.split(":", 1)
        (rec / stage).mkdir(parents=True, exist_ok=True)
        (rec / stage / f"{rest}.json").write_text(json.dumps({"content": content}))
    app = create_app({**FAST, "recordings": str(rec), **(standin or {})})
    cfg = load_config(overrides={"retrieval.settings": {"S3": {"chunk_tokens": 200, "k": 3, "rerank": False},
                                                        "S4": {"chunk_tokens": 200, "k": 2, "rerank": False}},
                                 "understand.use_llm": False, **(cfg_over or {})})
    client = LLMClient("http://s/openai/v1", "k", "stand-in", transport=httpx.ASGITransport(app=app))
    ch = chunk_blocks(BLOCKS, 200)
    ps = PaperSearch(indexes={200: Index(ch, None)}, reranker=RR())
    lim = limits or {m: Limits(30, 8000, 1000, 200000) for m in (cfg.models.primary.id, cfg.models.fallback.id, cfg.models.sacrificial.id)}
    gov = Governor(lim, safety=1.0)
    pm = PaperMap.build(BLOCKS) if papermap else None
    return Relay(cfg, client, gov, ps, "stand-in", papermap=pm, traces_dir=tmp_path / "tr"), app, {c.text[:20]: c.id for c in ch}


def ask(relay, q, qid="q1", **kw):
    ev = []
    r = asyncio.run(relay.ask(q, request_id=qid, trace_tags={"qid": qid}, on_event=ev.append, **kw))
    return r, ev


Q = "How many layers does the encoder have?"


def test_happy_path_events(tmp_path):
    relay, _, ids = make(tmp_path, {"draft:q1__S3__120b__0": "The encoder has 6 identical layers [A1]\n"})
    r, ev = ask(relay, Q)
    assert ev[0].kind == "status" and ev[1].kind == "sentence" and "(Attention, §3)" in ev[1].text
    assert r.status == "ok" and r.first_token_ms is not None and r.level == Level.FULL and r.tokens_in > 0


def test_first_line_fails_then_redraft(tmp_path):
    relay, app, _ = make(tmp_path, {"draft:q1__S3__120b__0": "It has 512 layers [A1]\n", "draft:q1__S3__120b__1": "Six identical layers, N = 6 [A1]\n"})
    r, ev = ask(relay, Q)
    assert r.status == "ok" and "N = 6" in r.text and app.state.st.stats["requests"] == 2


def test_never_emits_unverified(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S3__120b__0": "It has 512 layers [A1]\n", "draft:q1__S3__120b__1": "Remember: 99 layers\n"})
    r, ev = ask(relay, Q)
    assert r.level == Level.EXTRACTIVE and all("512" not in e.text and "99" not in e.text for e in ev)


def test_later_line_fails(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S3__120b__0": "Six layers, N = 6 [A1]\nAlso 77 heads [A1]\nMore [A1]\n"})
    r, ev = ask(relay, Q)
    assert [e.kind for e in ev] == ["status", "sentence", "notice"] and r.truncated and r.status == "ok"


def test_sentinel_refusal(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S3__120b__0": "NOT_IN_PAPERS\n"})
    r, ev = ask(relay, "Which optimizer did Moshi use for the codec?")
    assert r.refused and "don't answer" in r.text


def test_daily_primary_exhausted_uses_fallback(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S4__20b__0": "N = 6 identical layers [A1]\n"})
    relay.gov.for_model(relay.P).day_reset_at = relay.gov.for_model(relay.P).clock() + 3600
    r, _ = ask(relay, Q)
    assert r.retrieval_setting.endswith("@20b") and r.status == "ok" and r.level == Level.FALLBACK


def test_all_exhausted_extractive(tmp_path):
    relay, app, _ = make(tmp_path, {})
    for m in (relay.P, relay.F):
        g = relay.gov.for_model(m); g.day_reset_at = g.clock() + 5400
    r, ev = ask(relay, Q)
    assert r.status == "rate_limited" and "1 h 29 min" in r.text and app.state.st.stats["requests"] == 0


def test_standin_429_daily_mid_run_falls_back(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S4__20b__0": "N = 6 layers [A1]\n"},
                       standin={"limits": {"openai/gpt-oss-120b": {"rpd": 0.5}}})
    r, _ = ask(relay, Q)
    assert r.status == "ok" and r.retrieval_setting.endswith("@20b")


def test_ambiguous_question_per_paper(tmp_path):
    relay, _, _ = make(tmp_path, {})
    from relay.pipeline.understand import heuristic_understand
    from relay.trace import Trace
    u = heuristic_understand("What is the frame rate and how many layers?")
    hits = asyncio.run(relay._retrieve("What is the frame rate and how many layers?", u, relay.cfg.retrieval.settings["S3"], Trace("x")))
    assert {h.chunk.paper for h in hits} == {"attention", "moshi", "hnet"}


def test_coalesce_duplicates(tmp_path):
    relay, app, _ = make(tmp_path, {"draft:q1__S3__120b__0": "N = 6 identical layers [A1]\n"})
    async def two():
        return await asyncio.gather(relay.ask(Q, request_id="a", trace_tags={"qid": "q1"}), relay.ask(Q, request_id="b", trace_tags={"qid": "q1"}))
    a, b = asyncio.run(two())
    assert app.state.st.stats["requests"] == 1 and b.retrieval_setting.endswith("+cache") and b.tokens_in == 0 and b.first_token_ms is not None


def test_cancel_releases_grant(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S3__120b__0": "N = 6 layers [A1]\n"},
                       standin={"latency": {"*": {**FAST["latency"]["*"], "ttft_base_ms": 3000}}})
    g = relay.gov.for_model(relay.P)
    async def go():
        t = asyncio.ensure_future(relay.ask(Q, request_id="a", trace_tags={"qid": "q1"}))
        await asyncio.sleep(0.3); lvl = g.tpm.level; t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        await asyncio.sleep(0.05)
        return lvl
    lvl = asyncio.run(go())
    assert g.tpm.level > lvl + 300


def test_stream_without_usage(tmp_path, monkeypatch):
    relay, _, _ = make(tmp_path, {"draft:q1__S3__120b__0": "N = 6 layers [A1]\n"})
    orig = relay.client.stream
    async def nousage(*a, **k):
        rec = await orig(*a, **k); rec.usage = None; return rec
    monkeypatch.setattr(relay.client, "stream", nousage)
    r, _ = ask(relay, Q)
    assert r.status == "ok"


def test_midstream_after_first_line(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S3__120b__0": "N = 6 identical layers [A1]\n" + "filler words here " * 30 + "[A1]\n"},
                       standin={"faults": {"p_midstream_cut": 1.0}, "seed": 3})
    r, ev = ask(relay, Q)
    if r.truncated:
        assert r.status == "ok" and ev[-1].kind == "notice" and r.retries == 0


def test_patched_table_answer(tmp_path):
    relay, _, _ = make(tmp_path, {"draft:q1__S3__120b__0": "H-Net reaches a final perplexity of 2.705 on HG38 [H2]\n"}, papermap=True)
    r, _ = ask(relay, "What does Table 5 of the H-Net paper report?")
    assert r.status == "ok" and "Table 5" in r.text


def test_map_hit_replaces_lowest_and_disabled(tmp_path):
    from relay.pipeline.understand import heuristic_understand
    from relay.trace import Trace
    q = "What is in Table 5 of H-Net?"
    relay, _, _ = make(tmp_path, {}, papermap=True)
    s = relay.cfg.retrieval.settings["S3"]
    hits = asyncio.run(relay._retrieve(q, heuristic_understand(q), s, Trace("x")))
    assert hits[0].chunk.label == "Table 5" and hits[0].from_map and len(hits) <= s.k
    relay.cfg.papermap.enabled = False
    hits2 = asyncio.run(relay._retrieve(q, heuristic_understand(q), s, Trace("x")))
    assert not any(h.from_map for h in hits2)
