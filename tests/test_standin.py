import asyncio, json, math
import httpx, numpy as np, pytest
from relay.standin.server import create_app
from relay.standin.calibrate import fit_samples
from relay.llm.client import LLMClient, LLMRequest, LLMError

M = "openai/gpt-oss-120b"
FAST = {"latency": {"*": {"ttft_base_ms": 50, "ttft_per_1k_prompt_ms": 0, "ttft_sigma": 0, "tps_mu": math.log(5000), "tps_sigma": 0, "p_slow": 0}},
        "time_scale": 0.01, "load": {"c0": 1, "alpha": 0}}


def client(cfg):
    app = create_app({**FAST, **cfg})
    return LLMClient("http://s/openai/v1", "k", "stand-in", transport=httpx.ASGITransport(app=app)), app


def call(cl, n=1, mct=10, trace=None, ttft=2):
    async def one():
        return await cl.stream(LLMRequest(M, [{"role": "user", "content": "hello\n[A1] (h) x"}], mct, trace=trace), ttft_timeout_s=ttft, total_timeout_s=5)
    async def many():
        return await asyncio.gather(*(one() for _ in range(n)), return_exceptions=True)
    return asyncio.run(many())


def test_stream_shape():
    cl, _ = client({})
    r = call(cl)[0]
    assert r.usage["total_tokens"] > 0 and r.content and "x-ratelimit-remaining-tokens" in r.headers and len([h for h in r.headers if h.startswith("x-ratelimit")]) == 6


def test_rpm_refusal():
    cl, _ = client({"limits": {"*": {"rpm": 3}}})
    res = call(cl, 4)
    errs = [e for e in res if isinstance(e, LLMError)]
    assert len(errs) == 1 and errs[0].kind == "rate_limited_minute" and errs[0].retry_after_s is not None
    assert json.loads(errs[0].record.error_body)["error"]["code"] == "rate_limit_exceeded"


def test_tpm_refusal():
    cl, _ = client({"limits": {"*": {"tpm": 3}}})  # admission is on input tokens only
    e = call(cl, mct=500)[0]
    assert isinstance(e, LLMError) and "(TPM)" in e.record.error_body


def test_daily_refusal():
    cl, _ = client({"limits": {"*": {"rpd": 2}}})
    res = call(cl, 3)
    e = [x for x in res if isinstance(x, LLMError)][0]
    assert e.kind == "rate_limited_day" and "(RPD)" in e.record.error_body and e.retry_after_s > 300


def test_admin_halves_limit():
    cl, app = client({})
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s") as c:
            await c.post("/admin/config", json={"limits": {"*": {"tpm": 4000}}})
    call(cl); asyncio.run(go())
    assert call(cl)[0].headers["x-ratelimit-limit-tokens"] == "4000"


@pytest.mark.parametrize("fault,kind", [("p_500", "server"), ("p_midstream_cut", "midstream"), ("p_hang", "timeout_ttft")])
def test_faults(fault, kind):
    cl, _ = client({"faults": {fault: 1.0}})
    e = call(cl, ttft=0.3)[0]
    assert isinstance(e, LLMError) and e.kind == kind


def test_slower_under_load():
    cfg = {"load": {"c0": 1, "alpha": 1.0}, "latency": {"*": {**FAST["latency"]["*"], "ttft_base_ms": 2000}}}
    cl, _ = client(cfg)
    one = call(cl, 1)[0]; eight = call(cl, 8)
    t1 = one.t_first_content - one.t_send
    t8 = np.mean([r.t_first_content - r.t_send for r in eight])
    assert t8 > t1 * 1.5


def test_replay_hit_and_miss(tmp_path):
    d = tmp_path / "draft"; d.mkdir()
    (d / "q1__S3__120b__0.json").write_text(json.dumps({"content": "Recorded answer [A1]"}))
    cl, app = client({"recordings": str(tmp_path)})
    assert call(cl, trace="stage=draft;qid=q1;setting=S3;model=120b;attempt=0")[0].content == "Recorded answer [A1]"
    assert "[A1]" in call(cl, trace="stage=draft;qid=zz;setting=S3;model=120b;attempt=0")[0].content


def test_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    x = rng.uniform(100, 3000, 400)
    y = (200 + 80 * x / 1000) * rng.lognormal(0, 0.3, 400)
    f = fit_samples(x, y, rng.lognormal(6, 0.2, 400))
    assert abs(f["ttft_sigma"] - 0.3) / 0.3 < 0.15 and abs(f["tps_mu"] - 6) < 0.05
