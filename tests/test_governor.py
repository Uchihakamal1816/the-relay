import asyncio, json, math, random
from datetime import datetime, timezone
import pytest
from relay.config import NS
from relay.types import Limits
from relay.llm.governor import Bucket, ModelGovernor, Governor, WouldMissDeadline, DailyExhausted, Grant
from relay.llm.client import CallRecord, LLMError
from relay.llm.retry import RetryPolicy


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


L = Limits(rpm=30, tpm=8000, rpd=1000, tpd=200000)


def mg(clock=None, safety=1.0):
    return ModelGovernor("m", L, clock or Clock(), safety)


def rec(headers=None, usage=None, body=None):
    r = CallRecord("m"); r.headers = headers or {}; r.usage = usage; r.error_body = body; return r


def test_refill():
    c = Clock(); b = Bucket(8000, 8000 / 60, c()); b.take(6000, c())
    assert b.available_at(4000, c()) == pytest.approx(c() + 15)


def test_resize_on_header():
    c = Clock(); g = mg(c); g.tpm.take(8000, c())
    w1 = g.wait_s(2000)
    g.on_response(Grant("m", 0, c()), rec({"x-ratelimit-limit-tokens": "4000"}))
    assert g.wait_s(2000) == pytest.approx(2 * w1, rel=0.05)


def test_clamp_and_refund_uses_uncached():
    c = Clock(); g = mg(c); g.tpm.take(2000, c())
    g.on_response(Grant("m", 2000, c()), rec(usage={"total_tokens": 1500, "prompt_tokens_details": {"cached_tokens": 400}}))
    assert g.tpm.level == pytest.approx(6000 + 900)
    g.on_response(Grant("m", 0, c()), rec({"x-ratelimit-remaining-tokens": "500"}))
    assert g.tpm.level == 500


def test_deadline_no_sleep():
    c = Clock(); g = mg(c); g.tpm.take(8000, c())
    with pytest.raises(WouldMissDeadline):
        asyncio.run(g.acquire(5000, c() + 5))
    assert c() == 1000.0


def test_daily_exhausted():
    c = Clock(); g = mg(c)
    g.on_error(Grant("m", 10, c()), LLMError("rate_limited_day", rec(body="per day"), 5000))
    assert g.wait_s(10) == math.inf and g.daily_frac_left() == 0
    with pytest.raises(DailyExhausted):
        asyncio.run(g.acquire(10, c() + 100))


def test_minute_block():
    c = Clock(); g = mg(c)
    g.on_error(Grant("m", 10, c()), LLMError("rate_limited_minute", rec(body="(TPM)"), 7))
    assert g.wait_s(10) >= 7


def test_release_returns_tokens():
    c = Clock(); g = mg(c)
    gr = asyncio.run(g.acquire(3000, c() + 1)); g.release(gr)
    assert g.tpm.level == 8000 and g.rpm.level == 30


def test_fifo():
    g = ModelGovernor("m", Limits(rpm=600, tpm=600, rpd=10000, tpd=10**7), safety=1.0)
    g.tpm.take(600, g.clock())
    order = []
    async def go(i):
        await g.acquire(5, g.clock() + 5); order.append(i)
    async def main():
        await asyncio.gather(*(go(i) for i in range(4)))
    asyncio.run(main())
    assert order == [0, 1, 2, 3]


def test_ledger_rebuild(tmp_path):
    led = tmp_path / "l.jsonl"
    led.write_text(json.dumps({"model": "openai/gpt-oss-120b", "wall": datetime.now(timezone.utc).isoformat(),
                               "usage": {"total_tokens": 150000}}) + "\n")
    cfg = NS(models=NS(primary=NS(id="openai/gpt-oss-120b"), fallback=NS(id="f"), sacrificial=NS(id="s")),
             limits_prior=L, governor=NS(safety=0.9))
    g = Governor.from_config(cfg, ledger=led)
    assert g.for_model("openai/gpt-oss-120b").daily_frac_left() == pytest.approx((180000 - 150000) / 180000, abs=0.01)


RC = NS(policy="budgeted", max_attempts=2, base_s=0.25, cap_s=2.0, budget_ratio=0.1, min_per_min=3)
E = lambda k, ra=None: LLMError(k, rec(), ra)


def pol(**kw):
    p = RetryPolicy(NS({**RC, **kw}), random.Random(1), Clock())
    for _ in range(100):
        p.note_request()
    return p


def test_retry_rules():
    p = pol()
    assert p.decide(E("server"), 0, sent_any=True) is None
    assert p.decide(E("rate_limited_day"), 0, False) is None
    assert p.decide(E("server"), 2, False) is None
    assert 0 <= p.decide(E("server"), 1, False) <= 0.5
    assert p.decide(E("rate_limited_minute", 3), 0, False) == 3


def test_budget_stops_retries():
    p = pol()
    got = [p.decide(E("server"), 0, False) for _ in range(20)]
    assert sum(x is not None for x in got) == 10


def test_naive_retries_429():
    p = pol(policy="naive")
    assert p.decide(E("rate_limited_minute", 5), 0, False) == 0.0 and p.decide(E("server"), 3, False) is None


def test_wait_counts_queue_ahead():
    c = Clock(); g = mg(c)
    assert g.wait_s(4000) == 0
    g.queue_len, g.queued_tokens = 1, 8000  # someone ahead holds a full minute of tokens
    assert g.wait_s(4000) == pytest.approx(30, rel=0.01)


def test_lock_wait_bounded_by_deadline():
    g = ModelGovernor("m", L, safety=1.0)
    async def main():
        await g._lock.acquire()
        with pytest.raises(WouldMissDeadline):
            await g.acquire(10, g.clock() + 0.05)
        assert g.queue_len == 0 and g.queued_tokens == 0
    asyncio.run(main())
