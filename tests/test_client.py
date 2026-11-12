import asyncio, json
import httpx, pytest
from relay.llm.client import LLMClient, LLMRequest, LLMError, parse_duration, classify_429, redact


def sse(chunks, done=True):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + ("data: [DONE]\n\n" if done else "")
    return body.encode()


def mk(handler, ledger=None):
    return LLMClient("http://x/v1", "gsk_SECRET", "real" if ledger else "stand-in", transport=httpx.MockTransport(handler), ledger=ledger)


REQ = LLMRequest("openai/gpt-oss-120b", [{"role": "user", "content": "hi"}], 50)
C = lambda **d: {"choices": [{"delta": d}]}


def run(cl, **kw):
    return asyncio.run(cl.stream(REQ, ttft_timeout_s=kw.get("ttft", 2), total_timeout_s=5, on_delta=kw.get("cb")))


def test_parse_duration():
    assert parse_duration("2m59.56s") == 179.56 and parse_duration("7.66s") == 7.66
    assert parse_duration("120ms") == 0.12 and parse_duration("1h2m3s") == 3723.0 and parse_duration("2") == 2.0


def test_stream_content_and_reasoning():
    seen = []
    cl = mk(lambda r: httpx.Response(200, content=sse([C(reasoning="think"), C(content="Hello"), C(content=" world"),
                                                       {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}])))
    rec = run(cl, cb=seen.append)
    assert [d.kind for d in seen] == ["reasoning", "content", "content"] and rec.content == "Hello world"
    assert rec.t_first_reasoning <= rec.t_first_content and rec.usage["total_tokens"] == 5 and not rec.usage_estimated


def test_usage_in_x_groq():
    cl = mk(lambda r: httpx.Response(200, content=sse([C(content="a"), {"choices": [], "x_groq": {"usage": {"total_tokens": 9}}}])))
    assert run(cl).usage["total_tokens"] == 9


def test_usage_estimated_when_missing():
    rec = run(mk(lambda r: httpx.Response(200, content=sse([C(content="abc")]))))
    assert rec.usage_estimated and rec.usage["completion_tokens"] >= 1


@pytest.mark.parametrize("body,hdr,kind", [
    ('{"error":{"message":"... on tokens per minute (TPM): Limit 8000"}}', {"retry-after": "7"}, "rate_limited_minute"),
    ('{"error":{"message":"... on requests per day (RPD): Limit 1000"}}', {"retry-after": "60"}, "rate_limited_day"),
    ("slow down", {"retry-after": "900"}, "rate_limited_day"),
    ("slow down", {}, "rate_limited_minute")])
def test_429(body, hdr, kind):
    with pytest.raises(LLMError) as e:
        run(mk(lambda r: httpx.Response(429, content=body.encode(), headers=hdr)))
    assert e.value.kind == kind


def test_500_is_server():
    with pytest.raises(LLMError) as e:
        run(mk(lambda r: httpx.Response(503, content=b"x")))
    assert e.value.kind == "server"


def test_midstream_cut():
    with pytest.raises(LLMError) as e:
        run(mk(lambda r: httpx.Response(200, content=sse([C(content="a")], done=False))))
    assert e.value.kind == "midstream"


def test_ttft_timeout():
    async def slow(request):
        await asyncio.sleep(1); return httpx.Response(200, content=sse([C(content="a")]))
    with pytest.raises(LLMError) as e:
        run(mk(slow), ttft=0.05)
    assert e.value.kind == "timeout_ttft"


def test_redact_and_ledger(tmp_path):
    assert "authorization" not in redact({"Authorization": "Bearer x", "x-ratelimit-limit-tokens": "8000"})
    led = tmp_path / "l.jsonl"
    run(mk(lambda r: httpx.Response(200, content=sse([C(content="a")]), headers={"x-ratelimit-remaining-tokens": "10"}), ledger=led))
    txt = led.read_text()
    assert "gsk_SECRET" not in txt and "x-ratelimit-remaining-tokens" in txt
