from __future__ import annotations
import asyncio, json, re, time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping
import httpx

LEDGER = Path("raw/real_calls.jsonl")


@dataclass
class LLMRequest:
    model: str
    messages: list[dict]
    max_completion_tokens: int
    reasoning_effort: str | None = "low"
    temperature: float = 0.2
    json_mode: bool = False
    seed: int | None = 7
    trace: str | None = None


@dataclass
class Delta:
    kind: Literal["reasoning", "content"]
    text: str
    t: float


@dataclass
class CallRecord:
    model: str
    status: int = 0
    t_send: float = 0.0
    t_headers: float | None = None
    t_first_reasoning: float | None = None
    t_first_content: float | None = None
    t_done: float | None = None
    usage: dict | None = None
    usage_estimated: bool = False
    headers: dict = field(default_factory=dict)
    error_body: str | None = None
    content: str = ""
    wall: str = ""


class LLMError(Exception):
    def __init__(self, kind: str, record: CallRecord, retry_after_s: float | None = None):
        super().__init__(f"{kind}: {record.error_body or ''}"[:300])
        self.kind, self.record, self.retry_after_s = kind, record, retry_after_s


_DUR = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        pass
    mult = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return round(sum(float(v) * mult[u] for v, u in _DUR.findall(s)), 6)


def classify_429(body: str, retry_after_s: float | None) -> str:
    b = (body or "").lower()
    if "per day" in b or "(rpd)" in b or "(tpd)" in b or (retry_after_s is not None and retry_after_s > 300):
        return "rate_limited_day"
    return "rate_limited_minute"


def redact(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items() if k.lower() not in ("authorization", "set-cookie", "cookie")}


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


class LLMClient:
    def __init__(self, base_url: str, api_key: str, source: Literal["real", "stand-in"],
                 transport: httpx.AsyncBaseTransport | None = None, ledger: Path | None = LEDGER):
        self.base_url, self.source = base_url.rstrip("/"), source
        self._key = api_key
        self.http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(60, connect=10),
                                      limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
        self.ledger = ledger if source == "real" else None

    def body(self, req: LLMRequest) -> dict:
        b = {"model": req.model, "messages": req.messages, "stream": True, "stream_options": {"include_usage": True},
             "max_completion_tokens": req.max_completion_tokens, "temperature": req.temperature}
        if req.seed is not None:
            b["seed"] = req.seed
        if req.reasoning_effort:
            b["reasoning_effort"] = req.reasoning_effort
        if _is_reasoning_model(req.model):
            b["include_reasoning"] = True
        if req.json_mode:
            b["response_format"] = {"type": "json_object"}
        return b

    async def stream(self, req: LLMRequest, *, ttft_timeout_s: float, total_timeout_s: float,
                     on_delta: Callable[[Delta], None] | None = None) -> CallRecord:
        rec = CallRecord(model=req.model, wall=datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        hdr = {"Authorization": f"Bearer {self._key}"}
        if req.trace and self.source == "stand-in":
            hdr["x-relay-trace"] = req.trace
        rec.t_send = time.perf_counter()
        first_deadline = rec.t_send + ttft_timeout_s
        end_deadline = rec.t_send + total_timeout_s
        resp = None
        try:
            try:
                resp = await asyncio.wait_for(self.http.send(
                    self.http.build_request("POST", f"{self.base_url}/chat/completions", json=self.body(req), headers=hdr),
                    stream=True), ttft_timeout_s)
            except asyncio.TimeoutError:
                rec.status = 0; raise LLMError("timeout_ttft", rec)
            except httpx.TransportError as e:
                rec.error_body = type(e).__name__; raise LLMError("connection", rec)
            rec.t_headers = time.perf_counter()
            rec.status = resp.status_code
            rec.headers = redact(resp.headers)
            if resp.status_code != 200:
                rec.error_body = (await resp.aread()).decode(errors="replace")[:2000]
                ra = parse_duration(resp.headers.get("retry-after"))
                if resp.status_code == 429:
                    raise LLMError(classify_429(rec.error_body, ra), rec, ra)
                raise LLMError("server" if resp.status_code >= 498 else "bad_request", rec, ra)
            done = False
            it = resp.aiter_lines()
            while True:
                now = time.perf_counter()
                got_first = rec.t_first_content is not None or rec.t_first_reasoning is not None
                limit = (end_deadline if got_first else min(first_deadline, end_deadline)) - now
                try:
                    line = await asyncio.wait_for(it.__anext__(), max(0.001, limit))
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise LLMError("timeout_total" if got_first else "timeout_ttft", rec)
                except httpx.TransportError as e:
                    rec.error_body = type(e).__name__; raise LLMError("midstream", rec)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True; break
                try:
                    ch = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in ch:
                    rec.error_body = json.dumps(ch["error"])[:2000]; raise LLMError("midstream", rec)
                u = ch.get("usage") or (ch.get("x_groq") or {}).get("usage")
                if u:
                    rec.usage = u
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    t = time.perf_counter()
                    if d.get("reasoning"):
                        rec.t_first_reasoning = rec.t_first_reasoning or t
                        on_delta and on_delta(Delta("reasoning", d["reasoning"], t))
                    if d.get("content"):
                        rec.t_first_content = rec.t_first_content or t
                        rec.content += d["content"]
                        on_delta and on_delta(Delta("content", d["content"], t))
            if not done:
                raise LLMError("midstream", rec)
            rec.t_done = time.perf_counter()
            if not rec.usage:
                from relay.ingest.chunk import count_tokens
                pt = sum(count_tokens(m.get("content") or "") for m in req.messages)
                rec.usage = {"prompt_tokens": pt, "completion_tokens": count_tokens(rec.content), "total_tokens": None}
                rec.usage["total_tokens"] = rec.usage["prompt_tokens"] + rec.usage["completion_tokens"]
                rec.usage_estimated = True
            return rec
        except asyncio.CancelledError:
            rec.status = rec.status or 499; rec.error_body = "cancelled"
            raise
        finally:
            if resp is not None:
                await resp.aclose()
            self._log(req, rec)

    def _log(self, req: LLMRequest, rec: CallRecord) -> None:
        if not self.ledger:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        row = {k: v for k, v in asdict(rec).items() if k != "content"}
        row.update(max_completion_tokens=req.max_completion_tokens, reasoning_effort=req.reasoning_effort,
                   content_chars=len(rec.content))
        with self.ledger.open("a") as f:
            f.write(json.dumps(row) + "\n")

    async def aclose(self):
        await self.http.aclose()
