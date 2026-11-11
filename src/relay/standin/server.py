"""Stand-in model service speaking Groq's OpenAI-compatible streaming API."""
from __future__ import annotations
import asyncio, copy, json, math, random, re, time
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from relay.llm.governor import Bucket
from relay.ingest.chunk import count_tokens

PROFILE = Path("config/standin_profile.json")
DEFAULT_LAT = {"ttft_base_ms": 250.0, "ttft_per_1k_prompt_ms": 60.0, "ttft_sigma": 0.35, "tps_mu": math.log(450),
               "tps_sigma": 0.25, "reasoning_tokens_mu": 60.0, "p_slow": 0.03, "slow_factor": 4.0}
DEFAULTS = {
    "limits": {"*": {"rpm": 30, "tpm": 8000, "rpd": 1000, "tpd": 200000}},
    "latency": {"*": DEFAULT_LAT},
    "load": {"c0": 8, "alpha": 0.3},
    "faults": {"p_500": 0.0, "p_503": 0.0, "p_hang": 0.0, "p_midstream_cut": 0.0, "p_slow_success": 0.0, "slow_success_factor": 5.0},
    "time_scale": 1.0, "seed": 7, "recordings": "raw/recordings",
    "error_429": None,  # captured real body template, filled by calibrate
}
ABBR = {"openai/gpt-oss-120b": "120b", "openai/gpt-oss-20b": "20b", "qwen/qwen3.8-27b": "qwen"}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def fmt_dur(s: float) -> str:
    s = max(0.0, s)
    return f"{int(s // 60)}m{s % 60:.2f}s" if s >= 60 else f"{s:.2f}s"


class Model:
    def __init__(self, lim: dict, now: float):
        self.lim = dict(lim)
        self.b = {k: Bucket(lim[k], lim[k] / (60 if k in ("rpm", "tpm") else 86400), now) for k in ("rpm", "tpm", "rpd", "tpd")}
        self.in_flight = 0

    def resize(self, lim: dict, now: float):
        for k, v in lim.items():
            if k in self.b and v != self.lim.get(k):
                self.b[k].resize(v, now)
        self.lim.update(lim)


class State:
    def __init__(self, cfg: dict):
        self.cfg = deep_merge(DEFAULTS, cfg)
        self.rng = random.Random(self.cfg["seed"])
        self.models: dict[str, Model] = {}
        self.stats = {"requests": 0, "ok": 0, "429": 0, "5xx": 0, "hang": 0, "cut": 0, "replay_hit": 0, "replay_miss": 0}

    def lim(self, m):
        L = self.cfg["limits"]; return {**L.get("*", {}), **L.get(m, {})}

    def lat(self, m):
        L = self.cfg["latency"]; return {**DEFAULT_LAT, **L.get("*", {}), **L.get(m, {})}

    def model(self, m):
        if m not in self.models:
            self.models[m] = Model(self.lim(m), time.monotonic())
        return self.models[m]

    def apply(self, patch: dict, reset: bool = False):
        self.cfg = deep_merge(self.cfg, patch)
        if reset:
            self.models.clear(); self.rng = random.Random(self.cfg["seed"])
            for k in self.stats: self.stats[k] = 0
        now = time.monotonic()
        for m, mod in self.models.items():
            mod.resize(self.lim(m), now)


def _replay(st: State, trace: str | None, prompt: str) -> str:
    tags = dict(kv.split("=", 1) for kv in (trace or "").split(";") if "=" in kv)
    root = Path(st.cfg["recordings"]) / tags.get("stage", "draft")
    for att in (tags.get("attempt", "0"), "0"):
        f = root / f"{tags.get('qid')}__{tags.get('setting')}__{tags.get('model')}__{att}.json"
        if tags and f.exists():
            st.stats["replay_hit"] += 1
            return json.loads(f.read_text())["content"]
    st.stats["replay_miss"] += 1
    if tags.get("stage") == "understand" or '"in_scope"' in prompt:
        # no recording: answer like a reasonable router would (keyword routing), not a fixed "all papers"
        from relay.pipeline.understand import heuristic_understand
        u = heuristic_understand(prompt.rsplit("QUESTION:", 1)[-1].strip())
        return json.dumps({"in_scope": True, "papers": u.papers, "ambiguous": u.ambiguous, "queries": []})
    ids = re.findall(r"^\[([AMH]\d+)\]", prompt, re.M)
    return f"The passages discuss this topic [{ids[0]}]" if ids else "NOT_IN_PAPERS"


def _headers(mod: Model, now: float) -> dict:
    b = mod.b
    return {"x-ratelimit-limit-requests": str(int(mod.lim["rpd"])), "x-ratelimit-remaining-requests": str(max(0, int(b["rpd"].level))),
            "x-ratelimit-reset-requests": fmt_dur((b["rpd"].capacity - b["rpd"].level) / b["rpd"].refill),
            "x-ratelimit-limit-tokens": str(int(mod.lim["tpm"])), "x-ratelimit-remaining-tokens": str(max(0, int(b["tpm"].level))),
            "x-ratelimit-reset-tokens": fmt_dur((b["tpm"].capacity - b["tpm"].level) / b["tpm"].refill)}


NAMES = {"rpm": "requests per minute (RPM)", "tpm": "tokens per minute (TPM)", "rpd": "requests per day (RPD)", "tpd": "tokens per day (TPD)"}


def create_app(cfg: dict | None = None) -> FastAPI:
    app = FastAPI()
    if cfg is None and PROFILE.exists():
        cfg = json.loads(PROFILE.read_text())
    st = State(cfg or {})
    app.state.st = st

    @app.get("/openai/v1/models")
    async def models():
        return {"data": [{"id": m} for m in ABBR]}

    @app.post("/admin/config")
    async def admin(req: Request):
        body = await req.json()
        st.apply(body.get("patch", body), reset=bool(body.get("reset")))
        return {"ok": True, "cfg": st.cfg}

    @app.get("/admin/stats")
    async def stats():
        return {**st.stats, "levels": {m: {k: round(b.level, 1) for k, b in mod.b.items()} for m, mod in st.models.items()}}

    @app.post("/openai/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        m = body["model"]; mod = st.model(m); lat = st.lat(m); ts = st.cfg["time_scale"]; f = st.cfg["faults"]
        st.stats["requests"] += 1
        prompt = "\n".join(x.get("content") or "" for x in body["messages"])
        ptok = count_tokens(prompt)
        reserve = ptok + int(body.get("max_completion_tokens") or 1024)
        now = time.monotonic()
        for k, amt in (("rpm", 1), ("rpd", 1), ("tpm", reserve), ("tpd", reserve)):
            at = mod.b[k].available_at(amt, now)
            if at > now:
                st.stats["429"] += 1
                ra = math.ceil(min(at - now, 86400)) if at != math.inf else 86400
                used = int(mod.lim[k] - mod.b[k].level)
                msg = (f"Rate limit reached for model `{m}` in organization `org_standin` service tier `on_demand` on "
                       f"{NAMES[k]}: Limit {int(mod.lim[k])}, Used {used}, Requested {amt}. Please try again in {fmt_dur(at - now if at != math.inf else 86400)}.")
                tpl = st.cfg.get("error_429")
                err = json.loads(json.dumps(tpl).replace("__MSG__", msg)) if tpl else \
                    {"error": {"message": msg, "type": "tokens" if "t" == k[0] else "requests", "code": "rate_limit_exceeded"}}
                return JSONResponse(err, status_code=429, headers={**_headers(mod, now), "retry-after": str(ra)})
        for k, amt in (("rpm", 1), ("rpd", 1), ("tpm", reserve), ("tpd", reserve)):
            mod.b[k].take(amt, now)
        r = st.rng.random()
        def refund_all():
            mod.b["tpm"].refund(reserve, time.monotonic()); mod.b["tpd"].refund(reserve, time.monotonic())
        if r < f["p_500"]:
            st.stats["5xx"] += 1; refund_all()
            return JSONResponse({"error": {"message": "Internal Server Error", "type": "internal_server_error"}}, 500)
        if r < f["p_500"] + f["p_503"]:
            st.stats["5xx"] += 1; refund_all()
            return JSONResponse({"error": {"message": "Service Unavailable", "type": "service_unavailable"}}, 503)
        hang = r < f["p_500"] + f["p_503"] + f["p_hang"]
        cut = st.rng.random() < f["p_midstream_cut"]
        slow = st.rng.random() < f["p_slow_success"]
        content = _replay(st, req.headers.get("x-relay-trace"), prompt)
        words = re.findall(r"\S+\s*", content) or [content]
        pieces = ["".join(words[i:i + 3]) for i in range(0, len(words), 3)]
        ctok = count_tokens(content)
        rtok = int(max(0, st.rng.expovariate(1 / lat["reasoning_tokens_mu"]))) if body.get("reasoning_effort") and m.startswith("openai/") else 0
        mod.in_flight += 1
        load = 1 + st.cfg["load"]["alpha"] * (mod.in_flight - 1) / st.cfg["load"]["c0"]
        ttft = (lat["ttft_base_ms"] + lat["ttft_per_1k_prompt_ms"] * ptok / 1000) * st.rng.lognormvariate(0, lat["ttft_sigma"]) * load / 1000
        if st.rng.random() < lat["p_slow"]:
            ttft *= lat["slow_factor"]
        tps = st.rng.lognormvariate(lat["tps_mu"], lat["tps_sigma"]) / load
        if slow:
            ttft *= f["slow_success_factor"]; tps /= f["slow_success_factor"]
        cut_at = st.rng.randint(0, max(0, len(pieces) - 1)) if cut else None
        hdrs = _headers(mod, now)

        async def gen():
            t0 = time.monotonic()
            try:
                if hang:
                    st.stats["hang"] += 1
                    await asyncio.sleep(3600); return
                await asyncio.sleep(ttft * ts)
                base = {"id": "chatcmpl-standin", "object": "chat.completion.chunk", "model": m}
                for i in range(0, rtok, 4):
                    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'reasoning': ' …'}}]})}\n\n"
                    await asyncio.sleep(min(4, rtok - i) / tps * ts)
                for i, p in enumerate(pieces):
                    if cut_at is not None and i >= cut_at:
                        st.stats["cut"] += 1; return
                    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': p}}]})}\n\n"
                    await asyncio.sleep(count_tokens(p) / tps * ts)
                total = time.monotonic() - t0
                usage = {"queue_time": round(ttft * 0.1 * ts, 6), "prompt_tokens": ptok, "prompt_time": round(ttft * 0.9 * ts, 6),
                         "completion_tokens": ctok + rtok, "completion_time": round(total - ttft * ts, 6),
                         "total_tokens": ptok + ctok + rtok, "total_time": round(total, 6),
                         "completion_tokens_details": {"reasoning_tokens": rtok}}
                yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}], 'x_groq': {'usage': usage}, 'usage': usage})}\n\n"
                yield "data: [DONE]\n\n"
                st.stats["ok"] += 1
                n = time.monotonic()
                mod.b["tpm"].refund(reserve - (ptok + ctok + rtok), n); mod.b["tpd"].refund(reserve - (ptok + ctok + rtok), n)
            finally:
                mod.in_flight -= 1

        return StreamingResponse(gen(), media_type="text/event-stream", headers=hdrs)

    return app


def cli(argv):
    import argparse, uvicorn
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", nargs="?", default="serve"); ap.add_argument("--port", type=int, default=8900)
    a = ap.parse_args(argv)
    uvicorn.run(create_app(), host="127.0.0.1", port=a.port, log_level="warning")
