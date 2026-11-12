"""Real-service probes (Task 5). Every run spends the shared daily budget: run only with --real."""
from __future__ import annotations
import argparse, asyncio, json, math, os, random, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from relay.config import load_config
from relay.llm.client import LLMClient, LLMRequest, LLMError, LEDGER
from relay.llm.governor import Governor, ledger_usage

OUT = Path("raw/probe")
M120, M20, MQ = "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"


@dataclass
class Probe:
    model: str
    sizes: list          # prompt sizes in tokens, one entry per call
    max_out: int
    est_tokens: int      # budget cap (prompt + reserved output), checked by test
    paced: bool = True   # paced by the governor (stays under RPM/TPM); False = deliberately hits the limit
    effort: str | None = "low"


PROBES = {
    "latency": Probe(M120, [300] * 10 + [1200] * 10 + [2400] * 10, 150, 48000),
    "latency-small": Probe(M20, [350] * 20, 100, 9500),
    "concurrency": Probe(M20, [1, 2, 4, 8] * 3, 16, 10000),           # sizes = burst width; prompt ~200
    "rpm-edge": Probe(MQ, [40], 1, 3000, paced=False, effort=None),   # 40 back-to-back tiny calls
    "tpm-edge": Probe(MQ, [2500] * 6, 1000, 21000, paced=False, effort=None),
    "cache": Probe(M120, [1200] * 6 + [400] * 6, 40, 12000),
    "effort": Probe(M120, [1500] * 10, 400, 19000),                  # 5 prompts x {low, medium}
    "rpd-burn": Probe(MQ, [], 1, 0, paced=False, effort=None),        # until daily 429; G1
}


def _corpus_text() -> str:
    lines = Path("data/corpus/c400/chunks.jsonl").read_text().splitlines()
    return "\n".join(json.loads(l)["text"] for l in lines)


def make_prompt(tokens: int, seed: int, prefix: str = "") -> str:
    from relay.ingest.chunk import count_tokens
    words = _corpus_text().split()
    rng = random.Random(seed)
    start = rng.randrange(0, max(1, len(words) - tokens * 2))
    body = " ".join(words[start:start + int(tokens * 0.7)])
    q = "\n\nIn one sentence, what is the passage above about?"
    txt = prefix + body
    while count_tokens(txt + q) > tokens and len(body) > 50:
        body = body[: int(len(body) * 0.9)]; txt = prefix + body
    return txt + q


class Runner:
    def __init__(self, name: str, real: bool):
        cfg = load_config()
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("GROQ_API_KEY", "") if real else "standin"
        if real and not key:
            raise SystemExit("GROQ_API_KEY missing in .env")
        self.cfg, self.name = cfg, name
        self.client = LLMClient(cfg.endpoints.real if real else cfg.endpoints.standin, key, "real" if real else "stand-in")
        self.gov = Governor.from_config(cfg, ledger=LEDGER if real else None)
        out = OUT if real else Path("raw/probe_standin")  # never mix stand-in rows into the real calibration data
        out.mkdir(parents=True, exist_ok=True)
        self.f = (out / f"{name}.jsonl").open("a")
        self.in_flight = 0

    async def call(self, model, prompt, max_out, effort, paced=True, extra=None):
        req = LLMRequest(model, [{"role": "user", "content": prompt}], max_out, effort, temperature=0.2, seed=None)
        from relay.ingest.chunk import count_tokens
        est = count_tokens(prompt) + max_out
        mg = self.gov.for_model(model)
        g = await mg.acquire(est, math.inf) if paced else None
        self.in_flight += 1
        row = {"probe": self.name, "model": model, "in_flight": self.in_flight, "prompt_tokens_est": est - max_out,
               "max_out": max_out, "effort": effort, "wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **(extra or {})}
        try:
            rec = await self.client.stream(req, ttft_timeout_s=30, total_timeout_s=60)
            g and mg.on_response(g, rec)
            row["record"] = {k: v for k, v in asdict(rec).items() if k != "content"} | {"content": rec.content[:200]}
            row["retry_after_s"] = None
        except LLMError as e:
            g and mg.on_error(g, e)
            row["record"] = {k: v for k, v in asdict(e.record).items() if k != "content"}
            row["error_kind"], row["retry_after_s"] = e.kind, e.retry_after_s
        finally:
            self.in_flight -= 1
        self.f.write(json.dumps(row) + "\n"); self.f.flush()
        s = row["record"].get("status")
        print(f"{self.name} {model.split('/')[-1]} st={s} {row.get('error_kind', '')} ra={row.get('retry_after_s')}")
        return row

    async def run(self):
        p = PROBES[self.name]
        n = self.name
        if n in ("latency", "latency-small", "cache", "effort"):
            prefix = make_prompt(1100, 99) + "\n\n" if n == "cache" else ""
            for i, size in enumerate(p.sizes):
                if n == "cache":
                    pre = prefix if i < 6 else prefix[: len(prefix) // 3]
                    prompt = pre + f"Question {i}: name one term used above."
                    await self.call(p.model, prompt, p.max_out, p.effort, extra={"cache_group": 0 if i < 6 else 1})
                elif n == "effort":
                    eff = "low" if i % 2 == 0 else "medium"
                    await self.call(p.model, make_prompt(size, i // 2), p.max_out, eff, extra={"effort_arm": eff})
                else:
                    await self.call(p.model, make_prompt(size, i), p.max_out, p.effort)
        elif n == "concurrency":
            for width in p.sizes:
                await asyncio.gather(*(self.call(p.model, make_prompt(200, j), p.max_out, p.effort, extra={"burst": width})
                                       for j in range(width)))
                await asyncio.sleep(30)
        elif n == "rpm-edge":
            first = None
            for i in range(40):
                r = await self.call(p.model, "Say ok.", 1, None, paced=False, extra={"i": i})
                if r["record"].get("status") == 429:
                    first = r; break
            if first:
                await asyncio.sleep((first.get("retry_after_s") or 2) + 0.2)
                rs = await asyncio.gather(*(self.call(p.model, "Say ok.", 1, None, paced=False, extra={"refill_test": True}) for _ in range(5)))
                ok = sum(r["record"].get("status") == 200 for r in rs)
                print("refill verdict:", "window reset" if ok >= 4 else "continuous refill" if ok <= 2 else "unclear", ok)
        elif n == "tpm-edge":
            for i, size in enumerate(p.sizes):
                r = await self.call(p.model, make_prompt(size, 50 + i), p.max_out, None, paced=False, extra={"i": i})
                if r["record"].get("status") == 429:
                    break
        elif n == "rpd-burn":
            while True:
                r = await self.call(MQ, "Say ok.", 1, None, paced=False)
                if r.get("error_kind") == "rate_limited_day":
                    break
                await asyncio.sleep(2.1 if r["record"].get("status") == 200 else (r.get("retry_after_s") or 5))
            t_end = time.monotonic() + 3600
            while time.monotonic() < t_end:
                await asyncio.sleep(600)
                await self.call(MQ, "Say ok.", 1, None, paced=False, extra={"after_exhaustion": True})
        self.f.close(); await self.client.aclose()


def summarize() -> None:
    rows = [json.loads(l) for f in sorted(OUT.glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        print("no probe data"); return
    by = {}
    for r in rows:
        rec = r["record"]
        if rec.get("status") == 200 and rec.get("t_done"):
            by.setdefault((r["probe"], r["model"]), []).append(r)
    for (probe, m), rs in sorted(by.items()):
        recs = [r["record"] for r in rs]
        fc = [(x["t_first_content"] - x["t_send"]) * 1000 for x in recs if x.get("t_first_content")]
        tot = [(x["t_done"] - x["t_send"]) * 1000 for x in recs]
        rt = [((x.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for x in recs]
        ct = [(x.get("usage") or {}).get("completion_tokens") or 0 for x in recs]
        qt = [(x.get("usage") or {}).get("queue_time") or 0 for x in recs]
        pc = lambda a: [int(np.percentile(a, q)) for q in (50, 95, 99)] if a else None
        print(f"{probe:14} {m:22} n={len(rs):3} first_content p50/95/99={pc(fc)} total={pc(tot)} "
              f"reasoning_share={sum(rt) / max(1, sum(ct)):.2f} queue_time_p50={np.median(qt) * 1000:.0f}ms")
    for r in rows:
        if r["record"].get("status") == 429:
            print(f"429 [{r['probe']} {r['model']} i={r.get('i')}] retry-after={r.get('retry_after_s')} body={r['record'].get('error_body', '')[:300]}")
            print("   headers:", {k: v for k, v in (r['record'].get('headers') or {}).items() if k.startswith('x-ratelimit') or k == 'retry-after'})
    cache = [r for r in rows if r["probe"] == "cache" and r["record"].get("usage")]
    for r in cache:
        u = r["record"]["usage"]
        print(f"cache g{r.get('cache_group')} prompt={u.get('prompt_tokens')} cached={(u.get('prompt_tokens_details') or {}).get('cached_tokens')} "
              f"remaining_tokens={(r['record'].get('headers') or {}).get('x-ratelimit-remaining-tokens')}")


def budget_cli(argv=None):
    cfg = load_config()
    used = ledger_usage(LEDGER) if LEDGER.exists() else {}
    L = cfg.limits_prior
    for m in (cfg.models.primary.id, cfg.models.fallback.id, cfg.models.sacrificial.id):
        rq, tk = used.get(m, (0, 0))
        print(f"{m:24} last 24h: {rq:5} requests ({rq / L.rpd:5.1%} of RPD), {tk:7} tokens ({tk / L.tpd:5.1%} of TPD); left ≈ {L.tpd - tk} tokens")


def guard(name: str, real: bool) -> None:
    if not real:
        return
    p = PROBES[name]
    used = ledger_usage(LEDGER) if LEDGER.exists() else {}
    left = load_config().limits_prior.tpd - used.get(p.model, (0, 0))[1]
    print(f"probe {name}: estimated {p.est_tokens} tokens on {p.model}; {left} left today")
    if p.est_tokens > 0.8 * left:
        raise SystemExit("aborting: estimate exceeds 80% of what is left today")


def cli(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--real", action="store_true")
    a = ap.parse_args(argv)
    if a.name == "summarize":
        return summarize()
    if a.name == "models":
        import httpx
        from dotenv import load_dotenv
        load_dotenv()
        r = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        print([m["id"] for m in r.json().get("data", [])]); return
    if a.name not in PROBES:
        raise SystemExit(f"unknown probe; choose from {list(PROBES)} or summarize/models")
    guard(a.name, a.real)
    asyncio.run(Runner(a.name, a.real).run())
