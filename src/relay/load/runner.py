from __future__ import annotations
import argparse, asyncio, json, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import httpx, yaml
from relay.config import load_config
from relay.results_csv import ResultsWriter
from relay.types import Limits
from relay.evalx.questions import load_questions, Question
from relay.evalx.grader import grade
from relay.evalx.run_eval import reply_row
from .schedule import build_schedule

FALLBACK_POOL = [Question(f"p{i:02d}", q, True) for i, q in enumerate([
    "How many layers does the Transformer encoder have?", "What frame rate does the Mimi codec run at?",
    "What does the routing module in H-Net predict?", "What BLEU does the big Transformer reach on English-to-German?",
    "How many codebooks does Mimi use?", "What learning rate schedule does the Transformer use?",
    "What is the inner monologue in Moshi?", "How does H-Net compare to a BPE Transformer?"])]


def load_scenario(name: str) -> dict:
    return yaml.safe_load(Path(f"config/scenarios/{name}.yaml").read_text())


def scaled_limits(cfg, scn: dict) -> dict[str, Limits]:
    L, s = cfg.limits_prior, scn.get("limit_scale", 1)
    out = {}
    for m in (cfg.models.primary.id, cfg.models.fallback.id, cfg.models.sacrificial.id):
        out[m] = Limits(int(L.rpm * s), int(L.tpm * s), int(L.rpd * s), int(L.tpd * s))
    return out


def daily_override(cfg, scn, capacity_per_min, lim):
    d = scn.get("daily_after_s")
    if not d:
        return
    seg = scn["segments"][0]["rate_x_capacity"] * capacity_per_min  # answers/min offered
    per = {"primary": cfg.capacity.tokens_per_answer_full, "fallback": cfg.capacity.get("tokens_per_answer_lean", cfg.capacity.tokens_per_answer_full * 0.6)}
    for role, after in d.items():
        m = cfg.models[role].id
        lim[m].tpd = int(seg * after / 60 * per[role])


async def admin(base: str, body: dict):
    async with httpx.AsyncClient() as c:
        r = await c.post(base.replace("/openai/v1", "") + "/admin/config", json=body); r.raise_for_status()


async def run_scenario(name: str, real: bool, cfg_path="config/relay.yaml"):
    from relay.pipeline.orchestrator import build
    from relay.llm.client import LEDGER
    scn = load_scenario(name)
    if real != (scn.get("target") == "real"):
        raise SystemExit(f"scenario {name} targets {scn.get('target')}; pass --real only for real targets")
    cfg = load_config(cfg_path, overrides=scn.get("overrides") or {})
    lim = scaled_limits(cfg, scn)
    cap = scn.get("limit_scale", 1) * cfg.limits_prior.tpm / cfg.capacity.tokens_per_answer_full
    daily_override(cfg, scn, cap, lim)
    qs = load_questions() or FALLBACK_POOL
    qmap = {q.id: q for q in qs}
    sched = build_schedule(scn, list(qmap), cap)
    Path("raw/traces").mkdir(parents=True, exist_ok=True)
    Path(f"raw/traces/{name}.schedule.json").write_text(json.dumps([a.__dict__ for a in sched]))
    tf = Path(f"raw/traces/{name}.jsonl")
    tf.unlink(missing_ok=True)
    if not real:
        await admin(cfg.endpoints.standin, {"reset": True, "patch": {
            "limits": {m: l.__dict__ for m, l in lim.items()}, "faults": {**{"p_500": 0, "p_503": 0, "p_hang": 0, "p_midstream_cut": 0, "p_slow_success": 0},
                                                                         **(scn.get("standin") or {}).get("faults", {})}}})
    relay = build(cfg, real, limits=None if real else lim, ledger=LEDGER if real else None)
    source = "real" if real else "stand-in"
    chunks = relay.search.chunks_by_id(cfg.retrieval.settings[cfg.retrieval.default].chunk_tokens)
    rows = []
    wall0 = datetime.now(timezone.utc) + timedelta(seconds=1)
    mono0 = time.monotonic() + 1
    pf, pt = cfg.slo.patience_first_s, cfg.slo.patience_total_s

    async def one(i, a):
        arr = mono0 + a.t_s
        await asyncio.sleep(max(0, arr - time.monotonic()))
        q = qmap[a.qid]
        got_first = asyncio.Event()
        task = asyncio.ensure_future(relay.ask(q.question, request_id=f"{name}:{i:05d}", arrival_t=arr,
                                               cache_key=f"{a.qid}#{a.variant}", trace_tags={"qid": a.qid, "scenario": name},
                                               on_event=lambda e: e.kind in ("sentence", "notice") and got_first.set()))
        start_iso = (wall0 + timedelta(seconds=a.t_s)).isoformat(timespec="milliseconds")
        fw = asyncio.ensure_future(got_first.wait())
        try:
            # first line OR the request finishing (an exception must not look like a give-up)
            await asyncio.wait({fw, task}, timeout=max(0.01, arr + pf - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
            if not (fw.done() or task.done()):
                raise asyncio.TimeoutError
            rep = await asyncio.wait_for(asyncio.shield(task), max(0.01, arr + pt - time.monotonic()))
            rows.append(reply_row(rep, name, source, start_iso, grade(q, rep, chunks) if q.gold or not q.answerable else None))
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except BaseException:
                pass
            rows.append({"request_id": f"{name}:{i:05d}", "scenario": name, "source": source, "start_time": start_iso,
                         "first_token_ms": None, "total_ms": int((time.monotonic() - arr) * 1000), "status": "timeout",
                         "retries": 0, "tokens_in": 0, "tokens_out": 0, "retrieval_setting": "", "answer_grade": ""})
        except Exception as e:  # a pipeline bug: record it as an error row instead of losing the request
            print(f"request {i} raised {type(e).__name__}: {e}")
            rows.append({"request_id": f"{name}:{i:05d}", "scenario": name, "source": source, "start_time": start_iso,
                         "first_token_ms": None, "total_ms": int((time.monotonic() - arr) * 1000), "status": "error",
                         "retries": 0, "tokens_in": 0, "tokens_out": 0, "retrieval_setting": "", "answer_grade": ""})
        finally:
            fw.cancel()

    async def events():
        for ev in scn.get("events") or []:
            await asyncio.sleep(max(0, mono0 + ev["at_s"] - time.monotonic()))
            if "scale_limits" in ev and not real:
                for m, l in lim.items():
                    for k, f in ev["scale_limits"].items():
                        setattr(l, k, int(getattr(l, k) * f))
                await admin(cfg.endpoints.standin, {"limits": {m: l.__dict__ for m, l in lim.items()}})
                print(f"[{ev['at_s']}s] limits scaled {ev['scale_limits']}")

    print(f"{name}: {len(sched)} arrivals over {sum(s['duration_s'] for s in scn['segments'])}s; capacity {cap:.1f} answers/min")
    ev_task = asyncio.ensure_future(events())
    await asyncio.gather(*(one(i, a) for i, a in enumerate(sched)))
    ev_task.cancel()
    ResultsWriter().replace_scenario(name, rows)
    await relay.client.aclose()
    from collections import Counter
    print(name, Counter(r["status"] for r in rows), "peak governor queue:", {m.split('/')[-1]: g.peak_queue for m, g in relay.gov.models.items()})


def cli(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run"]); ap.add_argument("scenario"); ap.add_argument("--real", action="store_true")
    a = ap.parse_args(argv)
    asyncio.run(run_scenario(a.scenario, a.real))
