from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np


def _rows(probe_dir: Path):
    for f in sorted(probe_dir.glob("*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                yield json.loads(l)


def fit_samples(prompt_tokens, ttft_ms, tps, slow_mult=3.0) -> dict:
    x, y = np.asarray(prompt_tokens, float), np.asarray(ttft_ms, float)
    A = np.vstack([np.ones_like(x), x / 1000]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = max(a, 1.0), max(b, 0.0)
    resid = np.log(np.clip(y, 1, None) / (a + b * x / 1000))
    med = np.median(y)
    slow = y > slow_mult * med
    lt = np.log(np.asarray(tps, float))
    return {"ttft_base_ms": float(a), "ttft_per_1k_prompt_ms": float(b), "ttft_sigma": float(np.std(resid[~slow]) if (~slow).sum() > 1 else 0.3),
            "tps_mu": float(lt.mean()), "tps_sigma": float(lt.std()), "p_slow": float(slow.mean()),
            "slow_factor": float(np.median(y[slow] / med)) if slow.any() else 1.0}


def fit_profile(probe_dir: Path = Path("raw/probe")) -> dict:
    by_model: dict[str, list] = {}
    load_pts = []
    err429 = None
    for r in _rows(probe_dir):
        rec = r.get("record") or {}
        if rec.get("status") == 429 and err429 is None and rec.get("error_body"):
            try:
                err429 = json.loads(rec["error_body"])
            except json.JSONDecodeError:
                pass
        if rec.get("status") != 200 or not rec.get("t_done"):
            continue
        u = rec.get("usage") or {}
        first = min(t for t in (rec.get("t_first_reasoning"), rec.get("t_first_content")) if t) if (rec.get("t_first_reasoning") or rec.get("t_first_content")) else None
        if not first:
            continue
        ttft = (first - rec["t_send"]) * 1000
        gen_s = max(1e-3, rec["t_done"] - first)
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        if r.get("in_flight", 1) == 1:  # baseline from unloaded calls only; load slope is fitted separately
            by_model.setdefault(r["model"], []).append((u.get("prompt_tokens", 0), ttft, max(1, u.get("completion_tokens", 1)) / gen_s, rt))
        if r.get("in_flight"):
            load_pts.append((r["in_flight"], ttft))
    lat = {}
    for m, pts in by_model.items():
        if len(pts) < 5:
            continue
        p, t, s, rt = zip(*pts)
        lat[m] = {**fit_samples(p, t, s), "reasoning_tokens_mu": float(np.mean(rt)) or 1.0, "n": len(pts)}
    alpha = 0.0
    if load_pts:
        groups = {}
        for n, t in load_pts:
            groups.setdefault(n, []).append(t)
        ns = sorted(groups)
        meds = [np.median(groups[n]) for n in ns]
        if 1 in groups and len(ns) > 1:
            slope = np.polyfit(np.array(ns, float) - 1, np.array(meds) / np.median(groups[1]) - 1, 1)[0]
            alpha = float(max(0.0, slope))
    prof = {"latency": lat, "load": {"c0": 1, "alpha": alpha}}
    if err429 and "error" in err429:
        tpl = json.loads(json.dumps(err429)); tpl["error"]["message"] = "__MSG__"; prof["error_429"] = tpl
    return prof


def cli(argv):
    prof = fit_profile()
    Path("config/standin_profile.json").write_text(json.dumps(prof, indent=1))
    print(json.dumps(prof, indent=1))
    rng = np.random.default_rng(7)
    for m, L in prof["latency"].items():
        sim = (L["ttft_base_ms"] + L["ttft_per_1k_prompt_ms"] * 1.2) * rng.lognormal(0, L["ttft_sigma"], 2000)
        print(m, "refit sample ttft p50/p95/p99 @1.2k prompt:", [int(np.percentile(sim, q)) for q in (50, 95, 99)])
