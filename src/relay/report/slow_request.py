from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from relay.trace import split


def pick(traces: list[dict], pct: float) -> dict:
    ok = [t for t in traces if t.get("notes", {}).get("status") == "ok"]
    tot = np.array([t["end_ms"] for t in ok])
    target = np.percentile(tot, pct)
    return ok[int(np.argmin(abs(tot - target)))]


def cli(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", default="overload"); ap.add_argument("--pct", type=float, default=99)
    a = ap.parse_args(argv)
    traces = [json.loads(l) for l in Path(f"raw/traces/{a.scenario}.jsonl").read_text().splitlines() if l.strip()]
    t = pick(traces, a.pct)
    print(f"request {t['request_id']}  total={t['end_ms']:.0f} ms  first line={t['first_ms']}")
    for s in t["spans"]:
        extra = {k: v for k, v in s.items() if k not in ("name", "kind", "start_ms", "end_ms")}
        print(f"  {s['start_ms']:>8.0f} → {s.get('end_ms', 0):>8.0f} ms  {s['kind']:6} {s['name']:16} {extra or ''}")
    print("split:", {k: (round(v) if v is not None else None) for k, v in split(t).items()})
    print("notes:", t.get("notes"))
