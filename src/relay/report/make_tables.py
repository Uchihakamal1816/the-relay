from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from relay.config import load_config

NM = "not measured"


def _load(path="results.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"answer_grade": str}, keep_default_na=False)
    for c in ("first_token_ms", "total_ms", "retries", "tokens_in", "tokens_out"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["t"] = pd.to_datetime(df["start_time"], utc=True, format="ISO8601")
    return df


def _pct(s: pd.Series, qs=(0.5, 0.95, 0.99)) -> str:
    s = s.dropna()
    if s.empty:
        return NM
    return " / ".join(str(int(round(s.quantile(q)))) for q in qs) + f" (n={len(s)})"


def peak_in_system(g: pd.DataFrame) -> int:
    ev = []
    for t, d in zip(g["t"], g["total_ms"].fillna(0)):
        ev += [(t, 1), (t + pd.Timedelta(milliseconds=float(d)), -1)]
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, x in ev:
        cur += x; best = max(best, cur)
    return best


def _p(x) -> str:
    return f"{100 * x:.1f}"


def table_2a(df) -> pd.DataFrame:
    rows = []
    for sc, g in df.groupby("scenario", sort=False):
        if sc.startswith("eval-"):
            continue
        ok = g[g.status == "ok"]
        dur = (g.t.max() - g.t.min()).total_seconds() or 1
        rows.append({"Scenario": sc, "Real or stand-in": ", ".join(sorted(g.source.unique())), "Requests sent": len(g),
                     "Duration (s)": int(round(dur)), "Offered rate (per min)": f"{len(g) / dur * 60:.1f}",
                     "First token p50 / p95 / p99 (ms)": _pct(ok.first_token_ms), "Total time p50 / p95 / p99 (ms)": _pct(ok.total_ms)})
    return pd.DataFrame(rows)


def cost(g, cfg) -> float:
    p, f = cfg.models.primary, cfg.models.fallback
    fb = g.retrieval_setting.str.contains("@20b")
    usd = ((g.tokens_in * np.where(fb, f.price_in, p.price_in)) + (g.tokens_out * np.where(fb, f.price_out, p.price_out))).sum() / 1e6
    n_ok = (g.status == "ok").sum()
    return 1000 * usd / n_ok if n_ok else float("nan")


def table_2b(df, cfg) -> pd.DataFrame:
    rows = []
    for sc, g in df.groupby("scenario", sort=False):
        if sc.startswith("eval-"):
            continue
        n_ok = (g.status == "ok").sum()
        c = cost(g, cfg)
        rows.append({"Scenario": sc, "Succeeded %": _p((g.status == "ok").mean()), "Rate-limited %": _p((g.status == "rate_limited").mean()),
                     "Timed out or given up %": _p((g.status == "timeout").mean()), "Retries per request": f"{g.retries.mean():.2f}",
                     "Peak queue length": peak_in_system(g),
                     "Tokens per answer": int(round((g.tokens_in + g.tokens_out).sum() / n_ok)) if n_ok else NM,
                     "Cost per 1,000 answers": f"${c:.3f}" if c == c else NM,
                     "_error %": _p((g.status == "error").mean())})
    return pd.DataFrame(rows)


def table_3(df) -> pd.DataFrame:
    rows = []
    for sc, g in df[df.scenario.str.startswith("eval-")].groupby("scenario", sort=False):
        n = len(g); ok = g[g.status == "ok"]
        gr = g.answer_grade
        n_ok = (g.status == "ok").sum()
        rows.append({"Setting": g.retrieval_setting.iloc[0] if n else sc, "Questions asked": n,
                     "Correct and cited %": _p((gr == "correct").mean()), "Wrong %": _p((gr == "wrong").mean()),
                     "Correctly refused %": _p((gr == "refused_correctly").mean()),
                     "Tokens per answer": int(round((g.tokens_in + g.tokens_out).sum() / n_ok)) if n_ok else NM,
                     "First token p50 / p95 (ms)": _pct(ok.first_token_ms, (0.5, 0.95)), "Total time p50 / p95 (ms)": _pct(ok.total_ms, (0.5, 0.95)),
                     "_wrongly refused %": _p((gr == "refused_wrongly").mean())})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no rows yet_\n"
    cols = [c for c in df.columns if not c.startswith("_")]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    notes = [c for c in df.columns if c.startswith("_")]
    for c in notes:
        out += f"\n{c[1:]} by row: " + ", ".join(f"{r.iloc[0]}={r[c]}" for _, r in df.iterrows()) + "\n"
    return out


def cli(argv):
    cfg = load_config()
    df = _load()
    txt = ("## Table 2a: Latency\n\n" + _md(table_2a(df)) + "\n## Table 2b: Outcomes\n\n" + _md(table_2b(df, cfg)) +
           "\n## Table 3: Retrieval settings\n\n" + _md(table_3(df)) +
           "\nDefinitions: percentiles are pandas linear quantiles over status=ok rows (n shown). Tokens per answer = all tokens of all rows / ok rows. "
           "Cost uses list prices per row's model (upper bound: cached input is billed at half). Peak queue = most requests in the system at once.\n")
    Path("report").mkdir(exist_ok=True); Path("report/tables.md").write_text(txt); print(txt)
