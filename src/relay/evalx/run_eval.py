from __future__ import annotations
import argparse, asyncio, json
from datetime import datetime, timezone
from pathlib import Path
from relay.config import load_config
from relay.results_csv import ResultsWriter
from relay.types import Level
from .questions import load_questions, validate_questions
from .grader import grade


def reply_row(rep, scenario, source, start_iso, g):
    return {"request_id": rep.request_id, "scenario": scenario, "source": source, "start_time": start_iso,
            "first_token_ms": rep.first_token_ms, "total_ms": rep.total_ms, "status": rep.status, "retries": rep.retries,
            "tokens_in": rep.tokens_in, "tokens_out": rep.tokens_out, "retrieval_setting": rep.retrieval_setting, "answer_grade": g or ""}


def apply_review(rows, review=Path("eval/grades_review.csv")):
    import csv
    if not review.exists():
        return rows
    over = {(r["scenario"], r["qid"]): r["final_grade"] for r in csv.DictReader(review.open()) if r.get("final_grade")}
    for r in rows:
        qid = r["request_id"].split(":")[-1]
        if (r["scenario"], qid) in over:
            r["answer_grade"] = over[(r["scenario"], qid)]
    return rows


async def run(setting_name: str, model: str, real: bool, limit: int | None):
    from relay.pipeline.orchestrator import build
    from relay.llm.client import LEDGER
    cfg = load_config(overrides={"cache.enabled": False})
    qs = load_questions()[:limit] if limit else load_questions()
    setting = cfg.retrieval.settings[setting_name]
    scenario = f"eval-{setting_name}" + ("@20b" if model == "fallback" else "")
    source = "real" if real else "stand-in"
    w = ResultsWriter()
    done = {r["request_id"] for r in w.read() if r["scenario"] == scenario and r["source"] == source and r["status"] != "rate_limited"}
    relay = build(cfg, real, ledger=LEDGER if real else None, record_dir=Path("raw/recordings") if real else None)
    chunks = relay.search.chunks_by_id(setting.chunk_tokens)
    rows = [r for r in w.read() if r["scenario"] == scenario and r["status"] != "rate_limited"]
    gap = 60 * cfg.capacity.tokens_per_answer_full / cfg.limits_prior.tpm if real else 0  # stay under TPM: eval is not a load test
    review = Path("eval/grades_review.csv")
    for q in qs:
        rid = f"{scenario}:{source}:{q.id}"
        if rid in done:
            continue
        await asyncio.sleep(gap)
        start = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        rep = await relay.ask(q.question, request_id=rid, setting=setting,
                              force_level=Level.FALLBACK if model == "fallback" else Level.FULL,
                              trace_tags={"qid": q.id, "scenario": scenario})
        g = grade(q, rep, chunks)
        rows.append(reply_row(rep, scenario, source, start, g))
        w.replace_scenario(scenario, rows)
        new = not review.exists()
        with review.open("a") as f:
            if new:
                f.write("scenario,qid,auto_grade,final_grade,note\n")
            f.write(f"{scenario},{q.id},{g or ''},,\"{rep.text[:200].replace(chr(34), chr(39)).replace(chr(10), ' / ')}\"\n")
        print(f"{q.id} {rep.status} {g} {rep.first_token_ms}ms tok={rep.tokens_in}+{rep.tokens_out}")
    await relay.client.aclose()


def cli(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "recall", "run"])
    ap.add_argument("--setting", default="S3"); ap.add_argument("--model", default="primary")
    ap.add_argument("--real", action="store_true"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        from relay.ingest.parse_html import load_blocks
        probs = validate_questions(load_questions(), load_blocks())
        print("\n".join(probs) or "no problems")
    elif a.cmd == "recall":
        from relay.retrieve.tool import PaperSearch
        from relay.retrieve.papermap import PaperMap
        from .recall import recall_table, naive_baseline, score_split
        ps = PaperSearch(); ps.warmup(); qs = load_questions()
        pm = PaperMap.load(Path("data/corpus/papermap.json"))
        df = recall_table(qs, ps, pm)
        Path("report").mkdir(exist_ok=True)
        df.to_csv("report/recall_sweep.csv", index=False); print(df.to_string(index=False))
        nb = naive_baseline(qs, ps.indexes[400])
        Path("report/naive_baseline.json").write_text(json.dumps(nb, indent=1))
        print("N0 recall:", sum(x["hit"] for x in nb) / max(1, len(nb)))
        print("top-score split:", score_split(qs, ps, load_config().retrieval.settings["S3"]))
    else:
        asyncio.run(run(a.setting, a.model, a.real, a.limit))
