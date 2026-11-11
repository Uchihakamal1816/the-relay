from __future__ import annotations
import re, subprocess
from pathlib import Path
from relay.results_csv import COLUMNS

HEADINGS = ["What I built and why", "The real service and my stand-in", "Results", "Where the time goes", "When a limit is hit",
            "Retrieval: quality versus cost", "How this library will mislead a naive retriever", "Retries", "What the person sees",
            "What I'm unsure about, what I left out, what I learned", "Your feedback on Vaani", "How I worked with the assistant"]


def check(root: Path = Path(".")) -> list[str]:
    probs = []
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True).stdout.split()
    for f in files:
        p = root / f
        if p.is_file() and p.suffix not in (".pdf", ".npy") and "gsk_" in p.read_text(errors="ignore"):
            probs.append(f"possible API key in {f}")
    if ".env" in files:
        probs.append(".env is tracked by git")
    rc = root / "results.csv"
    if not rc.exists() or rc.read_text().splitlines()[0] != ",".join(COLUMNS):
        probs.append("results.csv missing or header wrong")
    rep = root / "report/report.md"
    if rep.exists():
        heads = [re.sub(r"^#+\s*(\d+\s*)?", "", l).strip() for l in rep.read_text().splitlines() if l.startswith("#")]
        found = [h for h in heads if h in HEADINGS]
        if found != HEADINGS:
            probs.append(f"report headings out of order or missing: {[h for h in HEADINGS if h not in found]}")
        tm = root / "report/tables.md"
        if tm.exists() and tm.read_text().strip() not in rep.read_text():
            probs.append("report/tables.md is not pasted verbatim into report.md (rerun relay tables, re-paste)")
    else:
        probs.append("report/report.md missing")
    for f in (root / "data/patches").glob("*.md"):
        if not f.read_text().lower().startswith("verified: true"):
            probs.append(f"unverified patch {f.name}")
    return probs


def cli(argv):
    p = check()
    print("\n".join(p) if p else "submit-check: all good")
    raise SystemExit(1 if p else 0)
