from __future__ import annotations
import csv, os, tempfile
from pathlib import Path

COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms", "status",
           "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade"]
_ALLOWED = {"source": {"real", "stand-in"}, "status": {"ok", "rate_limited", "timeout", "error"},
            "answer_grade": {"", "correct", "wrong", "refused_correctly", "refused_wrongly"}}


def _check(row: dict) -> dict:
    out = {c: ("" if row.get(c) is None else row.get(c)) for c in COLUMNS}
    for c, ok in _ALLOWED.items():
        if out[c] not in ok:
            raise ValueError(f"bad {c}: {out[c]!r}")
    return out


class ResultsWriter:
    def __init__(self, path: str | Path = "results.csv"):
        self.path = Path(path)

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return list(csv.DictReader(f))

    def replace_scenario(self, scenario: str, rows: list[dict]) -> None:
        new = [_check({**r, "scenario": scenario}) for r in rows]
        keep = [r for r in self.read() if r["scenario"] != scenario]
        self._write(keep + new)

    def append(self, rows: list[dict]) -> None:
        self._write(self.read() + [_check(r) for r in rows])

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, self.path)
