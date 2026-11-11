from __future__ import annotations
import time
from contextlib import contextmanager


class Trace:
    def __init__(self, request_id: str, t0: float | None = None):
        self.request_id = request_id
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.spans: list[dict] = []
        self.first_ms: float | None = None
        self.end_ms: float | None = None
        self.notes: dict = {}

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def add(self, name: str, kind: str, start_ms: float, end_ms: float, **meta) -> None:
        self.spans.append({"name": name, "kind": kind, "start_ms": round(start_ms, 1), "end_ms": round(end_ms, 1), **meta})

    @contextmanager
    def span(self, name: str, kind: str, **meta):
        s = self.now_ms()
        rec = {"name": name, "kind": kind, "start_ms": round(s, 1), **meta}
        self.spans.append(rec)
        try:
            yield rec
        finally:
            rec["end_ms"] = round(self.now_ms(), 1)

    def mark_first(self) -> None:
        if self.first_ms is None:
            self.first_ms = self.now_ms()

    def to_dict(self) -> dict:
        return {"request_id": self.request_id, "first_ms": self.first_ms, "end_ms": self.end_ms or self.now_ms(),
                "spans": self.spans, "notes": self.notes}


def split(t: dict) -> dict:
    out = {"wait_ms": 0.0, "work_ms": 0.0, "retry_ms": 0.0}
    for s in t["spans"]:
        if s.get("kind") in ("wait", "work", "retry") and "end_ms" in s:
            out[f"{s['kind']}_ms"] += s["end_ms"] - s["start_ms"]
    end = t.get("end_ms") or 0
    first = t.get("first_ms")
    out["before_first_ms"] = first
    out["after_first_ms"] = None if first is None else end - first
    return out
