from __future__ import annotations
import asyncio, re, time


def norm_key(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


class Flight:
    """One in-progress answer shared by every identical question (single-flight)."""
    def __init__(self):
        self.subs: list = []      # objects with .push(kind, text)
        self.events: list = []    # (kind, text) so late joiners get a replay
        self.task: asyncio.Task | None = None

    def push(self, kind: str, text: str):
        self.events.append((kind, text))
        for s in list(self.subs):
            s.push(kind, text)

    def join(self, sub):
        for k, t in self.events:
            sub.push(k, t)
        self.subs.append(sub)

    def leave(self, sub):
        if sub in self.subs:
            self.subs.remove(sub)
        if not self.subs and self.task and not self.task.done():
            self.task.cancel()  # nobody is waiting any more: stop spending tokens


class AnswerCache:
    def __init__(self, ttl_s: float = 600, clock=time.monotonic):
        self.ttl, self.clock = ttl_s, clock
        self.store: dict[str, tuple[float, object]] = {}
        self.flights: dict[str, Flight] = {}

    def get(self, key):
        v = self.store.get(key)
        if v and self.clock() - v[0] < self.ttl:
            return v[1]
        return None

    def put(self, key, reply):
        self.store[key] = (self.clock(), reply)
