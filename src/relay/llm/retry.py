from __future__ import annotations
import random, time
from collections import deque

NEVER = {"rate_limited_day", "bad_request", "midstream"}


class RetryPolicy:
    def __init__(self, cfg, rng: random.Random | None = None, clock=time.monotonic):
        self.cfg, self.rng, self.clock = cfg, rng or random.Random(7), clock
        self.reqs: deque[float] = deque()
        self.retries: deque[float] = deque()

    def _trim(self, now):
        for d in (self.reqs, self.retries):
            while d and d[0] < now - 60:
                d.popleft()

    def note_request(self) -> None:
        self.reqs.append(self.clock())

    def decide(self, err, attempt: int, sent_any: bool) -> float | None:
        c = self.cfg
        if c.policy == "naive":
            return 0.0 if attempt < 3 and not sent_any else None
        if sent_any or err.kind in NEVER or attempt >= c.max_attempts:
            return None
        now = self.clock(); self._trim(now)
        if len(self.retries) + 1 > c.budget_ratio * max(1, len(self.reqs)):
            return None
        self.retries.append(now)
        if err.kind == "rate_limited_minute":
            return err.retry_after_s if err.retry_after_s is not None else 1.0
        return self.rng.uniform(0, min(c.cap_s, c.base_s * 2 ** attempt))
