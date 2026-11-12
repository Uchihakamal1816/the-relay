from __future__ import annotations
import asyncio, json, math, re, time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable
from relay.types import Limits


class Bucket:
    def __init__(self, capacity: float, refill_per_s: float, now: float):
        self.capacity, self.refill, self.level, self.t = capacity, refill_per_s, capacity, now

    def _upd(self, now):
        if now > self.t:
            self.level = min(self.capacity, self.level + (now - self.t) * self.refill)
            self.t = now

    def available_at(self, amount, now):
        self._upd(now)
        if amount > self.capacity:
            return math.inf
        if self.level >= amount:
            return now
        return now + (amount - self.level) / self.refill if self.refill > 0 else math.inf

    def take(self, amount, now):
        self._upd(now); self.level -= amount

    def refund(self, amount, now):
        self._upd(now); self.level = min(self.capacity, self.level + amount)

    def resize(self, capacity, now):
        self._upd(now)
        self.refill = self.refill * capacity / self.capacity if self.capacity else self.refill
        self.capacity = capacity; self.level = min(self.level, capacity)

    def clamp(self, level, now):
        self._upd(now); self.level = min(self.level, level)

    def frac_left(self, now):
        self._upd(now); return max(0.0, self.level / self.capacity) if self.capacity else 0.0


@dataclass
class Grant:
    model: str
    tokens: int
    t: float


class WouldMissDeadline(Exception):
    def __init__(self, wait_s): super().__init__(f"wait {wait_s:.2f}s"); self.wait_s = wait_s


class DailyExhausted(Exception):
    def __init__(self, reset_in_s): super().__init__(f"reset in {reset_in_s:.0f}s"); self.reset_in_s = reset_in_s


class ModelGovernor:
    def __init__(self, model: str, limits: Limits, clock: Callable[[], float] = time.monotonic, safety: float = 0.9):
        self.model, self.clock, self.safety = model, clock, safety
        n = clock()
        self.rpm = Bucket(limits.rpm * safety, limits.rpm * safety / 60, n)
        self.tpm = Bucket(limits.tpm * safety, limits.tpm * safety / 60, n)
        self.rpd = Bucket(limits.rpd * safety, limits.rpd * safety / 86400, n)
        self.tpd = Bucket(limits.tpd * safety, limits.tpd * safety / 86400, n)
        self.blocked_until = 0.0
        self.day_reset_at: float | None = None
        self.queue_len = 0
        self.peak_queue = 0
        self.queued_tokens = 0   # reservations of requests waiting ahead in the FIFO
        self._lock = asyncio.Lock()

    def _daily_short(self, tokens, now) -> float | None:
        if self.day_reset_at and now < self.day_reset_at:
            return self.day_reset_at - now
        for b, amt in ((self.rpd, 1), (self.tpd, tokens)):
            if b.available_at(amt, now) > now:
                return (amt - b.level) / b.refill if b.refill else math.inf
        return None

    def wait_s(self, tokens: int) -> float:
        now = self.clock()
        if self._daily_short(tokens, now) is not None:
            return math.inf
        ahead_r, ahead_t = self.queue_len, self.queued_tokens   # FIFO: everyone ahead is served first
        t = max(self.rpm.available_at(min(self.rpm.capacity, 1 + ahead_r), now),
                self.tpm.available_at(min(self.tpm.capacity, tokens + ahead_t), now), self.blocked_until)
        if tokens + ahead_t > self.tpm.capacity:  # more than one bucket's worth queued: add whole refill periods
            t = max(t, now + (tokens + ahead_t - self.tpm.level) / self.tpm.refill)
        return max(0.0, t - now)

    def daily_frac_left(self) -> float:
        now = self.clock()
        if self.day_reset_at and now < self.day_reset_at:
            return 0.0
        return min(self.rpd.frac_left(now), self.tpd.frac_left(now))

    def day_reset_in_s(self) -> float:
        now = self.clock()
        r = self._daily_short(1, now)
        return r if r is not None else 0.0

    async def acquire(self, tokens: int, deadline: float) -> Grant:
        now = self.clock()
        short = self._daily_short(tokens, now)
        if short is not None:
            raise DailyExhausted(short)
        w = self.wait_s(tokens)
        if now + w > deadline:
            raise WouldMissDeadline(w)  # refuse at the door, before joining the queue
        self.queue_len += 1; self.queued_tokens += tokens
        self.peak_queue = max(self.peak_queue, self.queue_len)
        queued = True
        try:
            timeout = None if deadline == math.inf else max(0.0, deadline - self.clock())
            try:
                await asyncio.wait_for(self._lock.acquire(), timeout)
            except asyncio.TimeoutError:
                raise WouldMissDeadline(deadline - now)
            try:
                self.queue_len -= 1; self.queued_tokens -= tokens; queued = False
                while True:
                    now = self.clock()
                    short = self._daily_short(tokens, now)
                    if short is not None:
                        raise DailyExhausted(short)
                    w = self.wait_s(tokens)
                    if now + w > deadline:
                        raise WouldMissDeadline(w)
                    if w <= 0:
                        break
                    await asyncio.sleep(w)
                for b, a in ((self.rpm, 1), (self.rpd, 1), (self.tpm, tokens), (self.tpd, tokens)):
                    b.take(a, now)
                return Grant(self.model, tokens, now)
            finally:
                self._lock.release()
        finally:
            if queued:
                self.queue_len -= 1; self.queued_tokens -= tokens

    def _refund_tokens(self, n):
        now = self.clock()
        if n > 0:
            self.tpm.refund(n, now); self.tpd.refund(n, now)

    def on_response(self, g: Grant, rec) -> None:
        now, h = self.clock(), rec.headers or {}
        try:
            if "x-ratelimit-limit-tokens" in h:
                cap = float(h["x-ratelimit-limit-tokens"]) * self.safety
                if abs(cap - self.tpm.capacity) > 1:
                    self.tpm.resize(cap, now)
            if "x-ratelimit-limit-requests" in h:
                cap = float(h["x-ratelimit-limit-requests"]) * self.safety
                if abs(cap - self.rpd.capacity) > 1:
                    self.rpd.resize(cap, now)
        except ValueError:
            pass
        u = rec.usage or {}
        actual = (u.get("total_tokens") or 0) - ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        self._refund_tokens(g.tokens - actual)
        try:
            if "x-ratelimit-remaining-tokens" in h:
                self.tpm.clamp(float(h["x-ratelimit-remaining-tokens"]), now)
            if "x-ratelimit-remaining-requests" in h:
                self.rpd.clamp(float(h["x-ratelimit-remaining-requests"]), now)
        except ValueError:
            pass

    def on_error(self, g: Grant, err) -> None:
        now = self.clock()
        body = (err.record.error_body or "").upper()
        if err.kind == "rate_limited_minute":
            self.blocked_until = max(self.blocked_until, now + (err.retry_after_s or 1.0))
            if "(TPM)" in body:
                self.tpm.clamp(0, now)
            if "(RPM)" in body:
                self.rpm.clamp(0, now)
            self._refund_tokens(g.tokens)
        elif err.kind == "rate_limited_day":
            self.rpd.clamp(0, now); self.tpd.clamp(0, now)
            self.day_reset_at = now + (err.retry_after_s or 3600.0)
        elif err.kind in ("server", "bad_request"):
            self._refund_tokens(g.tokens)

    def release(self, g: Grant) -> None:
        now = self.clock()
        self._refund_tokens(g.tokens); self.rpm.refund(1, now); self.rpd.refund(1, now)


class Governor:
    def __init__(self, limits: dict[str, Limits], clock=time.monotonic, safety: float = 0.9):
        self.models = {m: ModelGovernor(m, l, clock, safety) for m, l in limits.items()}

    def for_model(self, model_id: str) -> ModelGovernor:
        return self.models[model_id]

    @classmethod
    def from_config(cls, cfg, limits: dict[str, Limits] | None = None, ledger: Path | None = None, clock=time.monotonic):
        ids = [cfg.models.primary.id, cfg.models.fallback.id, cfg.models.sacrificial.id]
        g = cls(limits or {m: cfg.limits_prior for m in ids}, clock, cfg.governor.safety)
        if ledger and ledger.exists():
            used = ledger_usage(ledger)
            for m, (req, tok) in used.items():
                if m in g.models:
                    mg, now = g.models[m], clock()
                    mg.rpd.take(req, now); mg.tpd.take(tok, now)
        return g


def ledger_usage(ledger: Path, hours: float = 24) -> dict[str, tuple[int, int]]:
    cut = datetime.now(timezone.utc) - timedelta(hours=hours)
    out: dict[str, list[int]] = {}
    for line in ledger.read_text().splitlines():
        try:
            r = json.loads(line)
            if datetime.fromisoformat(r["wall"]) < cut:
                continue
        except (ValueError, KeyError):
            continue
        u = r.get("usage") or {}
        tok = (u.get("total_tokens") or 0) - ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        o = out.setdefault(r["model"], [0, 0]); o[0] += 1; o[1] += tok
    return {m: (a, b) for m, (a, b) in out.items()}
