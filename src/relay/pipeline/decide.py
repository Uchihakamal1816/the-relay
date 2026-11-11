from __future__ import annotations
from relay.types import Level


def choose_level(wait_s, daily_frac, tokens_full: int, tokens_lean: int, patience_left_s: float, cfg,
                 exclude: tuple = ()) -> tuple[Level, str | None]:
    """wait_s(model, tokens)->s; daily_frac(model)->0..1. Returns (level, model id)."""
    L = cfg.ladder
    p, f = cfg.models.primary.id, cfg.models.fallback.id
    p_ok = p not in exclude and daily_frac(p) > L.daily_stop_below
    f_ok = f not in exclude and daily_frac(f) > L.daily_stop_below
    if p_ok and daily_frac(p) > L.daily_lean_below and wait_s(p, tokens_full) <= L.wait_full_s:
        return Level.FULL, p
    if p_ok and wait_s(p, tokens_lean) <= L.wait_lean_s:
        return Level.LEAN, p
    if f_ok and wait_s(f, tokens_lean) <= L.wait_lean_s:
        return Level.FALLBACK, f
    cands = [(wait_s(m, tokens_lean), lvl, m) for ok, lvl, m in ((p_ok, Level.LEAN, p), (f_ok, Level.FALLBACK, f)) if ok]
    if cands:
        w, lvl, m = min(cands)
        if w <= patience_left_s - 1.5:
            return lvl, m
    return Level.EXTRACTIVE, None
