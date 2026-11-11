from __future__ import annotations
import random
from dataclasses import dataclass


@dataclass
class Arrival:
    t_s: float
    qid: str
    variant: int  # 0 = shared hot question (duplicate), else unique


def build_schedule(scn: dict, qids: list[str], capacity_per_min: float) -> list[Arrival]:
    rng = random.Random(scn.get("seed", 7))
    order = list(qids); rng.shuffle(order)
    hot = order[0]
    out, t0, n = [], 0.0, 0
    for seg in scn["segments"]:
        rate = seg["rate_x_capacity"] * capacity_per_min / 60.0  # per second
        dup = seg.get("dup_fraction", 0.0)
        t = t0
        while rate > 0:
            t += rng.expovariate(rate)
            if t >= t0 + seg["duration_s"]:
                break
            n += 1
            if dup and rng.random() < dup:
                out.append(Arrival(round(t, 4), hot, 0))
            else:
                out.append(Arrival(round(t, 4), rng.choice(order), n))
        t0 += seg["duration_s"]
    return out
