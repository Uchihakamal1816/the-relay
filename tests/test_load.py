import asyncio, time
from relay.load.schedule import build_schedule


def test_schedule_deterministic_and_rate():
    s = {"seed": 3, "segments": [{"duration_s": 3600, "rate_x_capacity": 1.0}]}
    a, b = build_schedule(s, ["q1", "q2"], 60), build_schedule(s, ["q1", "q2"], 60)
    assert a == b and abs(len(a) - 3600) < 180


def test_segments_and_duplicates():
    s = {"seed": 1, "segments": [{"duration_s": 100, "rate_x_capacity": 0.0}, {"duration_s": 600, "rate_x_capacity": 1.0, "dup_fraction": 0.5}]}
    a = build_schedule(s, [f"q{i}" for i in range(20)], 60)
    assert all(x.t_s >= 100 for x in a)
    share = sum(x.variant == 0 for x in a) / len(a)
    assert 0.4 < share < 0.6 and len({x.qid for x in a if x.variant == 0}) == 1


def test_open_loop():
    sched = build_schedule({"seed": 1, "segments": [{"duration_s": 2, "rate_x_capacity": 1.0}]}, ["q"], 600)
    starts = []
    async def slow(a, t0):
        await asyncio.sleep(max(0, t0 + a.t_s - time.monotonic())); starts.append(time.monotonic() - t0 - a.t_s); await asyncio.sleep(5)
    async def main():
        t0 = time.monotonic()
        tasks = [asyncio.ensure_future(slow(a, t0)) for a in sched]
        await asyncio.sleep(2.2)
        for t in tasks: t.cancel()
    asyncio.run(main())
    assert len(starts) >= 10 and max(starts) < 0.05
