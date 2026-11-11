import pandas as pd
from relay.report.make_tables import _load, table_2a, table_2b, table_3, peak_in_system
from relay.config import load_config

CSV = """request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
a1,s,stand-in,2026-01-01T00:00:00.000Z,100,1000,ok,0,1000,100,S3,
a2,s,stand-in,2026-01-01T00:00:00.500Z,200,1000,ok,1,1000,100,S4@20b,
a3,s,stand-in,2026-01-01T00:00:10.000Z,,6000,timeout,0,0,0,,
a4,s,stand-in,2026-01-01T00:01:00.000Z,50,60,rate_limited,0,0,0,extractive,
e1,eval-S3,real,2026-01-01T00:00:00.000Z,300,900,ok,0,1500,200,S3,correct
e2,eval-S3,real,2026-01-01T00:00:05.000Z,400,800,ok,0,1500,200,S3,wrong
e3,eval-S3,real,2026-01-01T00:00:09.000Z,350,700,ok,0,1500,100,S3,refused_correctly
e4,eval-S3,real,2026-01-01T00:00:12.000Z,,,error,2,800,0,S3,
"""


def test_tables(tmp_path):
    p = tmp_path / "r.csv"; p.write_text(CSV)
    df = _load(p); cfg = load_config()
    a = table_2a(df).iloc[0]
    assert a["Requests sent"] == 4 and a["Duration (s)"] == 60 and a["First token p50 / p95 / p99 (ms)"].startswith("150 / ")
    b = table_2b(df, cfg).iloc[0]
    assert b["Succeeded %"] == "50.0" and b["Timed out or given up %"] == "25.0" and b["Tokens per answer"] == 1100
    assert b["Peak queue length"] == 2
    exp = 1000 * ((1000 * 0.15 + 100 * 0.60) + (1000 * 0.075 + 100 * 0.30)) / 1e6 / 2
    assert b["Cost per 1,000 answers"] == f"${exp:.3f}"
    t = table_3(df).iloc[0]
    assert t["Questions asked"] == 4 and t["Correct and cited %"] == "25.0" and t["Tokens per answer"] == round((1700 + 1700 + 1600 + 800) / 3)
