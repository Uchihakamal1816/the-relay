<!-- [KAMAL] Write the prose yourself, max 2 pages before the tables. Evidence to pull is listed under each heading; delete the lists when done. -->

# 1 What I built and why
- Capacity arithmetic: 8,000 TPM / 1,678 tokens per real answer (steady-real) = 4.8 full answers/min on 120b; lean 20b answer 948 tokens. Config uses 1,700.
- Pipeline diagram (PLAN.md D12). The one decision: tokens per answer is the capacity unit.

# 2 The real service and my stand-in
- Table 1 (below), from `relay probe summarize`.
- steady-real vs steady-twin: first token p50 1767 vs 1622 ms (stand-in 8% fast); tokens/answer 1678 vs 1374 (18% low).
- Least-supported term: load slowdown alpha=0.18, fitted from one 45-request concurrency probe.
- Stand-in is kinder: understand fallback uses the keyword router; no undocumented qwen ITPM 7000.

# 3 Results
- Tables 2a, 2b; n per row.
- Number nobody could guess: under overload the binding limit was my CPU (re-ranker 1,131 ms batched vs 326 ms at batch_size=1), not Groq; fixing it took overload from 448 to 994 ok of 1,165 (dry run).
- S2 (k6, no re-rank) best: 52.2% correct, 13.0% wrong, 2,066 tokens/answer. S3: 43.5% at 1,333. S1 only 12 questions (quota), all answerable, so not comparable.

# 4 Where the time goes
- p99 overload request 5,951 ms: understand 810, retrieve 33, wait in 120b queue 4,061, fall to S4/20b, model 964. Waiting is 68% of the time; deciding the level before queueing would remove it.

# 5 When a limit is hit
- minute-halved (limits halved at 240 s): 96.1% ok, 2.8% rate-limited. daily-exhausted: 90.8% ok, 9.2% rate-limited, 0 timeouts.
- Ladder: FULL (120b, S3) → LEAN (120b, S4) → FALLBACK (20b, S4) → EXTRACTIVE (quoted passages, no model).

# Retrieval: quality versus cost
- Recall sweep: k≥6 gives recall 1.0 at ~1,500 tokens; c400 k2 rr 0.74; paper map adds nothing to recall.
- 120b at k=2 refuses instead of guessing (0% wrong, 56.5% wrongly refused); 20b at k=2 guesses (21.7% wrong).
- Re-ranker top score does not separate unanswerable questions (3 of 4 inside the answerable range), so no score-based refusal.

# How this library will mislead a naive retriever
- Three questions from report/naive_baseline.json with the wrong passage each.

# 6 Retries
- Budgeted: 10% of trailing-minute requests, floor 3/min, honour retry-after. overload 74.1% ok vs naive 73.2%; retries 0.04/request in both, because the ladder degrades before retrying.

# 7 What the person sees
- The EXTRACTIVE_DAY reply exactly as sent (src/relay/pipeline/messages.py).

# 8 What I'm unsure about, what I left out, what I learned

# Your feedback on Vaani

# How I worked with the assistant

## Table 1: Limits you found in the real service

| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | 30 | 31st request in a minute returned 429 (rpm-edge probe) | high |
| Tokens per minute | 8,000 (qwen also 7,000 input tokens/min, undocumented) | tpm-edge probe; 429 body states the limit; admission counts input tokens only | high |
| Requests per day | 1,000 | `x-ratelimit-limit-requests` header; rejected 429s did not appear to count | medium |
| Tokens per day | 200,000 | console limits page; not hit | low |
| Requests in flight at once | no cap found up to 45 | concurrency probe on 20b: no 429, but queue_time p50 367 ms | medium |
| What it returns when a limit is hit | 429, JSON body naming the limit, `retry-after` seconds | rpm-edge, tpm-edge | high |
| Time until it recovers | continuous refill; `retry-after` 2-3 s; 1 of 5 calls succeeded right after it | rpm-edge retries | medium |

<!-- paste report/tables.md here verbatim after `relay tables` -->

## Table 2a: Latency

| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
|---|---|---|---|---|---|---|
| steady-real | real | 29 | 343 | 5.1 | 1767 / 2567 / 2665 (n=29) | 1817 / 2567 / 2665 (n=29) |
| steady-twin | stand-in | 29 | 343 | 5.1 | 1622 / 2288 / 2334 (n=29) | 1624 / 2348 / 2356 (n=29) |
| steady | stand-in | 394 | 597 | 39.6 | 1792 / 2480 / 4570 (n=394) | 1834 / 2501 / 4587 (n=394) |
| burst | stand-in | 552 | 598 | 55.3 | 1714 / 3571 / 4839 (n=549) | 1742 / 3572 / 4904 (n=549) |
| burst-nocache | stand-in | 552 | 598 | 55.3 | 1904 / 5201 / 5825 (n=503) | 1980 / 5308 / 5887 (n=503) |
| overload | stand-in | 1395 | 360 | 232.6 | 3771 / 5607 / 5921 (n=1034) | 3798 / 5607 / 5944 (n=1034) |
| overload-naive-retry | stand-in | 1395 | 360 | 232.6 | 3829 / 5599 / 5885 (n=1021) | 3843 / 5599 / 5885 (n=1021) |
| minute-halved | stand-in | 459 | 599 | 46.0 | 2023 / 4603 / 5411 (n=441) | 2099 / 4603 / 5411 (n=441) |
| daily-exhausted | stand-in | 359 | 479 | 45.0 | 2298 / 3556 / 4001 (n=326) | 2328 / 3557 / 4001 (n=326) |

## Table 2b: Outcomes

| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
|---|---|---|---|---|---|---|---|
| steady-real | 100.0 | 0.0 | 0.0 | 0.00 | 2 | 1678 | $0.334 |
| steady-twin | 100.0 | 0.0 | 0.0 | 0.00 | 3 | 1374 | $0.246 |
| steady | 100.0 | 0.0 | 0.0 | 0.01 | 8 | 1369 | $0.246 |
| burst | 99.5 | 0.2 | 0.4 | 0.01 | 14 | 996 | $0.176 |
| burst-nocache | 91.1 | 3.6 | 5.3 | 0.01 | 36 | 1198 | $0.195 |
| overload | 74.1 | 19.9 | 5.7 | 0.04 | 25 | 915 | $0.117 |
| overload-naive-retry | 73.2 | 22.1 | 4.7 | 0.04 | 25 | 921 | $0.118 |
| minute-halved | 96.1 | 2.8 | 1.1 | 0.00 | 10 | 1266 | $0.221 |
| daily-exhausted | 90.8 | 9.2 | 0.0 | 0.00 | 9 | 1090 | $0.151 |

error % by row: steady-real=0.0, steady-twin=0.0, steady=0.0, burst=0.0, burst-nocache=0.0, overload=0.3, overload-naive-retry=0.0, minute-halved=0.0, daily-exhausted=0.0

## Table 3: Retrieval settings

| Setting | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| S3:c200-k4-rr | 23 | 43.5 | 26.1 | 17.4 | 1333 | 1679 / 3879 (n=23) | 1682 / 3947 (n=23) |
| S4:c200-k2-rr@20b | 23 | 39.1 | 21.7 | 17.4 | 948 | 996 / 2095 (n=23) | 996 / 2095 (n=23) |
| S4:c200-k2-rr | 23 | 26.1 | 0.0 | 17.4 | 1262 | 1667 / 2653 (n=23) | 1667 / 2653 (n=23) |
| S2:c200-k6-norr | 23 | 52.2 | 13.0 | 17.4 | 2066 | 1460 / 2320 (n=23) | 1460 / 2330 (n=23) |
| S1:c400-k6-norr | 12 | 16.7 | 50.0 | 0.0 | 1331 | 1008 / 1992 (n=12) | 1032 / 1992 (n=12) |

wrongly refused % by row: S3:c200-k4-rr=13.0, S4:c200-k2-rr@20b=21.7, S4:c200-k2-rr=56.5, S2:c200-k6-norr=17.4, S1:c400-k6-norr=33.3

Definitions: percentiles are pandas linear quantiles over status=ok rows (n shown). Tokens per answer = all tokens of all rows / ok rows. Cost uses list prices per row's model (upper bound: cached input is billed at half). Peak queue = most requests in the system at once.

