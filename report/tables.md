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

wrongly refused % by row: S3:c200-k4-rr=13.0, S4:c200-k2-rr@20b=21.7, S4:c200-k2-rr=56.5

Definitions: percentiles are pandas linear quantiles over status=ok rows (n shown). Tokens per answer = all tokens of all rows / ok rows. Cost uses list prices per row's model (upper bound: cached input is billed at half). Peak queue = most requests in the system at once.

