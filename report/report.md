# 1 What I built and why

Under a real Groq `8,000` TPM limit on `openai/gpt-oss-120b` (`steady-real` measured `1,678` tokens/answer; `1,700` configured), raw uncached throughput is `8,000 / 1,678 ≈ 4.8` full answers/min (`4.2/min` at `0.9` safety, or `6.0/min` at `S3` eval's `1,333` tokens). **Tokens per answer** is therefore the binding capacity unit. I built **Relay**, a paced RAG streamer over *Attention Is All You Need*, *Moshi*, and *H-Net* that protects `slo.patience_first_s = 6.0s` (`first_line_p95_ms = 2500ms`) by shrinking retrieval/output tokens, shedding to `openai/gpt-oss-20b`, or returning cited quotes.

```text
Query ─▶ Cache/coalesce ─▶ Understand (20b, 0.8s; heuristic fallback)
  ─▶ Retrieve (BM25+dense RRF, opt. BGE rerank + PaperMap)
  ─▶ Ladder (Full 120b/S3 ─▶ Lean 120b/S4 ─▶ Fallback 20b/S4 ─▶ Extractive)
  ─▶ Draft + LineGate (cite + number check, ≤1 redraft) ─▶ Verified lines
```

# 2 The real service and my stand-in

Groq probes (`probe summarize`) measured first-content `p50/p95/p99` of `549/891/1093 ms` (`120b`, `n=30`) and `507/857/949 ms` (`20b`, `n=20`), with prefix caching (`cached=1024` on `1,134` prompt tokens). `relay calibrate` (`@1.2k` prompt) refit TTFT samples of `[527, 815, 985]` ms (`120b`, `n=51`), `[594, 931, 1117]` ms (`20b`, `n=32`), and `[559, 833, 962]` ms (`qwen`, `n=33`). On identical 29-request workloads (`343s`, `5.1/min`), both `steady-real` and `steady-twin` finished `100%` `ok` with `0` retries (`Table 2a/2b`): the stand-in slightly under-predicted first-token latency (`1622/2288/2334` vs `1767/2567/2665 ms`) and token spend (`1374` vs `1678` tokens/answer; `$0.246` vs `$0.334` per `1k`), due to `120b` reasoning variance (`0.08–0.62` share). The least-supported term is shared `load.alpha = 0.17787` (`c0=1`)—plus zeroed `p_slow` on `120b`/`20b` and `20b`'s clamped `ttft_per_1k_prompt_ms = 0.0`—because bursts (`concurrency.jsonl`, `n=45`) were only probed on `20b`.

# 3 Results

In `Tables 2a`/`2b`/`3`, `steady-real` (`n=29`), `steady-twin` (`n=29`), and `steady` (`n=394`, `39.6/min`) all hit `100%` success (`steady` `p95 = 2480 ms < 2500 ms` SLO). One number nobody could guess is the impact of the 600s cache + in-flight coalescing at `55.3/min`: `burst` (`n=552`, `549 ok`) sustains `99.5%` success (`p95 = 3571 ms`, peak queue `14`), whereas `burst-nocache` (`n=552`, `503 ok`) drops to `91.1%` (`p95 = 5201 ms`, queue `36`, `3.6%` rate-limited, `5.3%` timed out). Coalescing also explains why `steady` sustains `39.6/min` (`8x` raw capacity). Under `232.6/min` (`overload`, `n=1395`), `74.1%` succeed (`$0.117/1k`), `19.9%` shed to extractive quotes, and `5.7%` time out. On our 23-question eval (`Table 3`), default `S3` scores `52.2%` correct & cited (`1,333` tok/ans) vs `S2`'s `56.5%` (`2,066` tok/ans).

# 4 Where the time goes

In `relay slow --scenario overload --pct 99` (`overload:00354`, `5951 ms` total, `before_first_ms=5951`, `after_first_ms=0`), governor queueing dominates: `understand:20b` took `807 ms` (`3→810`), `retrieve:S3` took `33 ms` (`810→843`, `rerank_skipped=True`), **`queue:120b` waited `4,061 ms` (`68.2%`, `849→4910`)**, then stepped down to `retrieve:S4` (`70 ms`, `4912→4982`), `0 ms` queue on `20b`, and `964 ms` on `model:20b` (`4987→5951`, `ttft_raw_ms=942.3`, `retry_ms=0`).

# 5 When a limit is hit

`choose_level` (`decide.py`) cascades: **Full (`120b/S3`)** if `daily_frac > 0.25` and `wait <= 1.0s`; **Lean (`120b/S4`)** if `daily_frac > 0.02` and `wait <= 3.0s`; **Fallback (`20b/S4`)** if `daily_frac > 0.02` and `wait <= 3.0s` (or min wait within `patience_left_s - 1.5s`); else **Extractive**. Under `minute-halved` (`n=459`, `96.1%` ok), `353` stayed at `S3`, `55` used `S4`, `33` spilled to `S4@20b`, `13` (`2.8%`) returned extractive quotes (`rate_limited`), and `5` (`1.1%`) timed out. Under `daily-exhausted` (`n=359`, `90.8%` ok), `108` ran `S3`, `52` `S4`, **`166` `S4@20b`**, `33` (`9.2%`) extractive, with **`0` timeouts**.

# Retrieval: quality versus cost

In `report/recall_sweep.csv`, `S2` (`c200k6`, no rerank) hits `1.000` recall at `1,104` passage tokens vs `S1` (`c400k6`) `1.000` at `1,459`. Cross-encoder reranking at small `k` actually lowered recall (`c200k4` `0.947→S3` `0.842`; `c200k2` `0.895→S4` `0.737`), likely because BGE down-ranks structural headers (`Table`, `§`). In `Table 3`, `S2` leads (`56.5%` correct, `8.7%` wrong, `2,066` tok/ans) ahead of `S3` (`52.2%`, `1,333` tok/ans; `10/23` degraded to `S4@20b`). `S4` on `120b` over-refused (`56.5%` wrong refusal, `26.1%` correct), whereas **`S4@20b`** reached `43.5%` correct at `948` tokens and `996/2095 ms`. Leaving `retrieval.refuse_below: null` (G4) let `in_scope` + `NOT_IN_PAPERS` achieve `4/4` (`17.4%`) correct refusals across all 23-question evals.

# How this library will mislead a naive retriever

Dense `400`-token retrieval (`k=4`, no routing, `report/naive_baseline.json`) scores `18/19` on coarse section hits yet misranks real queries:
- **`q18`** (*"model dimension and number of layers of the Transformer?"*, `hit=False`): top hits `A4` (`Figure 1`) and `A34` (`§6.2`) miss both `Attention §3.1` and `Moshi Table 1`.
- **`q03`** (*"warmup steps and label smoothing... base Transformer"*): `M38` (`Moshi §3.4.1 RQ-Transformer`) and `M28` (`§3.3.1`) outrank `A25` (`Attention §5.3`).
- **`q19`** (*"batch size... Helium in Moshi vs Attention"*): returns four Moshi chunks (`M58`, `M65`, `M17`, `M66`) and zero `Attention §5.1` chunks without `per_paper=True`.

# 6 Retries

`RetryPolicy` (`retry.py`, D6) never retries after first streamed content (`sent_any`) or on `rate_limited_day`/`bad_request`/`midstream`, caps `max_attempts=2`, budgets `max(3, 0.1*reqs)` retries/60s, respects `retry_after_s`, and uses `0.25–2.0s` jitter otherwise. Comparing `overload` vs `overload-naive-retry` (`n=1395`), budgeted retries beat naive retries on success (`74.1%` vs `73.2%`), 429 rate (`19.9%` vs `22.1%`), and p50 latency (`3798` vs `3843 ms`), both averaging `0.04` retries/req (`25` queue) because the ladder deflects most pressure upfront.

# 7 What the person sees

On daily exhaustion (`9.2%` of `daily-exhausted`; `2.8%` of `minute-halved`), the user gets `EXTRACTIVE_DAY` (`messages.py`) plus up to two 320-char cited quotes (`extractive.py`):
> `Today's model budget is used up. Written answers resume in about {h} h {m} min. Until then, here is what the papers say, quoted directly:`
> `"..." [Paper | §Section | p.X]`

# 8 What I'm unsure about, what I left out, what I learned

Gaps remaining: `eval-S1` stopped at `12/23` (`10` degraded to `S4@20b`) and `10/23` `eval-S3` runs also degraded under Groq quota; scenarios ran before updating `tokens_per_answer_full` from `1400` to `1700`; `limits_prior.tpd` (`200,000`) was never exhausted (`low` confidence). Both `hnet_table5.md` and `hnet_table7.md` match `hnet.pdf` pp. 17 & 37 (`verified: true`), and manual review of `eval/grades_review.csv` rescued `10/20` false `"wrong"` exact-string grades. Lesson: under tight TPM, caching/coalescing and ladder shedding beat reranking and retries.

# Your feedback on Vaani

Vaani targets real-time multilingual voice agents (`~450ms` p90 latency, Hindi/Hinglish switching, call-screening handoffs, **Teacher AI** prompt diffs, **Memories** context filtering). Two concrete notes: (1) on the homepage HTML, hero counters (`0M+`, `0.0%`, `<0ms`) render as zeros before JS runs, and direct links from "Start free" to API rate-limit/latency docs would help developers; (2) building Relay showed why **Memories** context filtering matters for voice—once a TPM ceiling adds `4s` of upstream queueing, only smaller prompts and graceful fallbacks avoid dead air.

# How I worked with the assistant

I started from an assistant-authored `PLAN.md` (D1–D12, G1–G6) and an overnight run (`NEXT_STEPS.md`, `raw/dryrun/`) that fixed schema, caching, and fallback bugs. Afterward, I had the assistant explain each module in plain terms, create a 15–24 question study set, verify `hnet_table5.md`/`hnet_table7.md` against `hnet.pdf` pages 17 and 37, audit all 20 `"wrong"` rows in `eval/grades_review.csv` (rescuing 10 valid answers), update `results.csv`/`tables.md`, and draft this report.

## Table 1: Limits you found in the real service

| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | 30 (`qwen/qwen3.8-27b`; prior 30) | `rpm-edge` probe hit 429 at `i=30`: `"RPM: Limit 30, Used 30, Requested 1"` | high (`qwen`), medium (prior for `120b`/`20b`) |
| Tokens per minute | 8,000 TPM (`120b`, `qwen`); 7,000 ITPM (`qwen`) | `effort` (`120b`) & `tpm-edge` (`qwen`) 429 bodies + `x-ratelimit-limit-tokens: 8000` | high |
| Requests per day | 1,000 | `x-ratelimit-limit-requests: 1000` header on probe 429 responses | high |
| Tokens per day | 200,000 (prior; unexhausted) | `limits_prior.tpd` in `config/relay.yaml`; probes did not exhaust daily tokens | low |
| Requests in flight at once | No hard cap up to 45 bursts (`20b`); latency climbs (`alpha=0.17787`) | `concurrency` probe (`n=45` on `20b`: p50 `509` ms -> p95 `1532` ms) + `calibrate` fit | medium (`20b` only) |
| What it returns when a limit is hit | HTTP `429` JSON (`rate_limit_exceeded`, `type: tokens/requests`) + `retry-after` & `x-ratelimit-*` headers | `raw/probe/effort.jsonl`, `rpm-edge.jsonl`, `tpm-edge.jsonl` | high |
| Time until it recovers | `1–3 s` (`retry-after`) for RPM/TPM; `~43–56 min` `x-ratelimit-reset-requests` | `retry-after` (`1`, `2`, `3`) & `x-ratelimit-reset-*` headers in 429 responses | high (minute), medium (day) |

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
| S3:c200-k4-rr | 23 | 52.2 | 17.4 | 17.4 | 1333 | 1679 / 3879 (n=23) | 1682 / 3947 (n=23) |
| S4:c200-k2-rr@20b | 23 | 43.5 | 17.4 | 17.4 | 948 | 996 / 2095 (n=23) | 996 / 2095 (n=23) |
| S4:c200-k2-rr | 23 | 26.1 | 0.0 | 17.4 | 1262 | 1667 / 2653 (n=23) | 1667 / 2653 (n=23) |
| S2:c200-k6-norr | 23 | 56.5 | 8.7 | 17.4 | 2066 | 1460 / 2320 (n=23) | 1460 / 2330 (n=23) |
| S1:c400-k6-norr | 12 | 66.7 | 0.0 | 0.0 | 1331 | 1008 / 1992 (n=12) | 1032 / 1992 (n=12) |

wrongly refused % by row: S3:c200-k4-rr=13.0, S4:c200-k2-rr@20b=21.7, S4:c200-k2-rr=56.5, S2:c200-k6-norr=17.4, S1:c400-k6-norr=33.3

Definitions: percentiles are pandas linear quantiles over status=ok rows (n shown). Tokens per answer = all tokens of all rows / ok rows. Cost uses list prices per row's model (upper bound: cached input is billed at half). Peak queue = most requests in the system at once.
