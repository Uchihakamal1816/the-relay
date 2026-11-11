<!-- [KAMAL] Write the prose yourself, max 2 pages before the tables. Evidence to pull is listed under each heading; delete the lists when done. -->

# 1 What I built and why
- Capacity arithmetic: 8,000 TPM / measured tokens per answer = answers per minute (Table 3 + `relay probe summarize`).
- Pipeline diagram (PLAN.md D12). The one decision: tokens per answer is the capacity unit.

# 2 The real service and my stand-in
- Table 1 (below), from `relay probe summarize`.
- `relay calibrate` real vs refit percentiles; steady-real vs steady-twin rows; least-supported calibration term.

# 3 Results
- Tables 2a, 2b; n per row; one number nobody could guess.
- Answer quality on your questions (Table 3) and tokens per answer.

# 4 Where the time goes
- `relay slow --scenario overload --pct 99` output.

# 5 When a limit is hit
- minute-halved and daily-exhausted rows grouped by retrieval_setting; ladder table (PLAN.md D4).

# Retrieval: quality versus cost
- Table 3; report/recall_sweep.csv; G4 reasoning; lean setting under pressure.

# How this library will mislead a naive retriever
- Three questions from report/naive_baseline.json with the wrong passage each.

# 6 Retries
- Policy (PLAN.md D6); overload vs overload-naive-retry.

# 7 What the person sees
- The EXTRACTIVE_DAY reply exactly as sent (src/relay/pipeline/messages.py).

# 8 What I'm unsure about, what I left out, what I learned

# Your feedback on Vaani

# How I worked with the assistant

## Table 1: Limits you found in the real service

| Limit | Value you found | How you found it | Confidence (high, medium, low) |
|---|---|---|---|
| Requests per minute | | | |
| Tokens per minute | | | |
| Requests per day | | | |
| Tokens per day | | | |
| Requests in flight at once | | | |
| What it returns when a limit is hit | | | |
| Time until it recovers | | | |

<!-- paste report/tables.md here verbatim after `relay tables` -->
