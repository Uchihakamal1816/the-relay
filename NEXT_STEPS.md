# Morning handoff (written overnight, 2026-11-12)

Repo: https://github.com/Uchihakamal1816/the-relay (private). Work is on branch `build`; `main` has only the brief and plan.
Probes and all scenarios have run (real + calibrated stand-in). Tables: `report/tables.md`.

## Done (all tested, 102 tests pass)
| Task | State |
|---|---|
| 0 scaffold, config, results writer, traces | done |
| 1 fetch papers (pinned, sha256 manifest) | done |
| 2 HTML parse + PDF audit | done; H-Net Table 5/7 patches drafted, **unverified** |
| 3 chunk + hybrid index + re-rank + search tool | done; Attention Table 2 in top 3 for the sanity query |
| 3b paper map | done |
| 4 streaming client | done (mocked tests only) |
| 5 probes | code done, **not run** (needs your key) |
| 6 governor + retry | done |
| 7 stand-in + calibration fit | done; profile uses priors until probes run |
| 8 pipeline | done; end-to-end works against the stand-in |
| 9 eval harness | code done; **questions are yours** |
| 10 load runner + 9 scenarios | done; dry-run results in `raw/dryrun/` (pre-calibration, not for the report) |
| 11 tables + slow-request breakdown | done |
| 12 report skeleton + submit-check | done; prose is yours |

## Your steps, in order
1. Create the Groq key, put it in `.env` as `GROQ_API_KEY=...`, note RPM/TPM/RPD/TPD from the console limits page.
2. Decide G1 (burn qwen's daily quota for a real daily-limit 429?).
3. Run probes (each prints its token estimate and aborts above 80% of what is left today):
   ```bash
   uv run relay probe models --real
   uv run relay probe latency --real
   uv run relay probe latency-small --real
   uv run relay probe concurrency --real
   uv run relay probe cache --real
   uv run relay probe effort --real
   uv run relay probe rpm-edge --real
   uv run relay probe tpm-edge --real
   uv run relay probe summarize
   uv run relay calibrate
   ```
   If the first call returns 400 on `stream_options` or `include_reasoning`, remove that field in `src/relay/llm/client.py` (`LLMClient.body`).
4. Check `data/patches/hnet_table5.md` and `hnet_table7.md` against PDF pages 17 and 37, set `verified: true`, run `uv run relay ingest`.
5. Eyeball 20 glossary entries: `uv run python -c "import json,random;e=[x for x in json.load(open('data/corpus/papermap.json')) if x['kind'] in ('symbol','term')];random.seed(0);[print(x['paper'],x['key'],'|',x['snippet'][:80]) for x in random.sample(e,20)]"`
6. Write `eval/questions.yaml` (≥20), then `uv run relay eval validate` and `uv run relay eval recall` (free).
7. Real eval runs per PLAN.md Task 9 Step 7, then review `eval/grades_review.csv`, set `retrieval.default`, `capacity.tokens_per_answer_full` in `config/relay.yaml`.
8. Re-run stand-in scenarios (stand-in must be running: `uv run relay standin serve &`), then `uv run relay load run steady-real --real`, then `uv run relay tables`.

## Overnight dry run: overload (stand-in, 4x capacity, 1,165 requests, priors not calibrated)
Each row is a fix applied on top of the previous one. Data in `raw/dryrun/`. Good material for section 4 and "a number you could not guess".

| Version | ok | gave up | quoted fallback | retrieval p50 ms | governor queue p50 ms |
|---|---|---|---|---|---|
| as built | 448 | 713 | 4 | 2925 | 832 |
| + cap concurrent re-ranks at 4 | 458 | 624 | 81 | 1379 | 2170 |
| + queue-aware wait, first-token patience in the ladder | 540 | 494 | 129 | 2769 | 324 |
| + re-ranker `batch_size=1` | 994 | 43 | 125 | 464 | 1290 |

- **The binding limit under overload was my own CPU, not Groq.** The local re-ranker (20 pairs) took 1.13 s on an idle machine because batched ONNX pads every pair to the longest; `batch_size=1` gives identical scores in 326 ms.
- **Ladder bug fixed:** it compared waits with total patience (20 s) instead of first-token patience (6 s), so it queued people who then gave up.
- **Governor bug fixed:** its wait estimate ignored reservations already queued ahead, and the lock wait had no deadline.
- **Still open:** the quoted fallback reaches people at about 5 s p50, close to the 6 s patience, because the decision comes after retrieval. Deciding before retrieval when the queue is long would fix it; I left that to you.

## Things I changed or found (worth knowing for the live session)
- **Cold start**: the re-ranker's first call takes seconds, so the first requests timed out to "no re-rank". Added `PaperSearch.warmup()`, called at startup.
- **Moshi Table 17 sits outside any section in the arXiv HTML**; the parser now attaches such floats to the previous section. Counts now match: 4 / 19 / 8 tables.
- **Retry budget floor**: 10% of trailing-minute requests allowed zero retries at low traffic; added `retry.min_per_min: 3`.
- **Stand-in understand fallback**: with no recording it now answers like the keyword router instead of "all papers". This is one more way the stand-in is kinder than the real model; say so in section 2.
- **Probe budget**: the concurrency probe estimate is 10K tokens, not the plan's 5K.
- **Paper map**: symbol detection is conservative (2 to 8 symbols per paper); section titles are also term entries (e.g. "inner monologue" → Moshi §3.4.4).
- Low PDF-coverage pages are references, figures and appendix visualisations, except H-Net p.21 (Table 6 layout) and Moshi p.34; worth a glance.

## Final run (2026-11-12, calibrated stand-in + real steady)
| Scenario | ok % | rate-limited % | timeout % | first token p50/p95 ms |
|---|---|---|---|---|
| steady-real (real) | 100 | 0 | 0 | 1767 / 2567 |
| steady-twin (stand-in) | 100 | 0 | 0 | 1622 / 2288 |
| burst / burst-nocache | 99.5 / 91.1 | 0.2 / 3.6 | 0.4 / 5.3 | 1714 / 1904 p50 |
| overload (4x) | 74.1 | 19.9 | 5.7 | 3771 / 5607 |
| overload-naive-retry | 73.2 | 22.1 | 4.7 | 3829 / 5599 |
| minute-halved | 96.1 | 2.8 | 1.1 | 2023 / 4603 |
| daily-exhausted | 90.8 | 9.2 | 0 | 2298 / 3556 |

- Stand-in vs real: p50 first token 8% fast, tokens/answer 1374 vs 1678 (18% low). Say so in section 2.
- `capacity.tokens_per_answer_full` set to 1700 from steady-real (scenarios above ran at 1400, so stand-in offered rates are ~20% high vs the new figure).
- Budgeted vs naive retry under overload: only ~1 pt difference; retries were rare (0.04/request) because the ladder degrades before retrying.
- Overload p99 (`relay slow --scenario overload --pct 99`): 4.1 s of 5.95 s was waiting in the 120b queue before falling to 20b (`WouldMissDeadline`). Deciding the level before queueing would cut this; same open issue as the quoted fallback.
- overload had 4 `error` rows (0.3%); check `raw/traces/overload.jsonl` for the exception.
- Before submitting: redact the Groq org id from 429 bodies in `raw/`.

## Still yours
Eval questions → `eval validate` / `recall` / real eval (Table 3); review grades; confirm patches; Table 1 prose; write-up, Vaani feedback, assistant-history section; G1 decision.
