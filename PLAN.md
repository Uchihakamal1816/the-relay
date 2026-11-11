# The Relay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, measure and document a rate-limit-aware, cited question-answering pipeline over three pinned arXiv papers on Groq's free tier, plus a calibrated stand-in model service, producing `results.csv`, the four required tables and the evidence for a 2-page write-up.

**Architecture:** One asyncio Python process runs the pipeline (understand → retrieve → decide → draft → check). Every model call passes through a client-side governor that mirrors the provider's limits and picks a degrade level; the draft is streamed to the user one verified line at a time. A separate FastAPI process is the stand-in: it speaks Groq's wire format, replays recorded real answers, and has timing and limits fitted from real probe data.

**Tech Stack:** Python 3.12 (uv), asyncio + httpx (no provider SDK), FastAPI + uvicorn (stand-in), BeautifulSoup + lxml (arXiv HTML), PyMuPDF (PDF audit), rank-bm25, fastembed (ONNX, CPU), tiktoken, numpy, pandas, PyYAML, pytest + pytest-asyncio.

**Spec:** `The_Relay_Assignment_Brief.pdf` (repo root). Read it before any task; this plan argues from it.

**Tags used below:** `[REAL]` spends the shared daily Groq budget, run only when Kamal says go for that step. `[KAMAL]` is a human step the implementer must not do.

---

## 0. Kamal: read this before handing the plan over

**What the reviewers score** (brief, last page): how you direct the assistant, how you check it, where you overrule it, and whether choices follow from measurements. Your real conversation history is submitted, and a live session follows where you change the system.

**So these are yours, not the assistant's:**
- The eval questions and gold answers (brief: "questions you wrote yourself"). Task 9 gives the schema and the mix, not the questions.
- Every number marked `# KAMAL` in `config/relay.yaml` (what "fast" means, patience, ladder thresholds). Defaults are given; set them after the probe.
- The prose of the write-up, the Vaani feedback, "what was new to me", "where I overruled it".
- A manual pass over every graded answer (about 120 rows) and over two patched tables.

**Decision gates** (my recommendation first; overrule with data where you disagree):

| Gate | Recommendation | Alternative | Decide by |
|---|---|---|---|
| G1 Throwaway model for limit probing | Burn `qwen/qwen3.8-27b` (not in the pipeline) to hit the per-minute and per-day limits for real | Report daily limits from docs only, confidence "medium" | Before Task 5 |
| G2 Meaning of "fast" | Time to the first *verified* line, p95 ≤ 2500 ms at steady load; user gives up at 6 s | Raw first provider token | After Task 5 |
| G3 Keep the LLM "understand" step | Keep only if it raises recall@k over the heuristic router in Task 9 | Drop it, save a call | After Task 9 recall sweep |
| G4 Default retrieval setting | The setting with the fewest tokens per answer whose correct % is within one question of the best | Highest correct % | After Task 9 |
| G5 Answer cache for bursts | Exact-match cache + in-flight coalescing on; semantic cache deliberately not built | Off | After Task 10 burst A/B |
| G6 Stand-in load scale | Measured free-tier limits ×10 (keeps the RPM:TPM ratio, gives n ≥ 300 per run) | Developer-plan numbers from the docs | Before Task 10 |

**Two ambiguities in the brief to resolve yourself:**
- "Maximum 2 pages of writing" versus four required tables. My reading: 2 pages of prose with all headings, tables directly after. State your reading in one line at the top, or ask the recruiter.
- Deadline is 4 days from receipt. The brief file landed on 2026-11-11, so plan for 2026-11-15. Confirm.

---

## 1. Facts checked on 2026-11-11

| Fact | Source |
|---|---|
| Free-plan chat models: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`. Each: 30 RPM, 1K RPD, 8K TPM, 200K TPD. Llama models are Enterprise-only now. | console.groq.com/docs/rate-limits |
| Limits are per organisation and per model. | same |
| Headers on every response: `x-ratelimit-limit-requests` and `-remaining-requests` and `-reset-requests` refer to **RPD**; `x-ratelimit-limit-tokens`, `-remaining-tokens`, `-reset-tokens` refer to **TPM**. `retry-after` (seconds) only on 429. RPM and TPD are **not** in headers. | same |
| Limit hit returns HTTP 429. Server errors (500/502/503) are not charged. 413 = request too large. 499 = cancelled by caller. | console.groq.com/docs/errors |
| All three free models are reasoning models. gpt-oss: `reasoning_effort` low/medium/high, cannot be turned off. qwen: default is none. `include_reasoning` (bool) and `reasoning_format` are mutually exclusive. | console.groq.com/docs/reasoning, API reference |
| Prompt caching is automatic on gpt-oss-20b/120b only. Exact prefix match, minimum 128 to 1024 tokens depending on model, expires after 2 h idle. "Cached tokens do not count towards your rate limits", but they are subtracted *after* processing. | console.groq.com/docs/prompt-caching |
| `usage` carries `queue_time`, `prompt_time`, `completion_time`, `total_time`, `prompt_tokens_details.cached_tokens`. | same |
| Prices per 1M tokens: 120b $0.15 in / $0.60 out; 20b $0.075 / $0.30; qwen $0.80 / $4.00. Doc speeds: ~500, ~1000, ~450 tokens/s. Developer plan: 250K TPM, 1K RPM. | console.groq.com/docs/models |
| `max_tokens` is deprecated in favour of `max_completion_tokens`. `stream_options`, `seed`, `service_tier` (default `on_demand`) exist. | API reference |
| All three paper versions exist with PDF, TeX and HTML: `arxiv.org/{pdf,html}/1706.03762v7`, `/2410.00037v1`, `/2507.07955v2`. | arxiv.org abs pages |
| HTML structure: Attention 8 sections, 4 tables, 40 references, ~6.2K words. Moshi 7 sections + appendices A to F, 19 tables, 116 references, ~29.4K words. H-Net 6 sections + appendices A to F, 8 tables, 115 references, ~20.4K words. Every `<math>` has an `alttext` (LaTeX). Zero LaTeXML error nodes. | DOM inspection |
| **H-Net HTML has captions but no body for Table 5 and Table 7.** The PDF has them. | DOM inspection |
| fastembed 0.9.1 (Python ≥ 3.10): `TextEmbedding()` defaults to `BAAI/bge-small-en-v1.5` (384-d); `TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2").rerank(query, docs)`. | github.com/qdrant/fastembed, PyPI |
| PyMuPDF 2.1 needs Python ≥ 3.11. System Python is 3.10.12, so use a uv-managed 3.12. | PyPI, local check |
| Machine: 12 cores, 15 GB RAM (about 4 GB free), RTX 2050 4 GB (not needed). | local check |
| Vaani sells voice agents and advertises p90 latency of about 450 ms and "zero dead air". | vaani.dev |

**Not known until the probe runs (Task 5):** the exact 429 body; where `usage` sits in a stream; whether `max_completion_tokens` is reserved against TPM; whether rejected requests consume quota; whether limits refill continuously or per window; any cap on requests in flight; whether caching actually lowers `x-ratelimit-remaining-tokens` usage on the free tier.

---

## 2. Design decisions

### D1. The capacity unit is tokens per answer
- RPM allows 30 calls a minute; TPM allows 8000 tokens a minute. At about 1700 tokens per answer that is 4.7 answers a minute. Tokens bind about six times sooner than requests, per minute and per day (200K TPD ≈ 117 answers; 1K RPD = 1000).
- Every knob (piece size, pieces fetched, reasoning effort, output cap, degrade level) is judged by `answers/min = TPM / tokens per answer`.
- Cost: a retriever that fetches more is paid for in turned-away users, not in dollars.

### D2. Three models, three separate buckets
- Drafter: `openai/gpt-oss-120b`, `reasoning_effort: low`. Production model, cheapest per token, supports caching.
- Understand step and fallback drafter: `openai/gpt-oss-20b`, `reasoning_effort: low`.
- `qwen/qwen3.8-27b`: never in the pipeline. It is the bucket to exhaust on purpose (G1), so learning what a daily limit looks like does not cost a day of pipeline budget.
- This is model tiering inside published per-model limits, one account. It is not a way around a limit. Say so in the write-up.

### D3. Line-gated streaming: fast and checked
- Tension in the brief: report time to first token, yet check the reply before sending.
- The drafter writes one sentence per line, each ending with passage ids. A line is released to the user only after a local check passes: (a) it cites at least one id, (b) every id was actually retrieved, (c) every number in the line appears in a cited passage or in the question.
- `first_token_ms` is therefore the first *verified* line reaching the user. The raw provider first token is kept in the trace so the cost of gating is a measured number.
- If line 1 fails: nothing was sent, so one re-retrieve + re-draft is allowed (the loop, capped at 1, only at full level). If a later line fails: stop, cancel the stream, append "one statement left out". Unverified text is never sent.
- The check costs no quota, so it is never shed under load. That is the rule: the check that must always run has to be the one that is free.
- Cost: a correct sentence that rephrases a number ("65 million" for "65M") is withheld. Fewer complete answers, never a wrong number.

### D4. Degrade ladder (who gets what)
| Level | Model | Retrieval | When |
|---|---|---|---|
| L0 full | 120b | default setting, LLM understand, 1 re-draft allowed | wait for a full grant ≤ `wait_full_s` |
| L1 lean | 120b | lean setting (half the pieces), heuristic understand, no re-draft | minute bucket tight, or daily bucket under `daily_lean_below` |
| L2 fallback | 20b | lean setting | 120b wait too long or its day is spent |
| L3 extractive | none | top 2 passages quoted with citations, honest notice, time when written answers resume | no model can answer inside the user's patience, or all daily buckets are spent |
- L3 is never wrong (it only quotes) and costs zero tokens, so nobody is turned away. It is a weaker answer and is recorded as `rate_limited`, not `ok`.
- Requests whose user has already given up are dropped at dequeue, before any tokens are spent.

### D5. Governor: mirror the provider, learn from headers
- Token bucket: a counter that refills at a fixed rate up to a cap; a request proceeds only if it can take its cost. One bucket each for RPM, TPM, RPD, TPD, per model.
- Limits are never trusted from config. `x-ratelimit-limit-*` and `-remaining-*` overwrite local state on every response, so a limit that changes without notice is picked up within one response.
- Reservation = estimated prompt tokens + `max_completion_tokens`; refunded to actual usage when the call ends.
- Daily usage is rebuilt at start from an append-only ledger (`raw/real_calls.jsonl`), because the day's budget is shared by every run.

### D6. Retries
- Retry only before the first verified line has been sent.
- 429 per-minute: not retried blindly. The governor waits `retry-after` if that still fits the user's patience, otherwise it degrades.
- 429 per-day: never retried. Degrade.
- 5xx, connection error, no first token within `ttft_timeout_s`: at most 2 retries, exponential backoff with full jitter (sleep uniform in [0, min(cap, base·2ⁿ)]).
- Retry budget: retries may not exceed 10% of requests in the trailing 60 s. Above that, fail to the next ladder level.
- A/B in Task 10: the same overload with a naive policy (retry everything at once, 3 times, including 429s). This is the "retrying made things worse" case, measured.

### D7. Ingestion: arXiv HTML, audited against the PDF
- HTML gives exact section numbers, real table cells and LaTeX math. PDF text extraction scrambles tables, and tables are where the "cannot answer from memory" questions live.
- HTML is lossy (H-Net Tables 5 and 7 have no body). So: the PDF is fetched too, hash-pinned, and used to audit coverage page by page, to fill gaps with human-checked patches, and to add page numbers to citations.
- References are excluded from the index.

### D8. Retrieval: hybrid, local, no vector database
- BM25 (keyword scoring) + dense embeddings, fused by reciprocal rank fusion (RRF: score = Σ 1/(60 + rank)). Optional cross-encoder re-rank, on CPU.
- About 600 to 1500 chunks total: brute-force numpy search is exact and takes under a millisecond.
- Re-ranking is local compute that costs no quota. It is what makes fetching fewer pieces safe, which raises answers per minute. Local CPU is traded for provider tokens.
- Every chunk carries a header `[paper | §section title | Table N | p.X]` in both the index and the prompt.
- A second tool, the paper map, is built by code from the parsed HTML with no model calls: the section tree, an index of tables and figures with captions, and a glossary of where each symbol or named term is first defined. When a question names a table, figure, section, symbol or defined term, the map returns the exact chunk directly. Its hits replace the lowest-ranked search hits, so the number of passages and tokens per answer stay the same.
- Why a map and not a model-written wiki: a wiki costs tokens to build (a large share of a day's budget) and paraphrases the papers, so its numbers cannot be checked against source text. The map points into the papers; it never rewrites them.

### D9. Stand-in: real content, fitted timing, controllable limits
- Same HTTP surface as Groq, so one client talks to both; only `base_url` changes.
- Content is replayed from recorded real completions, keyed by (stage, question id, setting, model). So the full pipeline, including the gate and the grader, runs under load with real token counts.
- Timing: time to first token = (a + b·prompt_tokens) × lognormal noise × load factor; then tokens at a fitted rate. Limits use the refill behaviour seen in the probe.
- It can be slow, fail (500/503), hang, cut a stream midway, succeed slowly, refuse with 429, slow down with requests in flight, and change its limits mid-run.

### D10. Load generation is open-loop
- Open-loop: arrival times are fixed in advance (Poisson) and never wait for earlier replies. Closed-loop generators slow down when the system does, which hides the queue (coordinated omission).
- Latency is measured from the scheduled arrival, not from when the request was sent.
- Each simulated user has a patience limit and cancels when it expires. That is a "given up" row.

### D11. Bursts are correlated, so identical questions are coalesced
- The brief says bursts follow an incident that hits many users at once, so many of them ask the same thing.
- Exact-match cache of verified answers (normalised question text, TTL 10 min) plus single-flight (identical questions in flight share one model call).
- Semantic (near-match) caching is deliberately not built: a similar but different question would get a confident wrong answer.
- The duplicate share in a burst is an assumption (default 50%). The burst is run with the cache on and off so the cost of being wrong is a number.

### D12. One file of knobs
- Everything tunable lives in `config/relay.yaml`. The live session will ask for a change; it should be a one-line edit plus one command.

```
question ─► understand ─► retrieve (tool) ─► decide ─► draft (stream) ─► line gate ─► user
            20b, 0.8 s     BM25+dense+RR     level       120b / 20b       local
            │ timeout      │ timeout          │            ▲   │ line 1 fails, L0 only
            ▼ heuristic    ▼ no re-rank       ▼            └───┘ re-retrieve + re-draft once
                                         L3 extractive (no model)
                    every model call: governor.acquire → call → governor.on_response
```

---

## 3. Real-token budget

Per model per day: 200K tokens, 1000 requests. Estimates; replace with measured values after Task 5.

| Day | 120b | 20b | qwen | What |
|---|---|---|---|---|
| 1 | ~76K | ~15K | ~10K, then RPD burn | Probe P0 to P6 |
| 2 | ~130K | ~45K | (TPD burn, optional) | Eval S1, S2; understand calls |
| 3 | ~100K | ~75K | 0 | Eval S3, S4, S4 on 20b; real steady run |
| 4 | reserve | reserve | 0 | Re-runs after fixes, write-up |

Rules:
- All development and all load tests use the stand-in. Real calls happen only in Tasks 5, 9 and 10, behind `--real`.
- Every `--real` command prints its token estimate and aborts if it exceeds 80% of what the ledger says is left today.
- Before any full real run: 3-question smoke run, then the same run against the stand-in.
- Start Day 1's probe as early as possible. The budget clock, not coding time, is the bottleneck.

---

## Global Constraints

- Python 3.12 via uv. No `groq` SDK and no client with built-in retries (they hide 429s and corrupt the retry measurements).
- The API key lives only in `.env` as `GROQ_API_KEY`. Never print it, log it, commit it, or write the `Authorization` header to any file.
- Papers at exactly `1706.03762v7`, `2410.00037v1`, `2507.07955v2`. Fetch by script; do not commit the papers.
- One account, no extra keys, no proxying. Fetch arXiv with a generic User-Agent `relay-fetch/0.1` and 3 s between requests.
- `results.csv` has exactly these columns in this order: `request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade`.
- `source` ∈ {`real`, `stand-in`}. `status` ∈ {`ok`, `rate_limited`, `timeout`, `error`}. `answer_grade` ∈ {`correct`, `wrong`, `refused_correctly`, `refused_wrongly`} or empty when no generated answer exists.
- Times in milliseconds, integers. Percentages to one decimal. Unmeasured cells say `not measured`.
- Every figure in Tables 2a, 2b and 3 is computed from `results.csv` alone by `relay tables`.
- Intervals use `time.perf_counter()`; only `start_time` uses the wall clock (ISO 8601 UTC, ms).
- CPU-bound work (embedding, re-rank) runs in `asyncio.to_thread`. Never block the event loop.
- All randomness is seeded from config. Raw outputs are never edited by hand.
- Work on branch `build`. `main` holds only the brief and this plan until Kamal approves a merge.
- The implementer does not write eval questions, user-facing opinions, or the write-up prose.

## Review Focus

1. A question whose answer is only in H-Net Table 5 or 7 (missing from HTML): expected a cited answer from the patch, not a refusal. Test in Task 2.
2. A question that names no paper where the papers disagree ("How many layers does the encoder have?"): expected one attributed line per paper. Test in Task 8.
3. A 429 whose message cannot be parsed for limit type: expected classification by `retry-after` size, never a crash. Test in Task 4.
4. The user disconnects mid-stream: expected the provider call cancelled and the governor reservation released. Test in Task 8.
5. A stream that ends without a usage block or without `[DONE]` after lines were sent: expected a truncated verified reply, token counts estimated and flagged. Test in Task 8.

---

## File structure

```
config/relay.yaml                 all knobs
config/scenarios/*.yaml           load shapes
config/standin_profile.json       generated by `relay calibrate`
data/papers/                      fetched html + pdf + manifest.json (git-ignored)
data/patches/*.md                 human-checked tables missing from HTML
data/corpus/                      blocks, chunks per size, embeddings, audit.json (git-ignored)
eval/questions.yaml               [KAMAL]
eval/grades_review.csv            [KAMAL] overrides
raw/probe/*.jsonl                 raw real probe traffic
raw/real_calls.jsonl              ledger of every real call
raw/recordings/                   recorded real completions for replay
raw/traces/<scenario>.jsonl       spans per request
results.csv                       the required raw file
report/tables.md, report/report.md
src/relay/types.py                shared dataclasses and enums
src/relay/config.py               yaml → dataclasses
src/relay/trace.py                spans
src/relay/results_csv.py          schema-checked writer
src/relay/ingest/{fetch,parse_html,audit_pdf,chunk}.py
src/relay/retrieve/{index,rerank,tool,papermap}.py
src/relay/llm/{client,governor,retry}.py
src/relay/pipeline/{understand,decide,prompt,gate,extractive,messages,cache,orchestrator}.py
src/relay/standin/{server,limits,latency,replay,calibrate}.py
src/relay/probe.py
src/relay/evalx/{questions,grader,recall,run_eval}.py
src/relay/load/{schedule,runner}.py
src/relay/report/{make_tables,slow_request,submit_check}.py
src/relay/cli.py                  `relay <command>` (argparse)
tests/
```

Shared types (`src/relay/types.py`), used by every task:

```python
Paper = Literal["attention", "moshi", "hnet"]
Status = Literal["ok", "rate_limited", "timeout", "error"]
Grade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]

class Level(IntEnum): FULL = 0; LEAN = 1; FALLBACK = 2; EXTRACTIVE = 3

@dataclass
class Limits: rpm: int; tpm: int; rpd: int; tpd: int

@dataclass(frozen=True)
class Chunk:
    id: str                 # "A12", "M203", "H88": paper letter + running number
    paper: Paper
    section_num: str        # "3.2.1", "A", "" if unnumbered
    section_title: str
    kind: Literal["abstract", "text", "table", "figure_caption"]
    label: str | None       # "Table 3", "Figure 2"
    page: int | None        # 1-based PDF page, None if unmapped
    text: str
    n_tokens: int
    # header -> "[Moshi | §3.4.4 Inner Monologue | p.12]"
    # cite   -> "(Moshi, §3.4.4, p. 12)" or "(Attention, §6.2, Table 3, p. 9)"

@dataclass
class Hit: chunk: Chunk; score: float; rerank_score: float | None = None

@dataclass(frozen=True)
class RetrievalSetting:
    name: str; chunk_tokens: int; k: int; rerank: bool; candidates: int = 20
    # tag -> "S3:c200-k4-rr" / "S1:c400-k6-norr"

@dataclass
class OutEvent: kind: Literal["status", "sentence", "notice", "final"]; text: str; t_ms: int

@dataclass
class Reply:
    request_id: str; text: str; cited_ids: list[str]; refused: bool
    level: Level; status: Status
    first_token_ms: int | None; total_ms: int
    retries: int; tokens_in: int; tokens_out: int
    retrieval_setting: str  # tag, plus "@20b" at L2, "+cache" on a cache hit; "extractive" at L3
```

---

### Task 0: Scaffold, config, results writer, traces

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, `config/relay.yaml`, `src/relay/{__init__,types,config,trace,results_csv,cli}.py`
- Test: `tests/test_results_csv.py`, `tests/test_config.py`, `tests/test_trace.py`

**Interfaces:**
- Produces: `load_config(path="config/relay.yaml") -> Config`; `ResultsWriter(path).replace_scenario(scenario: str, rows: list[dict]) -> None`; `Trace(request_id)` with `span(name, kind)` context manager, `kind ∈ {"wait","work","retry"}`, `to_dict()`; `split(trace_dict) -> {"wait_ms","work_ms","retry_ms","before_first_ms","after_first_ms"}`.

- [ ] **Step 1: Init**

```bash
cd /home/uchihakamal/Desktop/vaani_ai && git init -b main
git add The_Relay_Assignment_Brief.pdf PLAN.md && git commit -m "docs: brief and plan"
git switch -c build
uv init --python 3.12 --package --name relay
uv add httpx fastapi uvicorn pyyaml numpy pandas rank-bm25 fastembed pymupdf beautifulsoup4 lxml tiktoken python-dotenv
uv add --dev pytest pytest-asyncio
```
`.gitignore`: `.env`, `.venv/`, `data/papers/`, `data/corpus/`, `.claude/`, `__pycache__/`. `[project.scripts] relay = "relay.cli:main"`. pytest: `asyncio_mode = "auto"`.

- [ ] **Step 2: Write `config/relay.yaml`**

```yaml
models:
  primary:  {id: openai/gpt-oss-120b, reasoning_effort: low,  price_in: 0.15,  price_out: 0.60}
  fallback: {id: openai/gpt-oss-20b,  reasoning_effort: low,  price_in: 0.075, price_out: 0.30}
  sacrificial: {id: qwen/qwen3.8-27b, reasoning_effort: none}
endpoints:
  real: https://api.groq.com/openai/v1
  standin: http://127.0.0.1:8900/openai/v1
limits_prior: {rpm: 30, tpm: 8000, rpd: 1000, tpd: 200000}   # docs 2026-11-11; headers overwrite
governor: {safety: 0.9}
slo:                         # KAMAL
  patience_first_s: 6.0
  patience_total_s: 20.0
  first_line_p95_ms: 2500
ladder:                      # KAMAL
  wait_full_s: 1.0
  wait_lean_s: 3.0
  daily_lean_below: 0.25
  daily_stop_below: 0.02
retrieval:
  settings:
    S1: {chunk_tokens: 400, k: 6, rerank: false}
    S2: {chunk_tokens: 200, k: 6, rerank: false}
    S3: {chunk_tokens: 200, k: 4, rerank: true, candidates: 20}
    S4: {chunk_tokens: 200, k: 2, rerank: true, candidates: 20}
  default: S3                # KAMAL, after Task 9
  lean: S4
  refuse_below: null         # set in Task 9
  tool_timeout_s: 1.5
understand: {timeout_s: 0.8, max_completion_tokens: 200}
draft: {max_completion_tokens: 400, lean_max_completion_tokens: 250, ttft_timeout_s: 4.0, total_timeout_s: 12.0, max_redrafts: 1}
retry: {policy: budgeted, max_attempts: 2, base_s: 0.25, cap_s: 2.0, budget_ratio: 0.1}
cache: {enabled: true, ttl_s: 600}
papermap: {enabled: true, max_hits: 2}
capacity: {tokens_per_answer_full: 1700}   # replace with the Task 9 measurement
seed: 7
```

- [ ] **Step 3: Write failing tests**
  - `test_header_exact`: first line of a new file equals the 12-column header in Global Constraints.
  - `test_replace_scenario`: writing scenario `a` twice, then `b`, leaves the second `a` batch and `b`.
  - `test_rejects_bad_values`: `status="done"` or `source="standin"` or `answer_grade="ok"` raises `ValueError`; empty `answer_grade` and empty `first_token_ms` are accepted.
  - `test_config_loads`: `cfg.retrieval.settings["S3"].k == 4`, `.rerank is True`, `.tag == "S3:c200-k4-rr"`.
  - `test_split`: a trace with spans wait 100 ms, work 50 ms, retry 30 ms and first line at 120 ms gives `wait_ms == 100`, `retry_ms == 30`, `before_first_ms == 120`.
- [ ] **Step 4:** `uv run pytest -q` → fails on missing modules.
- [ ] **Step 5:** Implement. `replace_scenario` rewrites the file atomically (temp file + rename).
- [ ] **Step 6:** `uv run pytest -q` → all pass. Commit `chore: scaffold, config, results writer, traces`.

---

### Task 1: Fetch and pin the papers

**Files:** Create `src/relay/ingest/fetch.py`; Test `tests/test_fetch.py`

**Interfaces:**
- Produces: `SOURCES: dict[Paper, dict]` with keys `arxiv_id`, `version`, `title`, `short`; `fetch_all(dest=Path("data/papers")) -> dict` writing `{paper}.html`, `{paper}.pdf`, `manifest.json` (`url`, `sha256`, `bytes`, `fetched_at` per file).

| paper | arxiv_id | version | short |
|---|---|---|---|
| attention | 1706.03762 | v7 | Attention |
| moshi | 2410.00037 | v1 | Moshi |
| hnet | 2507.07955 | v2 | H-Net |

- [ ] **Step 1:** Tests: `test_urls_are_version_pinned` (each URL ends with the exact id+version, hosts are `arxiv.org/html/` and `arxiv.org/pdf/`); `test_manifest_roundtrip` with a fake transport.
- [ ] **Step 2:** Implement with `httpx`, `User-Agent: relay-fetch/0.1`, `follow_redirects=True`, 3 s sleep between requests, skip files already present with matching sha256.
- [ ] **Step 3:** `uv run relay fetch` then verify: each HTML `<title>` contains the expected paper title; `pymupdf.open(pdf).page_count` is 15 for attention and > 0 for the others.
- [ ] **Step 4:** Commit `feat: fetch papers at pinned versions`.

---

### Task 2: Parse HTML, audit against the PDF, patch gaps

**Files:** Create `src/relay/ingest/parse_html.py`, `src/relay/ingest/audit_pdf.py`, `tests/fixtures/mini_paper.html`, `data/patches/`; Test `tests/test_parse_html.py`, `tests/test_audit_pdf.py`

**Interfaces:**
- Produces:
```python
@dataclass
class Block:
    paper: Paper; section_num: str; section_title: str
    kind: Literal["abstract", "text", "table", "figure_caption"]
    label: str | None; text: str; order: int; page: int | None = None
def parse_paper(html: str, paper: Paper, patches: dict[str, str] | None = None) -> tuple[list[Block], list[str]]
    # second value: labels of tables with no body and no patch, e.g. ["Table 5"]
def pdf_pages(path: Path) -> list[str]
def coverage_by_page(pages: list[str], corpus_text: str, n: int = 5) -> list[float]
def map_pages(blocks: list[Block], pages: list[str]) -> None   # sets block.page or leaves None
```

Parsing rules (LaTeXML class names):
- Replace every `<math>` with `$` + its `alttext` + `$`.
- Drop `section.ltx_bibliography`, nav, footer, author block. Keep `div.ltx_abstract` as section_num `"0"`, title `Abstract`.
- Sections are `section.ltx_section | ltx_appendix | ltx_subsection | ltx_subsubsection`. Number = heading's `span.ltx_tag` text, stripped, with a leading `Appendix ` removed. Title = heading text minus the tag. Unnumbered headings get `section_num ""`.
- Walk each section's direct children in order: `div.ltx_para` → text; lists → one item per line; `table.ltx_equation*` → `$$alttext$$` appended to the previous text block; `figure.ltx_table` → table; `figure.ltx_figure` → caption only; nested section → recurse.
- Footnotes (`.ltx_note`): replace inline with ` (footnote: …)` using `.ltx_note_content` text without the mark.
- Table: `label` from the caption tag (`Table 3`); text = caption line, then one line per row, cells joined by ` | `. If the figure has no `td`/`th`: use `patches[f"{paper}_{label}"]` when present, otherwise text = caption + ` [TABLE BODY MISSING]` and add the label to the gaps list.
- Patches are `data/patches/{paper}_table{N}.md`; first line `verified: true|false`, rest is the table as pipe rows.

Audit rules:
- Normalise both sides: lowercase, join hyphenated line breaks, keep alphanumeric tokens.
- `coverage_by_page` = share of a page's word 5-grams that occur in the HTML-derived corpus text.
- `map_pages`: for each block take up to three 5-grams with no `$`; the page containing most of them wins; no hit leaves `None`.

- [ ] **Step 1:** Build `mini_paper.html` by hand with: one numbered section, one subsection, one appendix, a paragraph containing `<math alttext="d_{model}=512">`, a footnote, a real table, a table with only a caption, a bibliography.
- [ ] **Step 2:** Failing tests:
  - `test_sections`: numbers `["0","1","1.1","A"]`, appendix title without the word "Appendix".
  - `test_math_alttext`: paragraph text contains `$d_{model}=512$`.
  - `test_table_rows`: table block has `label == "Table 1"` and a line `a | b | c`.
  - `test_empty_table_flagged`: gaps `== ["Table 2"]`; with a patch supplied gaps `== []` and the patch rows are in the block text.
  - `test_bibliography_dropped`: no block contains a reference title from the fixture.
  - `test_coverage_and_page_map`: a two-page fake PDF text maps a block to page 2; a page of unrelated text has coverage `< 0.1`.
- [ ] **Step 3:** Implement. PDF text via `pymupdf.open(path)` and `page.get_text("text", sort=True)` only; if the 2.x API differs, read the installed package's docs rather than guessing.
- [ ] **Step 4:** `uv run relay ingest` writes `data/corpus/blocks.jsonl` and `data/corpus/audit.json`. Acceptance on real data:
  - attention: 4 tables, 0 gaps. moshi: 19 tables, appendices A to F, 0 gaps. hnet: 8 tables, appendices A to F, gaps exactly `["Table 5","Table 7"]`.
  - Print every non-reference PDF page with coverage < 0.5 for inspection.
- [ ] **Step 5:** Draft `data/patches/hnet_table5.md` and `hnet_table7.md` from the PDF page text, `verified: false`. Re-run: gaps empty.
- [ ] **Step 6 [KAMAL]:** Compare both patches and each low-coverage page against the PDF. Fix, set `verified: true`. `relay ingest` warns while any patch is unverified.
- [ ] **Step 7:** Commit `feat: html parse with pdf audit and table patches`.

---

### Task 3: Chunk, index, re-rank, the retrieval tool

**Files:** Create `src/relay/ingest/chunk.py`, `src/relay/retrieve/{index,rerank,tool}.py`; Test `tests/test_chunk.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `Block` (Task 2), `Chunk`, `Hit`, `RetrievalSetting`.
- Produces:
```python
def count_tokens(text: str) -> int                       # tiktoken o200k_base
def chunk_blocks(blocks: list[Block], chunk_tokens: int) -> list[Chunk]
class Index:
    @classmethod
    def build(cls, chunks: list[Chunk]) -> "Index"
    def save(self, d: Path) -> None
    @classmethod
    def load(cls, d: Path) -> "Index"
    def search(self, query: str, papers: set[Paper] | None, n: int) -> list[Hit]
class Reranker:
    def score(self, query: str, texts: list[str]) -> list[float]
@dataclass
class ToolResult: hits: list[Hit]; timed_out: bool; ms: float
class PaperSearch:                                        # the pipeline's tool
    def __init__(self, corpus_dir: Path, reranker: Reranker | None = None): ...   # loads one Index per chunk size
    async def search_papers(self, queries: list[str], papers: list[Paper] | None, setting: RetrievalSetting,
                            *, question: str, timeout_s: float, per_paper: bool = False,
                            k: int | None = None) -> ToolResult
    def chunks_by_id(self, chunk_tokens: int) -> dict[str, Chunk]
```

Rules:
- Chunks never cross a section boundary. Paragraphs are packed greedily up to `chunk_tokens`; an over-long paragraph is split at sentence ends. A table is one chunk; a table over `2 × chunk_tokens` is split by rows, with caption and first row repeated. Figure captions are their own chunks. No overlap.
- Indexed text and prompt text are both `chunk.header + "\n" + chunk.text`.
- BM25 tokeniser: lowercase, regex `[a-z0-9_]+(?:[.\-][a-z0-9_]+)*` (keeps `41.8`, `h-net`, `d_model`).
- Dense: fastembed `TextEmbedding("BAAI/bge-small-en-v1.5")`, L2-normalised float32. Use `query_embed` for queries if the method exists, else `embed`.
- `Index.search`: BM25 top-n and dense top-n, fused by RRF with constant 60, after the paper filter.
- `search_papers`: run every query, keep each chunk's best fused score, take `setting.candidates`; if `setting.rerank`, score `(question, text)` pairs in a thread and sort; return top `k`. If re-ranking is not done by `timeout_s`, return the fused order with `timed_out=True`. With `per_paper=True`, split `k` evenly across `papers` (at least 1 each).
- Index directories: `data/corpus/c{120,200,400}/`.

- [ ] **Step 1:** Failing tests on a six-chunk toy corpus:
  - `test_no_cross_section`, `test_table_atomic`, `test_sizes_respected` (no text chunk exceeds `chunk_tokens` by more than one sentence).
  - `test_paper_filter`: `papers={"moshi"}` returns only Moshi chunks.
  - `test_rrf_deterministic`: same query twice gives the same order.
  - `test_rerank_timeout`: with a fake re-ranker sleeping 2 s and `timeout_s=0.05`, result has `timed_out is True` and equals the fused order.
  - `test_per_paper_quota`: `k=3`, three papers, one hit from each.
- [ ] **Step 2:** Implement; run tests.
- [ ] **Step 3:** `uv run relay index build` prints chunk count and token total per size. Sanity: `uv run relay search "BLEU of the big model on English-to-German" --setting S3` lists an Attention Table 2 chunk in the top 3.
- [ ] **Step 4:** Commit `feat: chunking, hybrid index, rerank, search tool`.

#### Task 3b: Paper map (second tool, no model calls)

**Files:** Create `src/relay/retrieve/papermap.py`; Test `tests/test_papermap.py`

**Interfaces:**
- Consumes: `Block` (Task 2), `Chunk`, `Hit`, `PaperSearch.chunks_by_id`.
- Produces:
```python
@dataclass
class MapEntry:
    paper: Paper; kind: Literal["section", "table", "figure", "symbol", "term"]
    key: str                 # normalised: "3.2.1", "table 3", "figure 2", "d_model", "inner monologue"
    display: str             # "§3.2.1 Scaled Dot-Product Attention", "Table 3", "$d_{model}$"
    section_num: str; snippet: str   # ≤ 120 chars of block text used to find the chunk
class PaperMap:
    @classmethod
    def build(cls, blocks: list[Block]) -> "PaperMap"
    def save(self, path: Path) -> None          # data/corpus/papermap.json
    @classmethod
    def load(cls, path: Path) -> "PaperMap"
    def lookup(self, question: str, papers: list[Paper] | None) -> list[MapEntry]
    def resolve(self, entries: list[MapEntry], chunks: dict[str, Chunk], max_hits: int) -> list[Hit]
def normalise_symbol(s: str) -> str             # "$d_{\text{model}}$" -> "d_model"
```

Build rules (all deterministic):
- Sections: every `(paper, section_num, section_title)` from blocks; keys `"3.2.1"` and `"appendix a"` for appendices.
- Tables and figures: every block with a `label`; key = label lowercased; snippet = caption start.
- Symbols: in text blocks, a math span whose alttext is ≤ 20 chars, followed within 60 chars by one of `denote|denotes|is the|are the|is defined as|represents|refers to`, or preceded within 20 chars by `where|let`. The first such occurrence per `(paper, symbol)` wins.
- Terms: phrases captured by `we (call|refer to .{0,40} as|term|name) "?([A-Z]?[\w\- ]{3,40})"?` and by bold/italic `<em>` spans in the sentence that introduces them. First occurrence per `(paper, term)` wins.

Lookup rules:
- References: regex `\b(table|tab\.|figure|fig\.)\s*(\d+)\b`, `\b(section|sec\.|§)\s*([\d.]+|[A-F])\b`, `\bappendix\s+([A-F])\b`.
- Symbols: question tokens containing `_`, `^`, `\`, a Greek letter, or inside backticks or `$…$`, normalised and matched against symbol keys. Terms: exact match of a term key inside the lowercased question.
- `papers` restricts entries. With no paper named, return the entry from every paper that has it (all three papers have a Table 3; the reply must say whose).
- `resolve` finds, for each entry, the chunk of the current chunk size with the same paper and section whose text contains the snippet (tables: same label). Unresolved entries are dropped and logged.

- [ ] **Step 1:** Failing tests on `mini_paper.html`: `test_table_entry` (`lookup("What is in Table 1?")` → one table entry); `test_symbol_definition` (fixture sentence "where $d_{model}$ denotes the model dimension" → key `d_model`, its section); `test_term_definition` (fixture "we call this the relay layer" → key `relay layer`); `test_section_and_appendix`; `test_no_false_hits` ("How fast is training?" → `[]`); `test_ambiguous_table_number` (three-paper fixture, "Table 3" with no paper → three entries, one per paper); `test_resolve_finds_chunk` at chunk sizes 120 and 400.
- [ ] **Step 2:** Implement. `relay ingest` also writes `data/corpus/papermap.json` and prints counts per kind per paper.
- [ ] **Step 3:** Sanity on real data: attention has 4 tables and a `d_model` symbol entry in §3; moshi has a term entry for `inner monologue`; hnet has 8 table entries including the patched Tables 5 and 7. Print 20 random glossary entries for Kamal to eyeball; a wrong definition here is a wrong citation later.
- [ ] **Step 4:** Commit `feat: paper map tool (sections, tables, glossary)`.

---

### Task 4: Model client (raw, timed, no retries)

**Files:** Create `src/relay/llm/client.py`; Test `tests/test_client.py`

**Interfaces:**
- Produces:
```python
@dataclass
class LLMRequest:
    model: str; messages: list[dict]; max_completion_tokens: int
    reasoning_effort: str | None = "low"; temperature: float = 0.2
    json_mode: bool = False; seed: int | None = 7; trace: str | None = None
@dataclass
class Delta: kind: Literal["reasoning", "content"]; text: str; t: float
@dataclass
class CallRecord:
    model: str; status: int; t_send: float; t_headers: float | None
    t_first_reasoning: float | None; t_first_content: float | None; t_done: float | None
    usage: dict | None; usage_estimated: bool; headers: dict[str, str]; error_body: str | None
ErrorKind = Literal["rate_limited_minute", "rate_limited_day", "server", "timeout_ttft",
                    "timeout_total", "connection", "midstream", "bad_request"]
class LLMError(Exception): kind: ErrorKind; retry_after_s: float | None; record: CallRecord
def parse_duration(s: str) -> float                       # "2m59.56s" -> 179.56, "120ms" -> 0.12
def classify_429(body: str, retry_after_s: float | None) -> ErrorKind
def redact(headers: Mapping[str, str]) -> dict[str, str]   # drops authorization, cookies
class LLMClient:
    def __init__(self, base_url: str, api_key: str, source: Literal["real", "stand-in"]): ...
    async def stream(self, req: LLMRequest, *, ttft_timeout_s: float, total_timeout_s: float,
                     on_delta: Callable[[Delta], None]) -> CallRecord    # raises LLMError
```

Rules:
- POST `{base_url}/chat/completions` with `stream: true`, `stream_options: {"include_usage": true}`, `max_completion_tokens`, `temperature`, `seed`; for gpt-oss models add `reasoning_effort` and `include_reasoning: true`; `response_format: {"type": "json_object"}` when `json_mode`.
- Parse SSE lines starting `data: `; stop at `[DONE]`. Text is `choices[0].delta.content`; reasoning is `choices[0].delta.reasoning`. Take usage from `chunk["usage"]` or `chunk["x_groq"]["usage"]`, whichever appears.
- No usage by the end: estimate with `count_tokens` and set `usage_estimated=True`.
- `classify_429`: `rate_limited_day` if the body mentions `per day`, `(RPD)` or `(TPD)`, or if `retry_after_s > 300`; otherwise `rate_limited_minute`.
- `trace` is sent as header `x-relay-trace` only when `source == "stand-in"`.
- Every real call appends one line to `raw/real_calls.jsonl`: model, timestamps, status, usage, redacted headers.
- One shared `httpx.AsyncClient`. No retries here.

- [ ] **Step 1:** Failing tests with `httpx.MockTransport` / a tiny ASGI app:
  - `test_parse_duration`: `"2m59.56s"→179.56`, `"7.66s"→7.66`, `"120ms"→0.12`, `"1h2m3s"→3723.0`.
  - `test_stream_content_and_reasoning`: deltas arrive in order; `t_first_reasoning < t_first_content`.
  - `test_usage_both_locations`: found under `usage` and under `x_groq.usage`.
  - `test_429_minute`, `test_429_day`, `test_429_unparseable` (body `"slow down"`, `retry-after: 900` → `rate_limited_day`; no header → `rate_limited_minute`).
  - `test_500_is_server`, `test_ttft_timeout`, `test_midstream_cut` (stream ends without `[DONE]` → `midstream`).
  - `test_redact`: no `authorization` key survives; the ledger file never contains the fake key.
- [ ] **Step 2:** Implement; run tests.
- [ ] **Step 3:** Commit `feat: streaming model client with raw timings`.

---

### Task 5: Probe the real service `[REAL]`

**Files:** Create `src/relay/probe.py`; outputs `raw/probe/<name>.jsonl`; Test `tests/test_probe_plan.py`

**Interfaces:**
- Consumes: `LLMClient`.
- Produces: `relay probe <name> --real`, `relay budget`; each JSONL row = request params (no key), `CallRecord`, wall-clock time.

Before any step: Kamal creates the key, writes `.env`, and notes RPM/TPM/RPD/TPD from the console limits page `[KAMAL]`.

| Probe | Model | Requests | Est. tokens | Question it answers |
|---|---|---|---|---|
| P0 `models` | none | 1 GET `/models` | 0 | Are the three ids live? |
| P1 `latency` | 120b | 30, one every 8 s; 10 each at ~300, ~1200, ~2400 prompt tokens, ~150 out | 44K | First-token and total-time spread; slope against prompt size; `queue_time` |
| P1b `latency-small` | 20b | 20, ~350 in / ~100 out, one every 4 s | 10K | Understand-step latency |
| P2 `concurrency` | 20b | bursts of 1, 2, 4, 8 at once, 3 rounds, 30 s apart, `max_completion_tokens: 16` | 5K | Does first-token time grow with requests in flight? Any cap? |
| P3a `rpm-edge` | qwen | 40 back to back, tiny, `max_completion_tokens: 1` | 2K | Index of the first 429, its body and headers, `retry-after`; do rejected calls lower `remaining-requests`? Then: wait `retry-after`, fire 5 at once, count successes (≥ 4 → window reset, ≤ 2 → continuous refill) |
| P3b `tpm-edge` | qwen | ~2.5K-token prompts with `max_completion_tokens: 1000` until 429 | 10K | Does "Requested" in the message include the output cap? |
| P4 `cache` | 120b | 6 calls sharing a 1200-token prefix, then 6 sharing a 400-token prefix, different tails | 12K | `cached_tokens`; drop in `x-ratelimit-remaining-tokens` per call with and without a hit |
| P5 `rpd-burn` | qwen | 1 tiny call every 2.1 s until a daily 429 (~35 min), then one call every 10 min for 1 h, then one the next morning | ~10K | Daily-limit body and `retry-after`; rolling or fixed reset. Needs G1 |
| P6 `effort` | 120b | 5 pipeline-shaped prompts × {low, medium} | 20K | Reasoning tokens and time to first content per effort level |

- [ ] **Step 1:** Test `test_probe_plans_fit_budget`: each probe's planned token total ≤ the table value and its pacing never exceeds 30 requests or 8000 tokens in any 60 s window (except P3a/P3b, which are meant to).
- [ ] **Step 2:** Implement probes and `relay budget` (sums the ledger over the trailing 24 h per model against `limits_prior`).
- [ ] **Step 3 [REAL]:** Run P0, then one single P1 call. Confirm in the raw row: streaming works, where usage sits, whether `stream_options` and `include_reasoning` are accepted. If either is rejected with 400, remove that field in `client.py`, note it, re-run the Task 4 tests.
- [ ] **Step 4 [REAL]:** Run P1, P1b, P2, P4, P6. Then P3a, P3b on qwen.
- [ ] **Step 5 [REAL] [KAMAL decides G1]:** Start P5 in the background.
- [ ] **Step 6:** `relay probe summarize` prints: per model p50/p95/p99 of first content token and total; reasoning share of output tokens; the first-429 index; the verbatim 429 bodies; cache effect; the refill verdict. Fill a Table 1 draft with a "how found" and a confidence per row.
- [ ] **Step 7 [KAMAL]:** Read the summary. Set `slo` and `ladder` numbers in `relay.yaml` (G2). Commit raw probe files and `feat: real-service probe`.

---

### Task 6: Governor and retry policy

**Files:** Create `src/relay/llm/governor.py`, `src/relay/llm/retry.py`; Test `tests/test_governor.py`, `tests/test_retry.py`

**Interfaces:**
- Consumes: `CallRecord`, `LLMError`, `parse_duration`.
- Produces:
```python
class Bucket:
    def __init__(self, capacity: float, refill_per_s: float, now: float): ...
    def available_at(self, amount: float, now: float) -> float     # absolute time; inf if amount > capacity
    def take(self, amount: float, now: float) -> None
    def refund(self, amount: float, now: float) -> None
    def resize(self, capacity: float, now: float) -> None           # keeps level ≤ new capacity
    def clamp(self, level: float, now: float) -> None               # level = min(level, given)
    def frac_left(self, now: float) -> float
@dataclass
class Grant: model: str; tokens: int; t: float
class WouldMissDeadline(Exception): wait_s: float
class DailyExhausted(Exception): reset_in_s: float
class ModelGovernor:
    def __init__(self, model: str, limits: Limits, clock: Callable[[], float], safety: float = 0.9): ...
    def wait_s(self, tokens: int) -> float                          # inf when a daily bucket cannot cover it
    def daily_frac_left(self) -> float
    async def acquire(self, tokens: int, deadline: float) -> Grant  # FIFO; raises the two errors above
    def on_response(self, grant: Grant, rec: CallRecord) -> None
    def on_error(self, grant: Grant, err: LLMError) -> None
    def release(self, grant: Grant) -> None                         # cancelled before sending
    queue_len: int
class Governor:
    def for_model(self, model_id: str) -> ModelGovernor
    @classmethod
    def from_ledger(cls, cfg: Config, ledger: Path | None, clock) -> "Governor"

class RetryPolicy:
    def __init__(self, cfg: RetryCfg, rng: random.Random, clock): ...
    def note_request(self) -> None
    def decide(self, err: LLMError, attempt: int, sent_any: bool) -> float | None   # backoff seconds, None = stop
```

Rules:
- Buckets per model: RPM (`rpm`, refill `rpm/60`), TPM, RPD (refill `rpd/86400`), TPD. All capacities × `safety`. If Task 5 showed fixed windows, switch refill to window reset behind one config flag `governor.mode: bucket|window`.
- `acquire(tokens)` takes 1 from RPM and RPD and `tokens` from TPM and TPD. `tokens` = estimated prompt tokens + `max_completion_tokens` (drop the cap from the reservation if P3b showed it is not counted).
- `on_response`: `resize` TPM to `x-ratelimit-limit-tokens × safety` and RPD to `x-ratelimit-limit-requests × safety` when they differ; `clamp` TPM and RPD to the `remaining` headers; refund `reserved − actual` where actual = `total_tokens − cached_tokens`.
- `on_error` minute: block the model until `now + retry_after_s` and empty the bucket named in the message. Day: empty the daily buckets, remember `reset_in_s`. Server/connection/timeouts: refund the reservation only if the call returned a server error status (not charged); otherwise keep it spent.
- `acquire` never sleeps past `deadline`: raise `WouldMissDeadline(wait)` at once, and again at dequeue if the deadline has passed.
- `RetryPolicy.decide` (`budgeted`): `None` if `sent_any`, if kind ∈ {`rate_limited_day`, `bad_request`, `midstream`}, if `attempt ≥ max_attempts`, or if retries in the trailing 60 s exceed `budget_ratio ×` requests in the trailing 60 s. `rate_limited_minute` → `retry_after_s`. Otherwise `uniform(0, min(cap_s, base_s · 2**attempt))`.
- `naive` policy: returns `0.0` for every kind until `attempt ≥ 3`.

- [ ] **Step 1:** Failing tests with a fake clock:
  - `test_refill`: capacity 8000, refill 8000/60, take 6000 → `available_at(4000) == now + 15`.
  - `test_resize_on_header`: TPM limit header 4000 after 8000 → the same request's `wait_s` roughly doubles.
  - `test_clamp_to_remaining`: header remaining 500 with local level 3000 → level 500.
  - `test_refund_uses_uncached`: reserved 2000, usage total 1500 with 400 cached → 900 refunded.
  - `test_deadline_no_sleep`: `acquire` with wait 20 s and deadline in 5 s raises `WouldMissDeadline` without advancing the clock.
  - `test_daily_exhausted`: after a day-kind error, `wait_s` is `inf` and `acquire` raises `DailyExhausted`.
  - `test_minute_block`: after a minute-kind error with `retry_after_s=7`, `wait_s ≥ 7`.
  - `test_fifo`, `test_release_returns_tokens`.
  - `test_ledger_rebuild`: a ledger with 150K tokens in the last 24 h gives `daily_frac_left ≈ 0.17` (with safety 0.9).
  - Retry: `test_no_retry_after_send`, `test_no_retry_daily`, `test_backoff_bounds`, `test_budget_stops_retries`, `test_naive_retries_429`.
- [ ] **Step 2:** Implement; run tests. Commit `feat: governor and retry policy`.

---

### Task 7: The stand-in service and its calibration

**Files:** Create `src/relay/standin/{server,limits,latency,replay,calibrate}.py`; Test `tests/test_standin.py`, `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `Bucket` (reuse from Task 6), probe JSONL (Task 5), `count_tokens`.
- Produces:
  - `create_app(cfg: StandinConfig) -> FastAPI` with `POST /openai/v1/chat/completions`, `GET /openai/v1/models`, `POST /admin/config` (deep-merge patch), `GET /admin/stats`.
  - `relay standin serve --port 8900`.
  - `fit_profile(probe_dir: Path) -> dict` and `relay calibrate` writing `config/standin_profile.json`.

```python
@dataclass
class StandinConfig:
    limits: dict[str, Limits]            # per model id; scale applied by the caller
    latency: dict[str, LatencyProfile]   # ttft_base_ms, ttft_per_1k_prompt_ms, ttft_sigma,
                                         # tps_mu, tps_sigma, reasoning_tokens_mu, p_slow, slow_factor
    load: LoadCfg                        # c0, alpha:  factor = 1 + alpha * in_flight / c0
    faults: FaultCfg                     # p_500, p_503, p_hang, p_midstream_cut, p_slow_success, slow_success_factor
    time_scale: float = 1.0              # tests use 0.01
    seed: int = 7
```

Behaviour per request, in order:
1. Limits. Same four buckets and the same reservation rule as the governor, no safety factor. On refusal: 429 with the body, `retry-after` and headers captured in Task 5, copied field for field. Until the probe is in, use this shape:
   `{"error": {"message": "Rate limit reached for model `<id>` in organization `org_standin` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7400, Requested 1800. Please try again in 9s.", "type": "tokens", "code": "rate_limit_exceeded"}}`
2. Faults, by seeded draw: 500, 503, hang (no bytes), slow success (all delays × `slow_success_factor`), midstream cut (stop after a random number of chunks, no `[DONE]`).
3. Timing: `ttft = (ttft_base_ms + ttft_per_1k_prompt_ms · prompt_tokens/1000) · lognormal(0, ttft_sigma) · (1 + alpha · in_flight/c0)`. Then a silent reasoning phase of `reasoning_tokens / tps`. Then content in chunks of 4 tokens at `tps ~ lognormal(tps_mu, tps_sigma)`.
4. Content: look up `x-relay-trace` (`stage=draft;qid=q07;setting=S3;model=120b;attempt=0`) in `raw/recordings/`. Miss on a draft: one line citing the first `[A|M|H]\d+` id found in the prompt, with no numbers. Miss on understand: `{"in_scope": true, "papers": ["attention","moshi","hnet"], "ambiguous": false, "queries": []}`.
5. All 200 responses carry the six `x-ratelimit-*` headers. The last stream chunk carries `usage` in the location Task 5 observed, with `queue_time`, `prompt_time`, `completion_time`.

Calibration (`fit_profile`):
- `ttft_base_ms`, `ttft_per_1k_prompt_ms`: least-squares line of first-content time against prompt tokens (P1). `ttft_sigma`: std of log residual ratios.
- `tps_mu`, `tps_sigma`: mean and std of `log(completion_tokens / (t_done − t_first_content))`.
- `p_slow`: share of calls slower than 3× the median; `slow_factor`: their median ratio.
- `alpha`: slope of median first-token time against requests in flight (P2), normalised by the single-request median; 0 if the slope is within noise, and say so in the output.
- `relay calibrate` also prints real against refitted-sample percentiles side by side.

- [ ] **Step 1:** Failing tests (ASGI transport, `time_scale=0.01`):
  - `test_stream_shape`: chunks then `[DONE]`; final usage present; six rate-limit headers present.
  - `test_rpm_refusal`: `rpm=3`, four quick calls → fourth is 429 with `retry-after` and `error.code == "rate_limit_exceeded"`.
  - `test_tpm_refusal`: a call needing more than remaining TPM → 429 mentioning `(TPM)`.
  - `test_daily_refusal`: `rpd=2` → third call 429 mentioning `(RPD)`, `retry-after > 300`.
  - `test_admin_halves_limit`: after `POST /admin/config` with `tpm: 4000`, the next response's `x-ratelimit-limit-tokens == "4000"`.
  - `test_faults`: `p_500=1` → 500; `p_midstream_cut=1` → no `[DONE]`; `p_hang=1` → client `timeout_ttft`.
  - `test_slower_under_load`: with `alpha=1, c0=1`, mean first-token time at 8 in flight > at 1.
  - `test_replay_hit_and_miss`.
  - `test_fit_recovers_parameters`: synthetic lognormal samples → fitted sigma within 15%.
- [ ] **Step 2:** Implement; run tests.
- [ ] **Step 3:** `uv run relay calibrate` on the Task 5 data. Replace the placeholder 429 body with the captured one. Commit `feat: calibrated stand-in service`.

---

### Task 8: The pipeline

**Files:** Create `src/relay/pipeline/{understand,decide,prompt,gate,extractive,messages,cache,orchestrator}.py`; Test `tests/test_gate.py`, `tests/test_decide.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `PaperSearch`, `LLMClient`, `Governor`, `RetryPolicy`, `Trace`, shared types.
- Produces:
```python
@dataclass
class Understanding: in_scope: bool; papers: list[Paper]; ambiguous: bool; queries: list[str]; source: Literal["llm", "heuristic"]
def heuristic_understand(question: str) -> Understanding
async def llm_understand(question: str, client, gov, cfg, trace) -> Understanding    # raises on any failure

def choose_level(wait_s: Callable[[str, int], float], daily_frac: Callable[[str], float],
                 tokens_full: int, tokens_lean: int, patience_left_s: float, cfg) -> tuple[Level, str | None]

def build_draft_messages(question: str, hits: list[Hit]) -> list[dict]
STATIC_PREFIX: str

@dataclass
class LineVerdict: line: str; ok: bool; reason: Literal["ok", "no_citation", "unknown_id", "number_not_in_source", "sentinel"]
class LineGate:
    def __init__(self, hits: list[Hit], question: str): ...
    def feed(self, text: str) -> list[LineVerdict]     # verdicts for lines completed by this delta
    def flush(self) -> list[LineVerdict]

def extractive_reply(question: str, hits: list[Hit], reason: Literal["minute", "day"], resume_in_s: float | None) -> str
class AnswerCache:
    async def get_or_run(self, key: str, run: Callable[[], Awaitable[Reply]]) -> tuple[Reply, bool]   # bool = served from cache/in-flight

class Relay:
    def __init__(self, cfg: Config, client: LLMClient, gov: Governor, search: PaperSearch,
                 source: Literal["real", "stand-in"]): ...
    async def ask(self, question: str, *, request_id: str, arrival_t: float | None = None,
                  setting: RetrievalSetting | None = None, force_level: Level | None = None,
                  cache_key: str | None = None, trace_tags: dict | None = None,
                  on_event: Callable[[OutEvent], None] | None = None) -> Reply
    # also appends the request's trace to raw/traces/<scenario>.jsonl when trace_tags has "scenario"
```

Rules:
- **Trace header.** Every model call sets `LLMRequest.trace` to `stage=<understand|draft>;qid=<trace_tags["qid"]>;setting=<setting name>;model=<120b|20b>;attempt=<n>`.
- **Understand.** LLM call on the fallback model, JSON mode, schema `{"in_scope": bool, "papers": [...], "ambiguous": bool, "queries": [≤3 strings]}`, hard `understand.timeout_s`. Skip the LLM call when the system is already under pressure (`wait_s(primary, tokens_full) > wait_full_s`, or the fallback model has any wait). Skipped, or any error, timeout or bad JSON → `heuristic_understand`: alias table (`moshi|mimi|helium|inner monologue|kyutai` → moshi; `h-net|hnet|dynamic chunking|byte-level` → hnet; `transformer|attention is all you need|bleu|wmt` → attention), `queries=[question]`, all papers when no alias matches, `ambiguous = no alias matched`.
- **Retrieve.** `search.search_papers(queries or [question], papers, setting, question=question, timeout_s=tool_timeout_s, per_paper=ambiguous)`. Then, if `papermap.enabled`: `papermap.resolve(papermap.lookup(question, papers), search.chunks_by_id(setting.chunk_tokens), max_hits)`. Map hits go first; duplicates are removed; the list is cut back to `k`, so map hits replace the lowest-ranked search hits. The trace records which hits came from the map.
- **Decide.**
  - Refuse early (no draft call) only when `in_scope is False` **and** `refuse_below` is set **and** the top score is below it.
  - `choose_level`, in order: (1) primary usable = `daily_frac(primary) > daily_stop_below`; (2) usable, `daily_frac > daily_lean_below`, `wait_s(primary, tokens_full) ≤ wait_full_s` → FULL; (3) usable and `wait_s(primary, tokens_lean) ≤ wait_lean_s` → LEAN; (4) fallback usable and `wait_s(fallback, tokens_lean) ≤ wait_lean_s` → FALLBACK; (5) the shortest lean wait among usable models `≤ patience_left_s − 1.5` → that model (LEAN or FALLBACK); (6) EXTRACTIVE with `None`.
  - LEAN and FALLBACK re-run retrieval with the lean setting (cheap, local).
- **Draft prompt.** One user message: `STATIC_PREFIX`, then `PASSAGES`, one `[id] (header) text` per passage, then `QUESTION: …`, then `ANSWER`. Static text first so the prefix can be cached.

```
You answer questions about three research papers using ONLY the numbered passages below.
Rules:
1. Use only facts stated in the passages. Do not use what you remember about these papers: this library pins specific versions and your memory may be of another one.
2. Write at most 4 short sentences, each on its own line.
3. End every line with the id(s) of the passage(s) that support it in square brackets, e.g. [M12] or [A3, A7].
4. Copy numbers, names and units exactly as they appear in the passage.
5. If the passages do not state the answer, output exactly NOT_IN_PAPERS on the first line and nothing else.
6. If the question names no paper and the passages give different answers for different papers, write one line per paper and name the paper in each line.
7. A passage describing someone else's system (a baseline, related work) is not a fact about the paper's own model. Say whose result a number is.

Example
PASSAGES
[X1] (Example Paper | §2 Method) The model uses 6 layers and a hidden size of 512.
[X2] (Example Paper | §4 Results | Table 1) Model | Score
Ours | 28.4
Baseline-B | 25.1
QUESTION: What score does the model reach, and what does the baseline reach?
ANSWER
The paper's own model reaches a score of 28.4 [X2]
Baseline-B reaches 25.1 [X2]

Example
PASSAGES
[X1] (Example Paper | §2 Method) The model uses 6 layers and a hidden size of 512.
QUESTION: Which optimizer was used?
ANSWER
NOT_IN_PAPERS
```

- **Gate.** Unit = one line. First line equal to `NOT_IN_PAPERS` → `sentinel`. Otherwise: ids from the trailing `\[[A-Z]\d+(?:\s*,\s*[A-Z]\d+)*\]`; none → `no_citation`; any id not in `hits` → `unknown_id`; every number (`\d+(?:[.,]\d+)*`, thousands commas removed) outside the brackets must occur in a cited chunk's header + text or in the question, else `number_not_in_source`.
- **Deliver.** Emit `status` at once ("Searching the three papers…"); it does not count as first token. Each `ok` line is emitted as `sentence` with ids replaced by `chunk.cite`. `first_token_ms` = first `sentence`, `notice` or refusal text, measured from `arrival_t`.
  - `sentinel` → templated refusal from `messages.py` listing the closest sections searched; `refused=True`; model text is not shown.
  - Line 1 not ok, nothing sent: at FULL with re-drafts left → retrieve again with `papers=None`, `k × 2`, draft again (`attempt=1`); otherwise EXTRACTIVE.
  - Later line not ok: cancel the stream, emit `notice` "I left out one statement I could not verify against the papers."
  - `midstream` error after lines were sent: same notice, `status="ok"`.
- **Errors.** Around every model call: `gov.acquire` → `client.stream` → `gov.on_response` / `gov.on_error` → `RetryPolicy.decide`. `WouldMissDeadline` or `DailyExhausted` on the chosen model → re-run `choose_level` once excluding that model. Still nothing → EXTRACTIVE.
- **Outcome mapping.** Verified answer or model refusal → `ok`. EXTRACTIVE because of limits → `rate_limited` (`first_token_ms` and `total_ms` still recorded). Retries exhausted on errors with no reply → `error`. Cancelled by the caller → `timeout`, and `gov.release` / stream cancel must run.
- **Tokens.** `tokens_in`/`tokens_out` = sums over every model call of the request, including failed attempts and the understand call. `retries` = failed attempts that were retried (re-drafts are counted separately in the trace).
- **Cache.** Key = `cache_key` or the question lowercased with whitespace collapsed. Only `ok`, non-truncated replies are stored, for `ttl_s`. A hit returns the stored text with zero tokens and tag `+cache`.
- **Messages** (`messages.py`, defaults; wording is `[KAMAL]`): `STATUS`, `REFUSAL`, `TRUNCATED`, `EXTRACTIVE_MINUTE` ("We're at capacity, so rather than keep you waiting here are the most relevant passages, quoted directly. …"), `EXTRACTIVE_DAY` ("Today's model budget is used up. Written answers resume in about {h} h {m} min. Until then, here is what the papers say, quoted directly: …").
- **CLI.** `relay ask "question" [--real]` prints events as they arrive, then the citations and the level.

- [ ] **Step 1:** Failing gate tests: `test_ok_line`, `test_no_citation`, `test_unknown_id`, `test_number_not_in_source` (line says `512`, chunk says `1024`), `test_number_from_question_allowed`, `test_sentinel_first_line_only`, `test_partial_delta_buffering` (a line split across three deltas yields one verdict), `test_decimal_and_latex` (`3.3 × 10^18` passes against `$3.3\cdot 10^{18}$`), `test_injection_still_gated` (an uncited line "As I remember, the answer is 42" is withheld).
- [ ] **Step 2:** Failing `choose_level` table test covering all six rules, including daily fraction at 0.2 → LEAN and both models stopped → EXTRACTIVE.
- [ ] **Step 3:** Failing pipeline tests against the in-process stand-in with scripted replies:
  - `test_happy_path_events`: `status`, then `sentence`s; `first_token_ms` set; cite strings rendered.
  - `test_first_line_fails_then_redraft`: two draft calls, second succeeds.
  - `test_never_emits_unverified`: both drafts bad → extractive text, no model text in any event.
  - `test_later_line_fails`: one sentence, one notice, stream cancelled.
  - `test_sentinel_refusal`: `refused is True`, templated text.
  - `test_daily_primary_exhausted_uses_fallback`: tag ends `@20b`, `status == "ok"`.
  - `test_all_exhausted_extractive`: `status == "rate_limited"`, text contains the resume time, zero draft calls.
  - `test_understand_timeout_uses_heuristic`, `test_tool_timeout_still_answers`.
  - `test_ambiguous_question_per_paper`: hits include all three papers (Review Focus 2).
  - `test_coalesce_duplicates`: two concurrent identical questions → one draft call, second reply tagged `+cache`.
  - `test_cancel_releases_grant`: cancelling `ask` mid-stream restores the governor's level and closes the stream (Review Focus 4).
  - `test_stream_without_usage`: reply delivered, tokens estimated, trace flags `usage_estimated` (Review Focus 5).
  - `test_midstream_after_first_line`: stream cut with no `[DONE]` after one verified line → that line plus the notice, `status == "ok"`, no retry (Review Focus 5).
  - `test_patched_table_answer`: with a corpus holding a patched table chunk, the reply cites it (Review Focus 1).
  - `test_map_hit_replaces_lowest`: a question naming "Table 1" puts that table chunk first and the hit count still equals `k`.
  - `test_map_disabled`: with `papermap.enabled: false` the hits equal the plain search result.
- [ ] **Step 4:** Implement; run all tests.
- [ ] **Step 5:** `uv run relay standin serve &` then `uv run relay ask "How many layers does the base Transformer encoder have?"` → streamed lines with citations.
- [ ] **Step 6:** Commit `feat: pipeline with line-gated streaming and degrade ladder`.

---

### Task 9: Evaluation, retrieval settings, recordings

**Files:** Create `src/relay/evalx/{questions,grader,recall,run_eval}.py`, `eval/questions.yaml`, `eval/grades_review.csv`; Test `tests/test_grader.py`, `tests/test_questions.py`

**Interfaces:**
- Produces:
```python
@dataclass
class Gold: paper: Paper; section: str; label: str | None = None
@dataclass
class Question:
    id: str; question: str; answerable: bool; category: str
    gold: list[Gold]; must_include: list[list[str]]; note: str = ""
def load_questions(path) -> list[Question]
def validate_questions(qs, blocks) -> list[str]          # human-readable problems
def grade(q: Question, reply: Reply, chunks_by_id: dict[str, Chunk]) -> Grade | None
def recall_table(qs, settings: list[RetrievalSetting], routers: list[str]) -> pd.DataFrame
```
- `relay eval validate`, `relay eval recall`, `relay eval run --setting S1 [--model fallback] [--real] [--limit N]`.

**Step 0 [KAMAL]: write `eval/questions.yaml`** (at least 20; 24 recommended). Read the three papers first. Schema:

```yaml
- id: q01
  question: "..."
  answerable: true
  category: table_number      # see mix below
  gold: [{paper: attention, section: "6.1", label: "Table 2"}]
  must_include: [["28.4"]]     # every group must match; alternatives inside a group
  note: "why a naive system gets this wrong"
```

Suggested mix: 8 numbers from tables or appendices; 3 notation or definition questions (defined once, used far away); 3 that name no paper and differ per paper; 2 where an older arXiv version or common knowledge gives a different value than the pinned version; 4 unanswerable-but-near (topic present, fact absent); 2 unanswerable-and-far; 2 answered only by H-Net Table 5 or 7.

Grading rules:
- Returns `None` when the reply is extractive or absent.
- Not answerable: refused → `refused_correctly`; otherwise `wrong`.
- Answerable: refused → `refused_wrongly`. Otherwise `correct` only if every `must_include` group has an alternative present (case-insensitive, thousands commas removed) **and** some cited chunk matches a gold entry (same paper, and `section_num` starts with `gold.section` or `label == gold.label`). Else `wrong`.
- `eval/grades_review.csv` (`scenario,qid,auto_grade,final_grade,note`) overrides the auto grade wherever `final_grade` is filled.

- [ ] **Step 1:** Failing tests: the full grader matrix (8 cases); `test_validate_catches_missing_string` (a `must_include` value absent from the gold section's text is reported); `test_validate_unanswerable_has_no_gold`; `test_resume_skips_done`.
- [ ] **Step 2:** Implement. `run_eval` asks each question once at `force_level=FULL` (or FALLBACK with `--model fallback`) with the cache off, paced by the governor, and for each: appends the row to `results.csv` under scenario `eval-<setting>` (`eval-S4@20b` for the fallback run), saves the trace, and writes one recording per model call to `raw/recordings/<stage>/<qid>__<setting>__<model>__<attempt>.json` (`content`, `reasoning_tokens`, `usage`), the same key the stand-in reads from `x-relay-trace`. It skips (qid, scenario) pairs already present.
- [ ] **Step 3:** `uv run relay eval validate` → no problems. Fix questions with Kamal if any.
- [ ] **Step 4 (free):** `uv run relay eval recall`. Sweep chunk sizes {120, 200, 400} × k {2, 4, 6, 8} × re-rank {no, yes} × router {heuristic} × paper map {off, on}, plus a naive baseline `N0` (400-token pieces, dense only, k = 4, no paper routing, references kept). Metric: share of answerable questions with a gold chunk in the top k, and passage tokens. Also print the top-score distribution for answerable against unanswerable questions.
- [ ] **Step 5 [REAL]:** One understand pass over all questions on 20b (about 11K tokens); re-run the recall sweep with router `llm`. **[KAMAL decides G3]** and confirms or changes S1 to S4 from the recall table.
- [ ] **Step 6:** Set `retrieval.refuse_below` only if the two score distributions separate with zero answerable questions below the cut; otherwise leave `null` and record that.
- [ ] **Step 7 [REAL]:** `relay eval run --setting S1 --real --limit 3` (smoke), then the same run against the stand-in, then in full: S1 and S2 (Day 2); S3, S4, and `--setting S4 --model fallback` (Day 3). Check `relay budget` before each.
- [ ] **Step 8 [KAMAL]:** Review every row in `eval/grades_review.csv`; fill `final_grade` where the auto grade is wrong. Decide G4 and set `retrieval.default`, `retrieval.lean`, and `capacity.tokens_per_answer_full` (mean primary-model tokens per full answer, cached tokens excluded, from the eval traces; understand tokens sit in the other model's bucket and are not part of it).
- [ ] **Step 9:** From the `N0` sweep, list every question where `N0` misses the gold and the chosen setting hits, with the wrong passage `N0` returned. This is the evidence for "How this library will mislead a naive retriever". Commit `feat: eval harness, graded retrieval settings, recordings`.

Candidate ways the library misleads (each must be backed by a question from Step 9, or dropped):
- Shared words, different meanings: encoder, decoder, tokens, attention, chunk appear in all three papers.
- Facts in tables: numeric rows embed poorly; H-Net Tables 5 and 7 vanish from the HTML.
- Baselines and related work: the closest passage is often about someone else's system.
- Reference lists: both later papers cite the first, and reference titles match queries strongly.
- Notation defined once and used everywhere.
- Same label, three papers: every paper has a "Table 3" and a "Section 3"; "What does Table 3 show?" retrieves whichever embeds closest.
- Topic present, fact absent: a confident-looking passage invites a guess.
- Version drift: the model's memory may hold another version's number.

---

### Task 10: Load shapes

**Files:** Create `src/relay/load/{schedule,runner}.py`, `config/scenarios/*.yaml`; Test `tests/test_schedule.py`, `tests/test_runner.py`

**Interfaces:**
- Produces:
```python
@dataclass
class Arrival: t_s: float; qid: str; variant: int      # variant 0 = shared (duplicate), else unique
def build_schedule(scn: Scenario, qids: list[str], capacity_per_min: float) -> list[Arrival]
async def run_scenario(scn: Scenario, *, real: bool) -> None
```
- `relay load run <scenario> [--real]`.

Scenario file:
```yaml
name: burst
target: stand-in            # or real
seed: 7
limit_scale: 10             # stand-in limits = measured free-tier limits × this
segments:
  - {duration_s: 240, rate_x_capacity: 0.5}
  - {duration_s: 45,  rate_x_capacity: 6.0, dup_fraction: 0.5}
  - {duration_s: 315, rate_x_capacity: 0.5}
standin: {faults: {p_500: 0.01, p_slow_success: 0.03}}
daily_tokens: {}            # e.g. {primary: 90000, fallback: 60000}: absolute stand-in TPD for this run
events: []                  # e.g. [{at_s: 240, scale_limits: {rpm: 0.5, tpm: 0.5}}]
overrides: {}               # e.g. {"cache.enabled": false, "retry.policy": naive}
```

Rules:
- `capacity_per_min = limit_scale × tpm / capacity.tokens_per_answer_full`. Arrival gaps are exponential (Poisson) within each segment, from `random.Random(seed)`.
- Each arrival is launched at its scheduled time whatever the state of earlier ones. `arrival_t` is the scheduled time, so queueing is inside the measured latency. `start_time` is the scheduled wall-clock time.
- `cache_key = f"{qid}#{variant}"`: duplicates share variant 0, all others are unique, so a 24-question pool does not fake cache hits.
- Patience: cancel the task if there is no first token by `patience_first_s` or no completion by `patience_total_s` → `status="timeout"`, `total_ms` = the give-up time.
- The runner posts the scenario's stand-in config (limits × `limit_scale`, `daily_tokens`, faults) to `/admin/config` before start. At each `events[].at_s` it multiplies the current limits of every model by `scale_limits` and posts the absolute values. A fresh governor per run (in-memory ledger for stand-in runs).
- `daily-exhausted` sizing: primary `daily_tokens` = offered rate per min × 3 × `tokens_per_answer_full`; fallback = offered rate per min × 3 × measured lean tokens per answer.
- Rows are graded with the Task 9 grader and written with `replace_scenario`. The schedule is saved to `raw/traces/<scenario>.schedule.json`.

Scenarios (rates as multiples of full-answer capacity; durations chosen for n ≥ 300):

| name | target | shape | Purpose |
|---|---|---|---|
| `steady-real` | real, scale 1 | 0.6× for 360 s (about 17 requests) | Real behaviour at a safe rate `[REAL]` ~30K tokens on 120b |
| `steady-twin` | stand-in, scale 1 | same schedule as `steady-real` | Stand-in validation: compare with the row above |
| `steady` | stand-in ×10 | 0.7× for 600 s | Baseline |
| `burst` | stand-in ×10 | 0.5×, 45 s at 6× with 50% duplicates, 0.5× | Sudden burst, cache on |
| `burst-nocache` | stand-in ×10 | same, `cache.enabled: false` | Cost of the duplicate assumption (G5) |
| `overload` | stand-in ×10 | 4× for 360 s | Sustained overload |
| `overload-naive-retry` | stand-in ×10 | same, `retry.policy: naive`, `p_500: 0.05` in both | Retrying made it worse |
| `minute-halved` | stand-in ×10 | 0.8× for 600 s; RPM and TPM halved at 240 s | Limit changes without notice |
| `daily-exhausted` | stand-in ×10 | 0.8× for 480 s; primary TPD sized to run out near 180 s, fallback near 360 s | What users get when the day is spent |

- [ ] **Step 1:** Failing tests: `test_schedule_deterministic`; `test_poisson_rate` (3600 s at 60/min gives 3600 ± 5%); `test_segments_and_duplicates` (in a `dup_fraction: 0.5` segment about half the arrivals have variant 0); `test_open_loop` (stub pipeline sleeping 5 s: the tenth arrival still starts within 50 ms of its schedule); `test_patience_timeout` (stub never answers: status `timeout`, `total_ms ≈ patience_first_s × 1000`); `test_events_fire`.
- [ ] **Step 2:** Implement; run tests.
- [ ] **Step 3:** Start the stand-in; run the seven stand-in scenarios (about 60 min wall time).
- [ ] **Step 4 [REAL]:** `relay load run steady-real --real`, after `relay budget`.
- [ ] **Step 5:** Sanity before trusting anything: in `minute-halved`, 429s seen by the client stay near zero after the change (the governor adapted); in `daily-exhausted`, rows after exhaustion are tagged `@20b` and later `extractive`; `steady-twin` percentiles are within the spread printed by `relay calibrate`. If any fails, fix and re-run that scenario only. Commit raw outputs and `feat: load scenarios`.

---

### Task 11: Tables and the slow-request breakdown

**Files:** Create `src/relay/report/{make_tables,slow_request}.py`; Test `tests/test_tables.py`

**Interfaces:**
- Produces: `table_2a(df)`, `table_2b(df, prices)`, `table_3(df)` returning DataFrames; `peak_in_system(df) -> int`; `relay tables` writing `report/tables.md`; `relay slow --scenario overload --pct 99`.

Definitions (state them under the tables in the write-up):
- Requests sent = rows. Duration (s) = last `start_time` − first. Offered rate = rows / duration × 60.
- Percentiles: `pandas.Series.quantile` (linear) over rows with `status == "ok"`; first token uses non-empty `first_token_ms`. Print n next to each.
- Succeeded / Rate-limited / Timed out or given up % = share of all rows with `ok` / `rate_limited` / `timeout`. Any `error` share goes in a footnote.
- Retries per request = mean of `retries` over all rows.
- Peak queue length = the largest number of requests in the system at once, by sweeping `[start_time, start_time + total_ms]`.
- Tokens per answer = Σ(`tokens_in + tokens_out`) over **all** rows ÷ number of `ok` rows (failed work is paid for by the answers that succeed).
- Cost per 1,000 answers = 1000 × Σ(`tokens_in`·p_in + `tokens_out`·p_out)/10⁶ ÷ `ok` rows, using fallback prices for rows tagged `@20b` and primary prices otherwise. This is an upper bound (understand tokens are on the cheaper model, cached tokens are billed at half); say so.
- Table 3: one row per `eval-*` scenario. Correct and cited %, Wrong %, Correctly refused % over questions asked; the wrongly-refused share in a footnote; first token and total at p50 / p95.
- Table 2 rows: every scenario from Task 10. Table 1 is filled from `relay probe summarize`.

`relay slow`: pick the `ok` request nearest the requested percentile of `total_ms`; print its spans in order and the `split()` totals (waiting, working, retrying; before and after the first line), with provider `queue_time` shown inside the waiting share.

- [ ] **Step 1:** Failing tests on a hand-written 8-row CSV with known answers: percentiles; `peak_in_system == 2` for two overlapping rows; tokens per answer counts failed rows' tokens; cost formula with one `@20b` row; one-decimal percentages; an empty scenario column prints `not measured`.
- [ ] **Step 2:** Implement; run tests; `uv run relay tables`.
- [ ] **Step 3:** Commit `feat: tables from results.csv and slow-request breakdown`.

---

### Task 12: Write-up scaffold and submission check

**Files:** Create `report/report.md`, `src/relay/report/submit_check.py`; Test `tests/test_submit_check.py`

- [ ] **Step 1:** `report/report.md` with exactly these headings in this order, each followed by an evidence list (file and number to cite), no prose:
  1. What I built and why
  2. The real service and my stand-in
  3. Results
  4. Where the time goes
  5. When a limit is hit
  - Retrieval: quality versus cost
  - How this library will mislead a naive retriever
  6. Retries
  7. What the person sees
  8. What I'm unsure about, what I left out, what I learned
  - Your feedback on Vaani
  - How I worked with the assistant

  | Heading | Evidence to pull |
  |---|---|
  | 1 | D1 capacity arithmetic with measured tokens per answer; the pipeline diagram |
  | 2 | Table 1; `relay probe summarize`; `relay calibrate` side-by-side; `steady-real` against `steady-twin`; the calibration term with the least data behind it |
  | 3 | Tables 2a, 2b; n per row; one number nobody could guess (candidates: gating overhead in ms, reasoning share of output tokens, index of the first 429 in a burst, whether rejected calls cost quota, cache effect on remaining tokens, 2 of 8 H-Net tables missing from HTML) |
  | 4 | `relay slow` output for one request |
  | 5 | `minute-halved` and `daily-exhausted` rows grouped by `retrieval_setting`; D4 table |
  | Retrieval | Table 3; recall sweep; G4 reasoning; the lean setting |
  | Mislead | Three questions from Task 9 Step 9 with the wrong passage each |
  | 6 | D6; `overload` against `overload-naive-retry` |
  | 7 | The `EXTRACTIVE_DAY` text exactly as a user receives it, with the resume time |
  | 8 | Least certain assumption (candidates: duplicate share in bursts, stand-in load slope, daily refill model); two things not built (candidates: semantic cache, model-based judge in the hot path, priorities per user, a second provider, vector database) |
- [ ] **Step 2:** Tests then implementation of `relay submit-check`: no `gsk_` string anywhere in the tree outside `.env`; `.env` git-ignored; `results.csv` header exact; every scenario in `report/tables.md` exists in `results.csv`; `report.md` headings match the list in order; `report/tables.md` appears verbatim in `report.md`; no patch has `verified: false`.
- [ ] **Step 3 [KAMAL]:** Write the prose in your own words. Paste `report/tables.md`. Visit vaani.dev and write the feedback. Write the half page on working with the assistant.
- [ ] **Step 4 [KAMAL]:** Export every session with the assistant, unedited, into `submission/conversation/` (this planning session and every implementation session).
- [ ] **Step 5:** `uv run relay submit-check` passes; `uv run pytest -q` passes. Commit. Kamal reviews `build` and decides the merge to `main`.

---

## Order and parallel work

- Day 1: Task 0, then Tasks 1 to 3b (no tokens) alongside Tasks 4 to 5 (probe). Kamal reads the papers and writes questions.
- Day 2: Tasks 6, 7, 8; Task 9 through Step 7 for S1 and S2.
- Day 3: Rest of Task 9; Tasks 10, 11.
- Day 4: Task 12; re-runs; buffer.
- Tasks 6, 7 and 8 carry the most interacting logic; use the higher effort setting there. Tasks 0 to 3, 11 and 12 are safe at the lower one.
