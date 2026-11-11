from __future__ import annotations
import asyncio, json, math, os, time
from dataclasses import replace
from pathlib import Path
from relay.types import Level, Reply, OutEvent, RetrievalSetting, Hit
from relay.trace import Trace
from relay.llm.client import LLMRequest, LLMError
from relay.llm.governor import WouldMissDeadline, DailyExhausted
from relay.llm.retry import RetryPolicy
from relay.ingest.chunk import count_tokens
from . import messages as M
from .understand import heuristic_understand, parse_understanding, PROMPT as U_PROMPT, Understanding
from .decide import choose_level
from .prompt import build_draft_messages
from .gate import LineGate, CITE
from .extractive import extractive_reply
from .cache import AnswerCache, Flight, norm_key

ABBR = {"openai/gpt-oss-120b": "120b", "openai/gpt-oss-20b": "20b"}


class _FirstBad(Exception): ...
class _LaterBad(Exception): ...


class _Sub:
    def __init__(self, t0: float, on_event):
        self.t0, self.on_event, self.first_ms, self.events = t0, on_event, None, []

    def push(self, kind, text):
        ms = int((time.monotonic() - self.t0) * 1000)
        if kind in ("sentence", "notice") and self.first_ms is None:
            self.first_ms = ms
        ev = OutEvent(kind, text, ms)
        self.events.append(ev)
        if self.on_event:
            self.on_event(ev)


class Relay:
    def __init__(self, cfg, client, gov, search, source: str, papermap=None, record_dir: Path | None = None,
                 traces_dir: Path = Path("raw/traces")):
        self.cfg, self.client, self.gov, self.search, self.source = cfg, client, gov, search, source
        self.papermap, self.record_dir, self.traces_dir = papermap, record_dir, traces_dir
        self.retry = RetryPolicy(cfg.retry)
        self.cache = AnswerCache(cfg.cache.ttl_s)
        self.P, self.F = cfg.models.primary.id, cfg.models.fallback.id

    # ---------- public ----------
    async def ask(self, question: str, *, request_id: str, arrival_t: float | None = None,
                  setting: RetrievalSetting | None = None, force_level: Level | None = None,
                  cache_key: str | None = None, trace_tags: dict | None = None, on_event=None) -> Reply:
        t0 = arrival_t if arrival_t is not None else time.monotonic()
        sub = _Sub(t0, on_event)
        tags = trace_tags or {}
        key = cache_key or norm_key(question)
        use_cache = self.cfg.cache.enabled
        if use_cache and (hit := self.cache.get(key)):
            for line in hit.text.split("\n"):
                sub.push("sentence", line)
            return replace(hit, request_id=request_id, first_token_ms=sub.first_ms, total_ms=sub.first_ms or 0,
                           tokens_in=0, tokens_out=0, retries=0, retrieval_setting=hit.retrieval_setting + "+cache")
        owner = not (use_cache and key in self.flights_())
        if owner:
            fl = Flight()
            tr = Trace(request_id, t0)
            fl.task = asyncio.ensure_future(self._run(question, setting, force_level, tr, fl, tags))
            if use_cache:
                self.cache.flights[key] = fl
        else:
            fl = self.cache.flights[key]
        fl.join(sub)
        try:
            rep = await asyncio.shield(fl.task)
        except asyncio.CancelledError:
            fl.leave(sub)
            if owner:
                self.cache.flights.pop(key, None)
            raise
        finally:
            if owner and fl.task.done():
                self.cache.flights.pop(key, None)
        total = int((time.monotonic() - t0) * 1000)
        if owner:
            if use_cache and rep.status == "ok" and not rep.truncated and not rep.refused and rep.level != Level.EXTRACTIVE:
                self.cache.put(key, rep)
            out = replace(rep, request_id=request_id, first_token_ms=sub.first_ms, total_ms=total)
            tr.end_ms = total
            tr.notes.update(level=int(rep.level), setting=rep.retrieval_setting, status=rep.status, **tags)
            self._save_trace(tr, tags)
            return out
        return replace(rep, request_id=request_id, first_token_ms=sub.first_ms, total_ms=total, tokens_in=0,
                       tokens_out=0, retries=0, retrieval_setting=rep.retrieval_setting + "+cache")

    def flights_(self):
        return self.cache.flights

    # ---------- core ----------
    async def _run(self, q, setting, force_level, tr: Trace, fl: Flight, tags) -> Reply:
        cfg = self.cfg
        rep = Reply(tr.request_id)
        fl.push("status", M.STATUS)
        setting = setting or cfg.retrieval.settings[cfg.retrieval.default]
        lean = cfg.retrieval.settings[cfg.retrieval.lean]
        gP, gF = self.gov.for_model(self.P), self.gov.for_model(self.F)
        pressure = (force_level not in (None, Level.FULL)) or gP.wait_s(cfg.capacity.tokens_per_answer_full) > cfg.ladder.wait_full_s \
            or gF.wait_s(cfg.understand.max_completion_tokens + 400) > 0
        u = heuristic_understand(q)
        if cfg.understand.get("use_llm", True) and not pressure:
            u = await self._understand(q, tr, rep, tags) or u
        tr.notes["understand"] = u.source
        hits = await self._retrieve(q, u, setting, tr)
        if not u.in_scope and cfg.retrieval.refuse_below is not None and (not hits or hits[0].score < cfg.retrieval.refuse_below):
            return self._refuse(rep, hits, fl, setting)
        exclude: tuple = ()
        for _ in range(2):
            est_full = count_tokens(build_draft_messages(q, hits)[0]["content"]) + cfg.draft.max_completion_tokens
            est_lean = int(est_full * lean.k / max(1, setting.k)) + cfg.draft.lean_max_completion_tokens
            left = cfg.slo.patience_total_s - tr.now_ms() / 1000
            if force_level is not None and not exclude:
                level = force_level
                model = None if level == Level.EXTRACTIVE else (self.F if level == Level.FALLBACK else self.P)
            else:
                level, model = choose_level(lambda m, t: self.gov.for_model(m).wait_s(t),
                                            lambda m: self.gov.for_model(m).daily_frac_left(), est_full, est_lean, left, cfg, exclude)
            tr.notes["level_choice"] = int(level)
            if level == Level.EXTRACTIVE:
                return self._extractive(rep, q, hits, fl, setting)
            use_setting = setting if level == Level.FULL else lean
            if use_setting != setting:
                hits = await self._retrieve(q, u, use_setting, tr)
            rep.level = level
            rep.retrieval_setting = use_setting.tag + ("@20b" if model == self.F else "")
            max_tok = cfg.draft.max_completion_tokens if level == Level.FULL else cfg.draft.lean_max_completion_tokens
            try:
                return await self._draft_loop(q, u, hits, model, level, use_setting, max_tok, tr, rep, fl, tags)
            except (WouldMissDeadline, DailyExhausted) as e:
                tr.notes.setdefault("limit_events", []).append(f"{ABBR.get(model, model)}:{type(e).__name__}")
                exclude = exclude + (model,)
        return self._extractive(rep, q, hits, fl, setting)

    async def _understand(self, q, tr, rep, tags) -> Understanding | None:
        cfg = self.cfg
        g = self.gov.for_model(self.F)
        req = LLMRequest(self.F, [{"role": "user", "content": U_PROMPT + q}], cfg.understand.max_completion_tokens,
                         cfg.models.fallback.reasoning_effort, json_mode=True,
                         trace=f"stage=understand;qid={tags.get('qid')};setting=-;model=20b;attempt=0")
        est = count_tokens(req.messages[0]["content"]) + req.max_completion_tokens
        try:
            gr = await g.acquire(est, time.monotonic() + 0.2)
        except (WouldMissDeadline, DailyExhausted):
            return None
        try:
            with tr.span("understand:20b", "work"):
                rec = await self.client.stream(req, ttft_timeout_s=cfg.understand.timeout_s, total_timeout_s=cfg.understand.timeout_s * 2)
            g.on_response(gr, rec)
            self._count(rep, rec.usage)
            self._record("understand", tags, "-", "20b", 0, rec)
            return parse_understanding(rec.content)
        except LLMError as e:
            g.on_error(gr, e); rep.tokens_in += est - req.max_completion_tokens
            return None
        except (ValueError, KeyError):
            return None
        except asyncio.CancelledError:
            g.release(gr); raise

    async def _retrieve(self, q, u, setting, tr, papers=..., k=None) -> list[Hit]:
        papers = u.papers if papers is ... else papers
        with tr.span(f"retrieve:{setting.name}", "work") as sp:
            r = await self.search.search_papers(u.queries or [q], papers, setting, question=q,
                                                timeout_s=self.cfg.retrieval.tool_timeout_s, per_paper=u.ambiguous and papers is not None, k=k)
            sp["rerank_timed_out"] = r.timed_out
        hits = r.hits
        pm_cfg = self.cfg.get("papermap") or {}
        if self.papermap and pm_cfg.get("enabled"):
            mh = self.papermap.resolve(self.papermap.lookup(q, papers), self.search.chunks_by_id(setting.chunk_tokens), pm_cfg.get("max_hits", 2))
            if mh:
                ids = {h.chunk.id for h in mh}
                hits = (mh + [h for h in hits if h.chunk.id not in ids])[:max(len(hits), len(mh))]
                tr.notes["map_hits"] = sorted(ids)
        return hits

    async def _draft_loop(self, q, u, hits, model, level, setting, max_tok, tr, rep, fl, tags) -> Reply:
        cfg = self.cfg
        redrafts = 0
        while True:
            try:
                return await self._draft(q, hits, model, max_tok, tr, rep, fl, tags, setting, redrafts)
            except _FirstBad:
                tr.notes.setdefault("gate_fail", []).append("first")
                if level == Level.FULL and redrafts < cfg.draft.max_redrafts:
                    redrafts += 1
                    hits = await self._retrieve(q, u, setting, tr, papers=None, k=setting.k * 2)
                    continue
                return self._extractive(rep, q, hits, fl, setting, reason="unverified")

    async def _draft(self, q, hits, model, max_tok, tr, rep, fl, tags, setting, attempt_no) -> Reply:
        cfg = self.cfg
        g = self.gov.for_model(model)
        msgs = build_draft_messages(q, hits)
        ptok = count_tokens(msgs[0]["content"])
        est = ptok + max_tok
        mcfg = cfg.models.primary if model == self.P else cfg.models.fallback
        tries = 0
        while True:
            req = LLMRequest(model, msgs, max_tok, mcfg.reasoning_effort, seed=cfg.seed,
                             trace=f"stage=draft;qid={tags.get('qid')};setting={setting.name};model={ABBR[model]};attempt={attempt_no}")
            left_first = cfg.slo.patience_first_s - tr.now_ms() / 1000
            with tr.span(f"queue:{ABBR[model]}", "wait"):
                gr = await g.acquire(est, time.monotonic() + max(0.0, left_first - 1.0))
            self.retry.note_request()
            gate = LineGate(hits, q)
            sent: list[str] = []
            state = {"refused": False}

            def handle(verdicts):
                for v in verdicts:
                    if v.reason == "sentinel":
                        state["refused"] = True
                    elif state["refused"]:
                        continue
                    elif v.ok:
                        line = v.line[:CITE.search(v.line).start()].rstrip() + " " + " ".join(dict.fromkeys(hits_by[i].cite for i in v.ids))
                        sent.append(line); rep.cited_ids += [i for i in v.ids if i not in rep.cited_ids]
                        if not tr.first_ms:
                            tr.mark_first()
                        fl.push("sentence", line)
                    else:
                        tr.notes.setdefault("gate_reasons", []).append(v.reason)
                        raise (_LaterBad if sent else _FirstBad)()

            hits_by = {h.chunk.id: h.chunk for h in hits}

            def on_delta(d):
                if d.kind == "content":
                    handle(gate.feed(d.text))

            sp = {"name": f"model:{ABBR[model]}", "kind": "work", "start_ms": round(tr.now_ms(), 1), "attempt": tries}
            tr.spans.append(sp)
            try:
                rec = await self.client.stream(req, ttft_timeout_s=cfg.draft.ttft_timeout_s, total_timeout_s=cfg.draft.total_timeout_s, on_delta=on_delta)
                sp["end_ms"] = round(tr.now_ms(), 1)
                handle(gate.flush())
                g.on_response(gr, rec)
                self._count(rep, rec.usage)
                sp.update(ttft_raw_ms=round((rec.t_first_content - rec.t_send) * 1000, 1) if rec.t_first_content else None,
                          queue_time=(rec.usage or {}).get("queue_time"), usage_estimated=rec.usage_estimated)
                self._record("draft", tags, setting.name, ABBR[model], attempt_no, rec)
            except (_FirstBad, _LaterBad) as e:
                sp["end_ms"] = round(tr.now_ms(), 1)
                rep.tokens_in += ptok; rep.tokens_out += count_tokens(gate.buf) + 20 * (len(sent) + 1)
                if isinstance(e, _FirstBad):
                    raise
                fl.push("notice", M.TRUNCATED); rep.truncated = True
                return self._finish(rep, sent)
            except LLMError as e:
                sp["end_ms"] = round(tr.now_ms(), 1); sp["kind"] = "retry"; sp["error"] = e.kind
                g.on_error(gr, e)
                rep.tokens_in += ptok if e.kind in ("midstream", "timeout_total", "timeout_ttft") else 0
                if e.kind == "midstream" and sent:
                    fl.push("notice", M.TRUNCATED); rep.truncated = True
                    return self._finish(rep, sent)
                if e.kind == "rate_limited_day":
                    raise DailyExhausted(e.retry_after_s or 3600)
                d = self.retry.decide(e, tries, bool(sent))
                if d is None:
                    if e.kind == "rate_limited_minute":
                        raise WouldMissDeadline(e.retry_after_s or 0)
                    rep.status = "error"
                    return self._extractive(rep, q, hits, fl, setting, reason="error")
                if tr.now_ms() / 1000 + d > cfg.slo.patience_first_s - 1.0:
                    raise WouldMissDeadline(d)
                with tr.span("backoff", "retry"):
                    await asyncio.sleep(d)
                tries += 1; rep.retries += 1
                continue
            except asyncio.CancelledError:
                sp["end_ms"] = round(tr.now_ms(), 1)
                g.release(gr)
                raise
            if state["refused"]:
                rep.refused = True
                secs = sorted({f"{h.chunk.cite}" for h in hits[:3]})
                txt = M.REFUSAL.format(sections=", ".join(secs))
                fl.push("notice", txt); rep.text = txt
                return rep
            if not sent:
                raise _FirstBad()
            return self._finish(rep, sent)

    # ---------- helpers ----------
    def _finish(self, rep, sent):
        rep.text = "\n".join(sent) + (("\n" + M.TRUNCATED) if rep.truncated else "")
        return rep

    def _refuse(self, rep, hits, fl, setting):
        txt = M.REFUSAL.format(sections=", ".join(sorted({h.chunk.cite for h in hits[:3]})) or "none")
        fl.push("notice", txt); rep.text, rep.refused, rep.retrieval_setting = txt, True, setting.tag
        return rep

    def _extractive(self, rep, q, hits, fl, setting, reason=None):
        gP, gF = self.gov.for_model(self.P), self.gov.for_model(self.F)
        if reason is None:
            reason = "day" if (gP.daily_frac_left() <= self.cfg.ladder.daily_stop_below and gF.daily_frac_left() <= self.cfg.ladder.daily_stop_below) else "minute"
            rep.status = "rate_limited"
        txt = extractive_reply(q, hits, "day" if reason == "day" else ("minute" if reason == "minute" else "unverified"),
                               gP.day_reset_in_s() if reason == "day" else None)
        fl.push("notice", txt)
        rep.text, rep.level, rep.retrieval_setting = txt, Level.EXTRACTIVE, "extractive"
        return rep

    def _count(self, rep, usage):
        u = usage or {}
        rep.tokens_in += u.get("prompt_tokens") or 0
        rep.tokens_out += u.get("completion_tokens") or 0

    def _record(self, stage, tags, setting, model, attempt, rec):
        if not self.record_dir or not tags.get("qid"):
            return
        d = self.record_dir / stage; d.mkdir(parents=True, exist_ok=True)
        u = rec.usage or {}
        (d / f"{tags['qid']}__{setting}__{model}__{attempt}.json").write_text(json.dumps(
            {"content": rec.content, "usage": u, "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}))

    def _save_trace(self, tr, tags):
        if not tags.get("scenario"):
            return
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        with (self.traces_dir / f"{tags['scenario']}.jsonl").open("a") as f:
            f.write(json.dumps(tr.to_dict()) + "\n")


def build(cfg, real: bool, limits=None, ledger=None, record_dir=None, warm=True):
    from dotenv import load_dotenv
    from relay.llm.client import LLMClient
    from relay.llm.governor import Governor
    from relay.retrieve.tool import PaperSearch
    from relay.retrieve.papermap import PaperMap
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY", "") if real else "standin"
    if real and not key:
        raise SystemExit("GROQ_API_KEY missing in .env")
    client = LLMClient(cfg.endpoints.real if real else cfg.endpoints.standin, key, "real" if real else "stand-in")
    gov = Governor.from_config(cfg, limits=limits, ledger=ledger)
    ps = PaperSearch()
    if warm:
        ps.warmup()
    pm_path = Path("data/corpus/papermap.json")
    pm = PaperMap.load(pm_path) if pm_path.exists() else None
    return Relay(cfg, client, gov, ps, "real" if real else "stand-in", papermap=pm, record_dir=record_dir)


def cli(argv):
    import argparse
    from relay.config import load_config
    from relay.llm.client import LEDGER
    ap = argparse.ArgumentParser(); ap.add_argument("question"); ap.add_argument("--real", action="store_true"); ap.add_argument("--setting")
    a = ap.parse_args(argv)
    cfg = load_config()
    relay = build(cfg, a.real, ledger=LEDGER if a.real else None)

    async def go():
        r = await relay.ask(a.question, request_id="cli", setting=cfg.retrieval.settings[a.setting] if a.setting else None,
                            on_event=lambda e: print(f"[{e.t_ms:>6} ms] {e.kind}: {e.text}"))
        print(f"\nlevel={r.level.name} status={r.status} setting={r.retrieval_setting} first={r.first_token_ms}ms total={r.total_ms}ms "
              f"tokens={r.tokens_in}+{r.tokens_out} retries={r.retries}")
        await relay.client.aclose()
    asyncio.run(go())
