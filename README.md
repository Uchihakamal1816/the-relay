# The Relay

Rate-limit-aware, cited question answering over three pinned arXiv papers on Groq's free tier, with a calibrated stand-in model service and open-loop load tests. Design and task list: `PLAN.md`.

```bash
uv sync
uv run relay fetch && uv run relay ingest && uv run relay index build   # papers -> corpus -> indexes (no tokens)
uv run relay standin serve &                                            # stand-in on :8900
uv run relay ask "How many layers does the Transformer encoder have?"   # add --real to use Groq
uv run pytest -q
```

Commands: `fetch ingest index search probe budget standin calibrate ask eval load tables slow submit-check` (`uv run relay <cmd> -h`).
The API key goes only in `.env` as `GROQ_API_KEY` (git-ignored).
