from __future__ import annotations
import argparse, asyncio, importlib, sys

# command -> "module:function"; each function takes argv list
COMMANDS = {
    "fetch": "relay.ingest.fetch:cli",
    "ingest": "relay.ingest.parse_html:cli",
    "index": "relay.retrieve.index:cli",
    "search": "relay.retrieve.tool:cli",
    "probe": "relay.probe:cli",
    "budget": "relay.probe:budget_cli",
    "standin": "relay.standin.server:cli",
    "calibrate": "relay.standin.calibrate:cli",
    "ask": "relay.pipeline.orchestrator:cli",
    "eval": "relay.evalx.run_eval:cli",
    "load": "relay.load.runner:cli",
    "tables": "relay.report.make_tables:cli",
    "slow": "relay.report.slow_request:cli",
    "submit-check": "relay.report.submit_check:cli",
}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print("usage: relay {" + ",".join(COMMANDS) + "} ..."); sys.exit(2)
    mod, fn = COMMANDS[argv[0]].split(":")
    r = getattr(importlib.import_module(mod), fn)(argv[1:])
    if asyncio.iscoroutine(r):
        asyncio.run(r)
