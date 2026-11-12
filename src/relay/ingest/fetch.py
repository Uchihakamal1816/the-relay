from __future__ import annotations
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import httpx

SOURCES = {
    "attention": {"arxiv_id": "1706.03762", "version": "v7", "title": "Attention Is All You Need", "short": "Attention"},
    "moshi": {"arxiv_id": "2410.00037", "version": "v1", "title": "Moshi: a speech-text foundation model for real-time dialogue", "short": "Moshi"},
    "hnet": {"arxiv_id": "2507.07955", "version": "v2", "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", "short": "H-Net"},
}
UA = {"User-Agent": "relay-fetch/0.1"}


def urls(paper: str) -> dict[str, str]:
    s = SOURCES[paper]; v = s["arxiv_id"] + s["version"]
    return {"html": f"https://arxiv.org/html/{v}", "pdf": f"https://arxiv.org/pdf/{v}"}


def fetch_all(dest: Path = Path("data/papers"), client: httpx.Client | None = None, pause_s: float = 3.0) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    mpath = dest / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    c = client or httpx.Client(headers=UA, follow_redirects=True, timeout=60)
    first = True
    for paper in SOURCES:
        for kind, url in urls(paper).items():
            out = dest / f"{paper}.{kind}"
            key = f"{paper}.{kind}"
            if out.exists() and manifest.get(key, {}).get("sha256") == hashlib.sha256(out.read_bytes()).hexdigest():
                continue
            if not first:
                time.sleep(pause_s)
            first = False
            r = c.get(url); r.raise_for_status()
            out.write_bytes(r.content)
            manifest[key] = {"url": url, "sha256": hashlib.sha256(r.content).hexdigest(), "bytes": len(r.content),
                             "fetched_at": datetime.now(timezone.utc).isoformat()}
            mpath.write_text(json.dumps(manifest, indent=1))
    return manifest


def cli(argv):
    m = fetch_all()
    for k, v in m.items():
        print(k, v["bytes"], v["sha256"][:12])
