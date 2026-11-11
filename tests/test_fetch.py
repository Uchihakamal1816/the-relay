import httpx
from relay.ingest.fetch import SOURCES, urls, fetch_all


def test_urls_are_version_pinned():
    for p, s in SOURCES.items():
        u = urls(p)
        assert u["html"] == f"https://arxiv.org/html/{s['arxiv_id']}{s['version']}"
        assert u["pdf"] == f"https://arxiv.org/pdf/{s['arxiv_id']}{s['version']}"


def test_manifest_roundtrip(tmp_path):
    calls = []
    def h(req):
        calls.append(str(req.url)); return httpx.Response(200, content=b"x" + str(req.url).encode())
    c = httpx.Client(transport=httpx.MockTransport(h))
    m = fetch_all(tmp_path, c, pause_s=0)
    assert len(m) == 6 and len(calls) == 6
    fetch_all(tmp_path, c, pause_s=0)
    assert len(calls) == 6  # cached by sha
