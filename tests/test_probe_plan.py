from relay.probe import PROBES


def test_probe_plans_fit_budget():
    for name, p in PROBES.items():
        if name in ("concurrency", "rpm-edge", "rpd-burn"):
            continue
        assert sum(s + p.max_out for s in p.sizes) <= p.est_tokens, name
    c = PROBES["concurrency"]
    assert sum(c.sizes) * (200 + c.max_out) <= c.est_tokens


def test_paced_probes_use_governor():
    assert all(p.paced for n, p in PROBES.items() if n not in ("rpm-edge", "tpm-edge", "rpd-burn"))
