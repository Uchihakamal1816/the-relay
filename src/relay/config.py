from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from .types import RetrievalSetting, Limits


class NS(dict):
    """dict with attribute access, recursive."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def _ns(x: Any) -> Any:
    if isinstance(x, dict):
        return NS({k: _ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_ns(v) for v in x]
    return x


def set_path(cfg: NS, dotted: str, value: Any) -> None:
    *head, last = dotted.split(".")
    node = cfg
    for h in head:
        node = node[h]
    node[last] = value


def load_config(path: str | Path = "config/relay.yaml", overrides: dict | None = None) -> NS:
    cfg = _ns(yaml.safe_load(Path(path).read_text()))
    for k, v in (overrides or {}).items():
        set_path(cfg, k, v)
    cfg.retrieval.settings = NS({n: RetrievalSetting(name=n, **s) for n, s in cfg.retrieval.settings.items()})
    cfg.limits_prior = Limits(**cfg.limits_prior)
    return cfg


def setting(cfg: NS, name: str | None = None) -> RetrievalSetting:
    return cfg.retrieval.settings[name or cfg.retrieval.default]
