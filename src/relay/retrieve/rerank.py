from __future__ import annotations
from functools import lru_cache

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _enc():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=MODEL)


class Reranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in _enc().rerank(query, texts)]
