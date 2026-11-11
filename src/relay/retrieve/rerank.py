from __future__ import annotations
from functools import lru_cache

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _enc():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model_name=MODEL)


class Reranker:
    def score(self, query: str, texts: list[str]) -> list[float]:
        # batch_size=1: batched ONNX pads every pair to the longest; measured 1131 ms -> 326 ms for 20 pairs
        return [float(x) for x in _enc().rerank(query, texts, batch_size=1)]
