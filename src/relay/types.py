from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

Paper = Literal["attention", "moshi", "hnet"]
Status = Literal["ok", "rate_limited", "timeout", "error"]
Grade = Literal["correct", "wrong", "refused_correctly", "refused_wrongly"]
SHORT = {"attention": "Attention", "moshi": "Moshi", "hnet": "H-Net"}
LETTER = {"attention": "A", "moshi": "M", "hnet": "H"}


class Level(IntEnum):
    FULL = 0
    LEAN = 1
    FALLBACK = 2
    EXTRACTIVE = 3


@dataclass
class Limits:
    rpm: int
    tpm: int
    rpd: int
    tpd: int


@dataclass(frozen=True)
class Chunk:
    id: str
    paper: str
    section_num: str
    section_title: str
    kind: str
    label: str | None
    page: int | None
    text: str
    n_tokens: int

    @property
    def header(self) -> str:
        parts = [SHORT[self.paper]]
        if self.section_title or self.section_num:
            parts.append(f"§{self.section_num} {self.section_title}".strip())
        if self.label:
            parts.append(self.label)
        if self.page:
            parts.append(f"p.{self.page}")
        return "[" + " | ".join(parts) + "]"

    @property
    def cite(self) -> str:
        parts = [SHORT[self.paper]]
        if self.section_num:
            parts.append(f"§{self.section_num}")
        if self.label:
            parts.append(self.label)
        if self.page:
            parts.append(f"p. {self.page}")
        return "(" + ", ".join(parts) + ")"


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rerank_score: float | None = None
    from_map: bool = False


@dataclass(frozen=True)
class RetrievalSetting:
    name: str
    chunk_tokens: int
    k: int
    rerank: bool
    candidates: int = 20

    @property
    def tag(self) -> str:
        return f"{self.name}:c{self.chunk_tokens}-k{self.k}-{'rr' if self.rerank else 'norr'}"


@dataclass
class OutEvent:
    kind: Literal["status", "sentence", "notice", "final"]
    text: str
    t_ms: int


@dataclass
class Reply:
    request_id: str
    text: str = ""
    cited_ids: list[str] = field(default_factory=list)
    refused: bool = False
    level: Level = Level.FULL
    status: Status = "ok"
    first_token_ms: int | None = None
    total_ms: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    retrieval_setting: str = ""
    truncated: bool = False
