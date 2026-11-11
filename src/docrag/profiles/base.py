"""Interfejs profilu: plik -> lista chunków. Każdy typ dokumentów to osobny profil."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import ChunkingCfg, ChunkResult, DocMeta
from ..splitting import CountFn


class Profile(ABC):
    name: str = "base"
    default_doc_type: str | None = None

    def __init__(self, chunking: ChunkingCfg, count: CountFn):
        self.cfg = chunking
        self.count = count

    @classmethod
    def detect(cls, sample: str, path: Path) -> float:
        """Pewność (0-1), że plik pasuje do profilu. Używane przy profile=auto."""
        return 0.0

    @abstractmethod
    def chunk(self, path: Path, meta: DocMeta) -> ChunkResult:
        ...

    def parent_or_none(self, text: str) -> str | None:
        return text if self.count(text) <= self.cfg.parent_max_tokens else None
