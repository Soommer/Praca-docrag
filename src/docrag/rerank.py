"""Rerankery: ocena par (pytanie, fragment) cross-encoderem - dokładniejsza niż podobieństwo wektorów."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .models import SearchCfg

log = logging.getLogger(__name__)


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Wyniki w zakresie 0-1, w kolejności tekstów."""


class BGEReranker(Reranker):
    """BAAI/bge-reranker-v2-m3 - wielojęzyczny (w tym polski), ok. 570M parametrów."""

    def __init__(self, cfg: SearchCfg):
        from FlagEmbedding import FlagReranker

        self.cfg = cfg
        try:
            self.model = FlagReranker(cfg.reranker_model, use_fp16=cfg.reranker_fp16, devices=[cfg.reranker_device])
        except TypeError:  # starsze wersje FlagEmbedding
            self.model = FlagReranker(cfg.reranker_model, use_fp16=cfg.reranker_fp16, device=cfg.reranker_device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.compute_score([[query, t] for t in texts], batch_size=self.cfg.reranker_batch_size,
                                          max_length=self.cfg.reranker_max_length, normalize=True)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]


class NoReranker(Reranker):
    """Bez rerankingu: zachowuje kolejność z wyszukiwania hybrydowego."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 - i / max(len(texts), 1) for i in range(len(texts))]


def build_reranker(cfg: SearchCfg) -> Reranker:
    if cfg.reranker == "bge":
        return BGEReranker(cfg)
    if cfg.reranker == "none":
        return NoReranker()
    raise ValueError(f"Nieznany reranker: {cfg.reranker}")
