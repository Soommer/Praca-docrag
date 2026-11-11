"""Backendy embeddingów. BGE-M3 daje jednocześnie wektor gęsty i rzadki (leksykalny)."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import EmbeddingCfg

log = logging.getLogger(__name__)


@dataclass
class Vec:
    dense: list[float]
    sparse: dict[int, float] | None = None


class Embedder(ABC):
    cfg: EmbeddingCfg
    dim: int

    def embed_passages(self, texts: list[str]) -> list[Vec]:
        return self._encode([self.cfg.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> Vec:
        return self._encode([self.cfg.query_prefix + text])[0]

    @abstractmethod
    def _encode(self, texts: list[str]) -> list[Vec]:
        ...


class BGEM3Embedder(Embedder):
    def __init__(self, cfg: EmbeddingCfg):
        from FlagEmbedding import BGEM3FlagModel

        self.cfg = cfg
        try:
            self.model = BGEM3FlagModel(cfg.model, use_fp16=cfg.use_fp16, devices=[cfg.device])
        except TypeError:  # starsze wersje FlagEmbedding
            self.model = BGEM3FlagModel(cfg.model, use_fp16=cfg.use_fp16, device=cfg.device)
        self.dim = 1024

    def _encode(self, texts: list[str]) -> list[Vec]:
        out = self.model.encode(texts, batch_size=self.cfg.batch_size, max_length=self.cfg.max_length,
                                return_dense=True, return_sparse=True, return_colbert_vecs=False)
        vecs = []
        for dense, lex in zip(out["dense_vecs"], out["lexical_weights"]):
            sparse = {int(k): float(v) for k, v in lex.items() if float(v) > 0}
            vecs.append(Vec(dense=[float(x) for x in dense], sparse=sparse))
        return vecs


class SentenceTransformerEmbedder(Embedder):
    """Tylko wektor gęsty, np. sdadas/mmlw-roberta-large, intfloat/multilingual-e5-large."""

    def __init__(self, cfg: EmbeddingCfg):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.model = SentenceTransformer(cfg.model, device=cfg.device)
        if cfg.use_fp16 and cfg.device.startswith("cuda"):
            self.model = self.model.half()
        self.model.max_seq_length = min(cfg.max_length, self.model.max_seq_length or cfg.max_length)
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str]) -> list[Vec]:
        arr = self.model.encode(texts, batch_size=self.cfg.batch_size, normalize_embeddings=True,
                                show_progress_bar=False)
        return [Vec(dense=[float(x) for x in row]) for row in arr]


def build_embedder(cfg: EmbeddingCfg) -> Embedder:
    if cfg.backend == "bge-m3":
        return BGEM3Embedder(cfg)
    if cfg.backend == "sentence-transformers":
        return SentenceTransformerEmbedder(cfg)
    raise ValueError(f"Nieznany backend embeddingów: {cfg.backend}")
