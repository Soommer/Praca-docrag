"""Wyszukiwanie: pytanie -> BGE-M3 (dense+sparse) -> Qdrant (RRF, filtr ACL) -> reranker -> źródła.

Po rerankingu fragmenty tego samego artykułu/sekcji są łączone w jedno źródło (small-to-big:
tekst rodzica zamiast fragmentu), a artykuły, do których źródła się odwołują (refs), są
dociągane jako kontekst pomocniczy.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .models import Config
from .pipeline import Pipeline
from .store import SearchFilter

log = logging.getLogger(__name__)

META_FIELDS = ("doc_type", "version", "valid_from", "source")


@dataclass
class Source:
    unit_id: str                     # parent_id podzielonej jednostki albo chunk_id
    doc_id: str
    title: str
    unit: str | None
    section_path: list[str]
    text: str
    score: float                     # wynik rerankera (0-1); 0 dla źródeł pomocniczych
    chunk_ids: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    related_to: str | None = None    # źródło pomocnicze: unit_id źródła, które się do niego odwołuje

    @property
    def label(self) -> str:
        return f"{self.title} - {self.unit}" if self.unit else self.title


@dataclass
class SearchResult:
    query: str
    sources: list[Source]
    related: list[Source]
    stats: dict[str, Any]


class _Unit:
    """Zbiera chunki jednej jednostki (artykułu/sekcji), zanim powstanie z nich Source."""

    def __init__(self, unit_id: str, first: dict, score: float):
        self.unit_id, self.first, self.score = unit_id, first, score
        self.parts: dict[str, dict] = {}

    def add(self, payload: dict, score: float = 0.0) -> None:
        self.parts.setdefault(payload["chunk_id"], payload)
        self.score = max(self.score, score)

    def build(self, expand_parents: bool, related_to: str | None = None) -> Source:
        p = self.first
        parts = sorted(self.parts.values(), key=lambda x: x.get("chunk_index", 0))
        whole = expand_parents and p.get("parent_id") == self.unit_id and p.get("parent_text")
        text = p["parent_text"] if whole else "\n".join(x["text"] for x in parts)
        unit = p.get("unit")
        if whole and p.get("article"):
            unit = f"{'art.' if '#art-' in p['chunk_id'] else '§'} {p['article']}"   # cały artykuł, bez "§ 1–3"
        return Source(self.unit_id, p["doc_id"], p.get("title") or p["doc_id"], unit, p.get("section_path") or [],
                      text, round(self.score, 4), [x["chunk_id"] for x in parts], p.get("refs") or [],
                      {k: p.get(k) for k in META_FIELDS if p.get(k)}, related_to)


class Searcher:
    def __init__(self, cfg: Config, embedder=None, store=None, reranker=None):
        self.cfg = cfg
        self._embedder, self._store, self._reranker = embedder, store, reranker

    # leniwe inicjalizacje - jak w Pipeline
    @property
    def embedder(self):
        if self._embedder is None:
            from .embedding import build_embedder

            self._embedder = build_embedder(self.cfg.embedding)
        return self._embedder

    @property
    def store(self):
        if self._store is None:
            from .store import QdrantStore

            self._store = QdrantStore(self.cfg.qdrant)
        return self._store

    @property
    def reranker(self):
        if self._reranker is None:
            from .rerank import build_reranker

            self._reranker = build_reranker(self.cfg.search)
        return self._reranker

    def search(self, query: str, flt: SearchFilter, top_k: int | None = None) -> SearchResult:
        sc = self.cfg.search
        top_k = top_k or sc.top_k
        t = [time.perf_counter()]
        qvec = self.embedder.embed_query(query)
        t.append(time.perf_counter())
        cands = self.store.search_hybrid(qvec, flt, sc.prefetch_limit, sc.rerank_candidates)
        t.append(time.perf_counter())
        scores = self.reranker.score(query, [Pipeline.embed_text(p) for p in cands])
        t.append(time.perf_counter())

        units: dict[str, _Unit] = {}
        for score, p in sorted(zip(scores, cands), key=lambda x: -x[0]):
            if score < sc.min_score:
                break
            key = p.get("parent_id") if sc.expand_parents and p.get("parent_id") else p["chunk_id"]
            if key not in units:
                if len(units) >= top_k:
                    continue
                units[key] = _Unit(key, p, score)
            units[key].add(p, score)
        sources = [u.build(sc.expand_parents) for u in units.values()]
        related = self._related(sources, flt) if sc.max_refs else []
        t.append(time.perf_counter())

        ms = [round((b - a) * 1000) for a, b in zip(t, t[1:])]
        stats = {"candidates": len(cands), "ms_embed": ms[0], "ms_search": ms[1], "ms_rerank": ms[2],
                 "ms_expand": ms[3]}
        return SearchResult(query, sources, related, stats)

    def _related(self, sources: list[Source], flt: SearchFilter) -> list[Source]:
        """Artykuły z refs (po kolei: najpierw odwołania najlepszego źródła), z tym samym filtrem ACL."""
        have = {s.unit_id for s in sources} | {c for s in sources for c in s.chunk_ids}
        wanted: dict[str, str] = {}
        for s in sources:
            for r in s.refs:
                if "#" in r and r not in have and r not in wanted and len(wanted) < self.cfg.search.max_refs:
                    wanted[r] = s.unit_id
        units: dict[str, _Unit] = {}
        for p in self.store.fetch_units(list(wanted), flt):
            key = p["chunk_id"] if p["chunk_id"] in wanted else p.get("parent_id")
            if key in wanted:
                units.setdefault(key, _Unit(key, p, 0.0)).add(p)
        return [units[k].build(True, wanted[k]) for k in wanted if k in units]
