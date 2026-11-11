"""Zapis do Qdrant: kolekcja z wektorem gęstym 'dense' i rzadkim 'sparse' + indeksy payloadu."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from .embedding import Vec
from .models import QdrantCfg

log = logging.getLogger(__name__)

PUBLIC_GROUP = "all"


@dataclass
class SearchFilter:
    """Ograniczenia wyszukiwania. groups=None wyłącza ACL (tylko administracja/diagnostyka)."""
    groups: list[str] | None = field(default_factory=list)   # grupy AD użytkownika; "all" dodawane zawsze
    doc_ids: list[str] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)

    def allowed_groups(self) -> list[str] | None:
        return None if self.groups is None else sorted({PUBLIC_GROUP, *self.groups})

    def matches(self, payload: dict) -> bool:
        """To samo co filtr Qdranta, w Pythonie (atrapy w testach)."""
        groups = self.allowed_groups()
        if groups is not None and not set(payload.get("acl") or []) & set(groups):
            return False
        if self.doc_ids and payload.get("doc_id") not in self.doc_ids:
            return False
        return not self.doc_types or payload.get("doc_type") in self.doc_types

KEYWORD_INDEXES = ["doc_id", "doc_type", "acl", "chunk_id", "parent_id", "unit", "section_path",
                   "profile", "article"]


class QdrantStore:
    def __init__(self, cfg: QdrantCfg):
        from qdrant_client import QdrantClient, models

        self.qm = models
        self.cfg = cfg
        self.name = cfg.collection
        self.client = QdrantClient(path=cfg.path) if cfg.path else QdrantClient(url=cfg.url, api_key=cfg.api_key)

    @staticmethod
    def point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "docrag:" + chunk_id))

    def ensure_collection(self, dim: int) -> None:
        qm = self.qm
        if self.client.collection_exists(self.name):
            params = self.client.get_collection(self.name).config.params.vectors
            existing = params["dense"].size if isinstance(params, dict) else params.size
            if existing != dim:
                raise RuntimeError(f"Kolekcja {self.name} ma wymiar {existing}, model daje {dim}. "
                                   "Zmień nazwę kolekcji albo usuń starą.")
            return
        log.info("Tworzę kolekcję %s (dim=%d)", self.name, dim)
        self.client.create_collection(
            self.name,
            vectors_config={"dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams()},
        )
        for field in KEYWORD_INDEXES:
            try:
                self.client.create_payload_index(self.name, field, qm.PayloadSchemaType.KEYWORD)
            except Exception as e:  # tryb lokalny nie wspiera indeksów - to nie błąd
                log.debug("Indeks %s pominięty: %s", field, e)

    def _doc_filter(self, doc_id: str):
        qm = self.qm
        return qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])

    def get_doc_hash(self, doc_id: str) -> str | None:
        if not self.client.collection_exists(self.name):
            return None
        points, _ = self.client.scroll(self.name, scroll_filter=self._doc_filter(doc_id), limit=1,
                                       with_payload=["doc_hash"], with_vectors=False)
        return points[0].payload.get("doc_hash") if points else None

    def delete_doc(self, doc_id: str) -> None:
        if self.client.collection_exists(self.name):
            self.client.delete(self.name, points_selector=self.qm.FilterSelector(filter=self._doc_filter(doc_id)),
                               wait=True)

    def upsert(self, payloads: list[dict[str, Any]], vecs: list[Vec]) -> None:
        qm = self.qm
        batch = []
        for payload, v in zip(payloads, vecs):
            vector: dict[str, Any] = {"dense": v.dense}
            if v.sparse:
                idx = sorted(v.sparse)
                vector["sparse"] = qm.SparseVector(indices=idx, values=[v.sparse[i] for i in idx])
            batch.append(qm.PointStruct(id=self.point_id(payload["chunk_id"]), vector=vector, payload=payload))
            if len(batch) >= self.cfg.upsert_batch:
                self.client.upsert(self.name, points=batch, wait=True)
                batch = []
        if batch:
            self.client.upsert(self.name, points=batch, wait=True)

    # ------------------------------------------------------------ wyszukiwanie
    def _search_filter(self, flt: SearchFilter, extra: list | None = None):
        qm = self.qm
        must = list(extra or [])
        groups = flt.allowed_groups()
        if groups is not None:
            must.append(qm.FieldCondition(key="acl", match=qm.MatchAny(any=groups)))
        if flt.doc_ids:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=flt.doc_ids)))
        if flt.doc_types:
            must.append(qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=flt.doc_types)))
        return qm.Filter(must=must) if must else None

    def search_hybrid(self, query: Vec, flt: SearchFilter, prefetch_limit: int, limit: int) -> list[dict]:
        """Dense + sparse, fuzja RRF. Zwraca payloady w kolejności rankingu (z polem _score)."""
        qm = self.qm
        f = self._search_filter(flt)
        prefetch = [qm.Prefetch(query=query.dense, using="dense", filter=f, limit=prefetch_limit)]
        if query.sparse:
            idx = sorted(query.sparse)
            prefetch.append(qm.Prefetch(query=qm.SparseVector(indices=idx, values=[query.sparse[i] for i in idx]),
                                        using="sparse", filter=f, limit=prefetch_limit))
        res = self.client.query_points(self.name, prefetch=prefetch, query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                                       query_filter=f, limit=limit, with_payload=True, with_vectors=False)
        return [{**pt.payload, "_score": pt.score} for pt in res.points]

    def fetch_units(self, unit_ids: list[str], flt: SearchFilter) -> list[dict]:
        """Chunki jednostek o podanych id: sam chunk (chunk_id) albo części podzielonego rodzica (parent_id)."""
        if not unit_ids:
            return []
        qm = self.qm
        by_id = qm.Filter(should=[qm.FieldCondition(key="chunk_id", match=qm.MatchAny(any=unit_ids)),
                                  qm.FieldCondition(key="parent_id", match=qm.MatchAny(any=unit_ids))])
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(self.name, scroll_filter=self._search_filter(flt, [by_id]),
                                                limit=256, offset=offset, with_payload=True, with_vectors=False)
            out.extend(pt.payload for pt in points)
            if offset is None:
                return out

    def doc_summaries(self) -> dict[str, dict[str, Any]]:
        """doc_id -> {chunks, title, acl} dla wszystkich dokumentów w kolekcji (np. wgranych przez CLI)."""
        if not self.client.collection_exists(self.name):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for hit in self.client.facet(self.name, key="doc_id", limit=100_000, exact=True).hits:
            points, _ = self.client.scroll(self.name, scroll_filter=self._doc_filter(hit.value), limit=1,
                                           with_payload=["title", "acl"], with_vectors=False)
            p = points[0].payload if points else {}
            out[hit.value] = {"chunks": hit.count, "title": p.get("title") or hit.value, "acl": p.get("acl") or []}
        return out

    def stats(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.name):
            return {"collection": self.name, "exists": False}
        out: dict[str, Any] = {"collection": self.name,
                               "points": self.client.count(self.name, exact=True).count}
        try:
            res = self.client.facet(self.name, key="doc_id", limit=1000)
            out["documents"] = {h.value: h.count for h in res.hits}
        except Exception:
            pass
        return out
