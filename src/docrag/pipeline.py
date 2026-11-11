"""Orkiestracja: pliki -> metadane -> profil -> chunki -> embeddingi -> Qdrant.

Ingest jest przyrostowy: hash pliku + konfiguracji zapisany w payloadzie; jeśli się nie
zmienił, dokument jest pomijany. Przy zmianie stare chunki dokumentu są usuwane.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .converters import SUPPORTED_EXT, sample_text
from .models import Chunk, ChunkResult, Config, DocMeta
from .profiles import PROFILES, LegalActProfile, Profile
from .splitting import TokenCounter
from .utils import sha256_bytes, sha256_file, slugify

log = logging.getLogger(__name__)

PIPELINE_VERSION = "2"  # podbij przy zmianie parsera/chunkingu - wymusza przeindeksowanie
META_DIR_FILE = "_meta.yaml"
META_SUFFIX = ".meta.yaml"


def iter_files(paths: Iterable[str | Path]) -> Iterable[tuple[Path, Path]]:
    """(plik, katalog główny wywołania) - katalog główny wyznacza zasięg plików _meta.yaml."""
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT and not f.name.endswith(META_SUFFIX):
                    yield f, p
        elif p.is_file():
            yield p, p.parent
        else:
            log.error("Nie istnieje: %s", p)


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class Pipeline:
    EMBED_SLICE = 64          # chunki liczone na GPU jednym ciągiem (między nimi blokada jest zwalniana)

    def __init__(self, cfg: Config, embedder=None, store=None):
        self.cfg = cfg
        self._embedder = embedder
        self._store = store
        self._collection_ready = False
        self.count = TokenCounter(cfg.embedding.model, cfg.chunking.use_model_tokenizer)

    # leniwe inicjalizacje - `inspect` nie potrzebuje GPU ani bazy
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

    # ------------------------------------------------------------ metadane i profil
    def resolve_meta(self, path: Path, root: Path, overrides: dict[str, Any]) -> tuple[DocMeta, str | None]:
        data: dict[str, Any] = {"doc_type": None, "acl": list(self.cfg.defaults.acl)}
        try:  # _meta.yaml z katalogu głównego i wszystkich podkatalogów aż do pliku
            rel = path.parent.relative_to(root)
            dirs = [root] + [root.joinpath(*rel.parts[:i + 1]) for i in range(len(rel.parts))]
        except ValueError:
            dirs = [path.parent]
        for d in dirs:
            if (d / META_DIR_FILE).exists():
                data.update(_load_yaml(d / META_DIR_FILE))
        sidecar = path.with_name(path.name + META_SUFFIX)
        if sidecar.exists():
            data.update(_load_yaml(sidecar))
        data.update({k: v for k, v in overrides.items() if v is not None})

        profile = data.pop("profile", None)
        acl = data.pop("acl")
        meta = DocMeta(
            doc_id=data.pop("doc_id", None) or slugify(path.stem),
            title=data.pop("title", None) or "",
            doc_type=data.pop("doc_type", None) or "",
            acl=[acl] if isinstance(acl, str) else list(acl),
            source=str(path),
            version=_str(data.pop("version", None)),
            valid_from=_str(data.pop("valid_from", None)),
            extra=data,
        )
        return meta, profile

    def pick_profile(self, path: Path, requested: str | None) -> Profile:
        name = requested or "auto"
        if name == "auto":
            sample = sample_text(path)
            scores = {n: cls.detect(sample, path) for n, cls in PROFILES.items()}
            name = max(scores, key=scores.get)
            if scores[name] < 0.5:
                name = "generic"
            log.debug("%s: profil %s (%s)", path.name, name, scores)
        if name not in PROFILES:
            raise ValueError(f"Nieznany profil {name}; dostępne: {', '.join(PROFILES)}")
        cls = PROFILES[name]
        if cls is LegalActProfile:
            return cls(self.cfg.chunking, self.count, self.cfg.legal)
        return cls(self.cfg.chunking, self.count)

    def chunk_file(self, path: Path, root: Path, overrides: dict[str, Any]) -> tuple[DocMeta, Profile, ChunkResult]:
        meta, profile_name = self.resolve_meta(path, root, overrides)
        profile = self.pick_profile(path, overrides.get("profile") or profile_name)
        result = profile.chunk(path, meta)
        if not meta.title:
            meta.title = result.meta_updates.get("detected_title") or path.stem
        if not meta.doc_type:
            meta.doc_type = profile.default_doc_type or self.cfg.defaults.doc_type
        return meta, profile, result

    # ------------------------------------------------------------ payload
    def doc_hash(self, path: Path, profile: Profile, meta: DocMeta) -> str:
        cfg_sig = json.dumps({"v": PIPELINE_VERSION, "profile": profile.name, "chunking": asdict(self.cfg.chunking),
                              "legal": asdict(self.cfg.legal), "model": self.cfg.embedding.model,
                              "prefix": self.cfg.embedding.passage_prefix, "meta": asdict(meta)},
                             sort_keys=True, default=str)
        return sha256_bytes(sha256_file(path).encode(), cfg_sig.encode())

    @staticmethod
    def context_header(meta: DocMeta, chunk: Chunk) -> str:
        parts = [meta.title]
        if chunk.section_path:
            parts.append(" > ".join(chunk.section_path))
        return "\n".join(p for p in parts if p)

    def payload(self, meta: DocMeta, profile: Profile, chunk: Chunk, idx: int, doc_hash: str, now: str) -> dict:
        p = {
            "chunk_id": chunk.chunk_id, "doc_id": meta.doc_id, "chunk_index": idx,
            "parent_id": chunk.parent_id, "parent_text": chunk.parent_text,
            "doc_type": meta.doc_type, "title": meta.title, "section_path": chunk.section_path,
            "unit": chunk.unit, "refs": chunk.refs, "acl": meta.acl, "source": meta.source,
            "version": meta.version, "valid_from": meta.valid_from, "profile": profile.name,
            "context": self.context_header(meta, chunk), "text": chunk.text,
            "token_count": chunk.token_count, "doc_hash": doc_hash, "ingested_at": now,
        }
        p.update(chunk.extra)
        p.update({k: v for k, v in meta.extra.items() if k not in p})
        return p

    @staticmethod
    def embed_text(payload: dict) -> str:
        return f"{payload['context']}\n\n{payload['text']}" if payload["context"] else payload["text"]

    # ------------------------------------------------------------ komendy
    def inspect(self, paths, overrides: dict[str, Any], out: Path | None = None) -> list[dict]:
        rows = []
        now = datetime.now(timezone.utc).isoformat()
        fh = open(out, "w", encoding="utf-8") if out else None
        try:
            for path, root in iter_files(paths):
                meta, profile, res = self.chunk_file(path, root, overrides)
                h = self.doc_hash(path, profile, meta)
                payloads = [self.payload(meta, profile, c, i, h, now) for i, c in enumerate(res.chunks)]
                rows.append({"path": str(path), "doc_id": meta.doc_id, "title": meta.title, "profile": profile.name,
                             "stats": res.stats, "payloads": payloads})
                if fh:
                    for p in payloads:
                        fh.write(json.dumps({**p, "embed_text": self.embed_text(p)}, ensure_ascii=False) + "\n")
        finally:
            if fh:
                fh.close()
        return rows

    def ingest(self, paths, overrides: dict[str, Any], force: bool = False) -> dict[str, Any]:
        summary = {"ingested": 0, "skipped": 0, "failed": 0, "chunks": 0}
        for path, root in iter_files(paths):
            r = self.ingest_file(path, root, overrides, force)
            summary[r["status"]] += 1
            summary["chunks"] += r["chunks"]
        return summary

    def ingest_file(self, path: Path, root: Path, overrides: dict[str, Any], force: bool = False,
                    gpu_lock=None) -> dict[str, Any]:
        """Jeden plik. Wynik: status (ingested | skipped | failed), chunks, doc_id, title, error.

        gpu_lock: blokada współdzielona z wyszukiwaniem (API) - brana osobno dla każdej paczki
        EMBED_SLICE chunków, żeby długi dokument nie wstrzymywał pytań na cały czas ingestu.
        """
        t0 = time.time()
        out: dict[str, Any] = {"status": "failed", "chunks": 0, "doc_id": None, "title": None, "error": None}
        try:
            meta, profile, res = self.chunk_file(path, root, overrides)
            out.update(doc_id=meta.doc_id, title=meta.title)
            h = self.doc_hash(path, profile, meta)
            if not force and self.store.get_doc_hash(meta.doc_id) == h:
                log.info("= %s bez zmian, pomijam", meta.doc_id)
                out["status"] = "skipped"
                return out
            if not res.chunks:
                log.warning("! %s: brak chunków", path)
                out["error"] = "Nie znaleziono tekstu do zindeksowania"
                return out
            if not self._collection_ready:
                self.store.ensure_collection(self.embedder.dim)
                self._collection_ready = True
            now = datetime.now(timezone.utc).isoformat()
            payloads = [self.payload(meta, profile, c, i, h, now) for i, c in enumerate(res.chunks)]
            texts = [self.embed_text(p) for p in payloads]
            vecs = []
            for i in range(0, len(texts), self.EMBED_SLICE):
                with gpu_lock or nullcontext():
                    vecs.extend(self.embedder.embed_passages(texts[i: i + self.EMBED_SLICE]))
            self.store.delete_doc(meta.doc_id)
            self.store.upsert(payloads, vecs)
            out.update(status="ingested", chunks=len(payloads))
            log.info("+ %s [%s] %d chunków, %.1fs %s", meta.doc_id, profile.name, len(payloads),
                     time.time() - t0, res.stats)
        except Exception as e:
            log.exception("x %s: błąd", path)
            out["error"] = f"{type(e).__name__}: {e}"
        return out


def _str(v) -> str | None:
    return None if v is None else str(v)
