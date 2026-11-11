"""Dokumenty użytkowników: wgranie pliku -> ingest w tle -> usunięcie (właściciel albo admin).

Widoczność w wyszukiwaniu jest zapisana w polu acl punktów Qdranta:
  private -> ["user:<id>"]  (tylko właściciel),   public -> ["all"]  (wszyscy).
Dokumenty wgrane przez CLI (np. KPC) trafiają do rejestru jako systemowe (bez właściciela) -
usunąć je może tylko admin.
"""
from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .converters import SUPPORTED_EXT
from .db import Database, DocumentRow
from .store import PUBLIC_GROUP
from .utils import slugify

log = logging.getLogger(__name__)


def user_group(user_id: int) -> str:
    """Grupa ACL, którą ma tylko ten użytkownik - jego prywatne dokumenty."""
    return f"user:{user_id}"


def safe_filename(name: str) -> str:
    """Nazwa pliku z przeglądarki bez ścieżek i znaków niedozwolonych w systemach plików (polskie litery zostają)."""
    stem, ext = Path(name).stem, Path(name).suffix.lower()
    stem = re.sub(r"[^\w.\- ]+", "_", stem).strip(" .") or "dokument"
    return stem[:100] + ext


def _remove(path: Path) -> None:
    """Plik dokumentu i jego katalog (upload_dir/<użytkownik>/<doc_id>/), jeśli został pusty.
    Błąd systemu plików (np. Windows: plik otwarty przez trwający ingest) nie przerywa usuwania."""
    try:
        path.unlink(missing_ok=True)
        path.parent.rmdir()
    except OSError as e:
        log.warning("Nie usunięto %s: %s", path, e)


@dataclass
class Actor:
    """Kto wykonuje operację na dokumentach."""
    id: int | None
    name: str | None
    is_admin: bool = False


class DocumentError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class DocumentService:
    def __init__(self, db: Database, pipeline, store, upload_dir: str | Path, max_upload_mb: int,
                 gpu_lock: threading.Lock | None = None, background: bool = True):
        self.db = db
        self.pipeline = pipeline
        self.store = store
        self.upload_dir = Path(upload_dir)
        self.max_bytes = max_upload_mb * 1024 * 1024
        self.gpu_lock = gpu_lock
        # jeden wątek: dokumenty przetwarzane po kolei (GPU i pamięć), wgrywanie nie czeka na koniec
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest") if background else None
        self._synced_at = float("-inf")

    SYNC_INTERVAL = 60.0      # s - lista admina odpytuje Qdranta o dokumenty z CLI najwyżej raz na minutę

    # ------------------------------------------------------------ wgrywanie
    def upload(self, actor: Actor, filename: str, stream: BinaryIO, title: str | None, public: bool) -> DocumentRow:
        if actor.id is None:
            raise DocumentError(401, "Wgrywanie dokumentów wymaga zalogowania")
        name = Path(filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXT:
            raise DocumentError(415, f"Nieobsługiwany format {ext or '(brak)'}; dozwolone: "
                                     f"{', '.join(sorted(SUPPORTED_EXT))}")
        doc_id = f"u{actor.id}-{slugify(Path(name).stem)[:40]}-{secrets.token_hex(3)}"
        # katalog na dokument, plik pod oryginalną nazwą: nazwa pliku jest zapasowym tytułem,
        # gdy treść go nie zawiera (inaczej w odpowiedziach pojawiałby się tytuł "u1-raport-a3f9c2")
        target = self.upload_dir / str(actor.id) / doc_id / safe_filename(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with open(target, "wb") as fh:
                while block := stream.read(1 << 20):
                    size += len(block)
                    if size > self.max_bytes:
                        raise DocumentError(413, f"Plik większy niż {self.max_bytes // 1024 // 1024} MB")
                    fh.write(block)
        except BaseException:
            _remove(target)
            raise
        if size == 0:
            _remove(target)
            raise DocumentError(422, "Plik jest pusty")

        self.db.add_document(doc_id, actor.id, (title or "").strip() or Path(name).stem, name, str(target),
                             "public" if public else "private", "processing")
        log.info("upload %s przez %s (%d B, %s)", doc_id, actor.name, size, "public" if public else "private")
        if self.executor:
            self.executor.submit(self._ingest, doc_id)
        else:
            self._ingest(doc_id)
        return self.db.get_document(doc_id)

    def _ingest(self, doc_id: str) -> None:
        rec = self.db.get_document(doc_id)
        if rec is None:                 # usunięty, zanim przyszła jego kolej
            return
        path = Path(rec.path)
        acl = [PUBLIC_GROUP] if rec.visibility == "public" else [user_group(rec.owner_id)]
        # tytuł podany przez użytkownika ma pierwszeństwo; równy nazwie pliku = wykryj go z treści
        # (a gdy się nie da, pipeline weźmie nazwę pliku)
        title = None if rec.title == Path(rec.filename or "").stem else rec.title
        overrides = {"doc_id": doc_id, "acl": acl, "title": title, "owner": rec.owner}
        r = self.pipeline.ingest_file(path, path.parent, overrides, force=True, gpu_lock=self.gpu_lock)
        if self.db.get_document(doc_id) is None:      # usunięty w trakcie ingestu - posprzątaj po nim
            self.store.delete_doc(doc_id)
            return
        if r["status"] == "ingested":
            self.db.update_document(doc_id, status="ready", chunks=r["chunks"], error=None,
                                    title=r["title"] or rec.title)
        else:
            self.db.update_document(doc_id, status="error", error=r["error"] or "Nieznany błąd przetwarzania")

    # ------------------------------------------------------------ lista i usuwanie
    def list(self, actor: Actor) -> list[DocumentRow]:
        if actor.id is None:
            return []
        if actor.is_admin:
            if time.monotonic() - self._synced_at > self.SYNC_INTERVAL:
                try:
                    self.sync_from_store()
                except Exception as e:      # Qdrant niedostępny - lista z samego rejestru
                    log.warning("Synchronizacja z Qdrantem nieudana: %s", e)
            return self.db.list_documents(all_owners=True)
        return self.db.list_documents(owner_id=actor.id)

    def delete(self, actor: Actor, doc_id: str) -> None:
        rec = self.db.get_document(doc_id)
        if rec is None:
            raise DocumentError(404, "Nie ma takiego dokumentu")
        if not (actor.is_admin or (actor.id is not None and rec.owner_id == actor.id)):
            # cudzy dokument wygląda tak samo jak nieistniejący - nie zdradzamy, że jest
            raise DocumentError(404, "Nie ma takiego dokumentu")
        # najpierw Qdrant: gdy nie działa, nic nie jest usuwane i można powtórzyć (inaczej wektory
        # zostałyby bez wpisu w rejestrze i wróciłyby przy synchronizacji jako dokument systemowy)
        try:
            self.store.delete_doc(doc_id)
        except Exception as e:
            log.error("delete %s: Qdrant: %s", doc_id, e)
            raise DocumentError(503, "Baza wektorowa niedostępna - dokument nie został usunięty")
        self.db.delete_document(doc_id)
        if rec.path:
            _remove(Path(rec.path))
        log.info("delete %s przez %s%s", doc_id, actor.name, " (admin)" if rec.owner_id != actor.id else "")

    def sync_from_store(self) -> int:
        """Rejestruje dokumenty obecne w Qdrancie, a nieznane aplikacji (wgrane przez CLI)."""
        self._synced_at = time.monotonic()
        added = 0
        for doc_id, info in self.store.doc_summaries().items():
            if self.db.get_document(doc_id) is None:
                acl = info["acl"]
                visibility = "public" if PUBLIC_GROUP in acl else "restricted"
                self.db.add_document(doc_id, None, info["title"], None, None, visibility, "ready", info["chunks"])
                added += 1
        return added

    def shutdown(self) -> None:
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
