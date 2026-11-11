"""SQLite: użytkownicy, sesje i rejestr dokumentów (kto wgrał, status przetwarzania).

Treść dokumentów i wektory są w Qdrancie; tu jest tylko to, czego potrzebuje aplikacja:
właściciel dokumentu, jego widoczność i stan ingestu. Każda operacja otwiera własne połączenie,
więc moduł jest bezpieczny dla wątków FastAPI i wątku ingestu.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    -- AUTOINCREMENT: id usuniętego konta nigdy nie wraca; inaczej nowe konto przejęłoby
    -- prywatne dokumenty poprzednika (acl "user:<id>" w Qdrancie)
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT,                       -- NULL: konto z nagłówków proxy (bez hasła)
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,              -- sha256 tokenu z ciasteczka (sam token nie jest zapisywany)
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,              -- to samo doc_id co w Qdrancie
    owner_id   INTEGER REFERENCES users(id) ON DELETE SET NULL,   -- NULL: dokument systemowy (CLI)
    title      TEXT NOT NULL,
    filename   TEXT,
    path       TEXT,                          -- plik w upload_dir; NULL dla dokumentów z CLI
    visibility TEXT NOT NULL,                 -- private | public | restricted (grupy AD z CLI)
    status     TEXT NOT NULL,                 -- processing | ready | error
    error      TEXT,
    chunks     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_owner ON documents(owner_id);
"""

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
_DUMMY_HASH: str | None = None     # dla nieistniejących kont - ten sam koszt sprawdzenia hasła


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(calc.hex(), h)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class UserRow:
    id: int
    username: str
    is_admin: bool


@dataclass
class DocumentRow:
    doc_id: str
    owner_id: int | None
    owner: str | None
    title: str
    filename: str | None
    path: str | None
    visibility: str
    status: str
    error: str | None
    chunks: int
    created_at: str


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_users_autoincrement()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _migrate_users_autoincrement(self) -> None:
        """Bazy utworzone przed wprowadzeniem AUTOINCREMENT: przebudowa tabeli users z zachowaniem id."""
        c = sqlite3.connect(self.path, timeout=10)
        try:
            row = c.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'").fetchone()
            if not row or "AUTOINCREMENT" in row[0].upper():
                return
            c.execute("PRAGMA foreign_keys = OFF")
            c.executescript("""
                BEGIN;
                ALTER TABLE users RENAME TO users_old;
                CREATE TABLE users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT NOT NULL UNIQUE,
                    password_hash TEXT,
                    is_admin      INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT NOT NULL
                );
                INSERT INTO users SELECT id, username, password_hash, is_admin, created_at FROM users_old;
                DROP TABLE users_old;
                COMMIT;
            """)
        finally:
            c.close()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA journal_mode = WAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ------------------------------------------------------------ użytkownicy
    def add_user(self, username: str, password: str | None, is_admin: bool = False) -> UserRow:
        username = username.strip()
        if not username:
            raise ValueError("Pusta nazwa użytkownika")
        with self._conn() as c:
            try:
                cur = c.execute("INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
                                (username, hash_password(password) if password else None, int(is_admin), now()))
            except sqlite3.IntegrityError:
                raise ValueError(f"Użytkownik {username} już istnieje")
        return UserRow(cur.lastrowid, username, is_admin)

    def set_password(self, username: str, password: str) -> None:
        """Zmiana hasła wylogowuje wszystkie sesje tego konta (np. po wycieku hasła)."""
        with self._conn() as c:
            if not c.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                             (hash_password(password), username)).rowcount:
                raise ValueError(f"Nie ma użytkownika {username}")
            c.execute("DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE username = ?)", (username,))

    def set_admin(self, username: str, is_admin: bool) -> None:
        with self._conn() as c:
            if not c.execute("UPDATE users SET is_admin = ? WHERE username = ?", (int(is_admin), username)).rowcount:
                raise ValueError(f"Nie ma użytkownika {username}")

    def delete_user(self, username: str) -> None:
        with self._conn() as c:
            if not c.execute("DELETE FROM users WHERE username = ?", (username,)).rowcount:
                raise ValueError(f"Nie ma użytkownika {username}")

    def get_user(self, username: str) -> UserRow | None:
        with self._conn() as c:
            r = c.execute("SELECT id, username, is_admin FROM users WHERE username = ?", (username,)).fetchone()
        return UserRow(r["id"], r["username"], bool(r["is_admin"])) if r else None

    def ensure_user(self, username: str, is_admin: bool) -> UserRow:
        """Konto dla użytkownika z nagłówków proxy - tworzone przy pierwszym wejściu."""
        u = self.get_user(username)
        if u is None:
            try:
                return self.add_user(username, None, is_admin)
            except ValueError:          # równoległe pierwsze żądania tego samego użytkownika
                u = self.get_user(username)
        if u.is_admin != is_admin:
            self.set_admin(username, is_admin)
            u.is_admin = is_admin
        return u

    def list_users(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT u.id, u.username, u.is_admin, u.created_at, "
                             "(SELECT COUNT(*) FROM documents d WHERE d.owner_id = u.id) AS documents "
                             "FROM users u ORDER BY u.username").fetchall()
        return [dict(r) for r in rows]

    def authenticate(self, username: str, password: str) -> UserRow | None:
        with self._conn() as c:
            r = c.execute("SELECT id, username, is_admin, password_hash FROM users WHERE username = ?",
                          (username.strip(),)).fetchone()
        if r is None:
            global _DUMMY_HASH
            _DUMMY_HASH = _DUMMY_HASH or hash_password(secrets.token_hex(8))
            verify_password(password, _DUMMY_HASH)     # jedno scrypt, jak dla istniejącego konta
            return None
        return UserRow(r["id"], r["username"], bool(r["is_admin"])) if verify_password(password, r["password_hash"]) else None

    # ------------------------------------------------------------ sesje
    def create_session(self, user_id: int, hours: int) -> str:
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE expires_at < ?", (now(),))
            c.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                      (_token_hash(token), user_id, expires))
        return token

    def session_user(self, token: str) -> UserRow | None:
        with self._conn() as c:
            r = c.execute("SELECT u.id, u.username, u.is_admin FROM sessions s JOIN users u ON u.id = s.user_id "
                          "WHERE s.token_hash = ? AND s.expires_at > ?", (_token_hash(token), now())).fetchone()
        return UserRow(r["id"], r["username"], bool(r["is_admin"])) if r else None

    def delete_session(self, token: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))

    # ------------------------------------------------------------ dokumenty
    _DOC_SELECT = ("SELECT d.*, u.username AS owner FROM documents d LEFT JOIN users u ON u.id = d.owner_id ")

    @staticmethod
    def _doc(r: sqlite3.Row) -> DocumentRow:
        return DocumentRow(r["doc_id"], r["owner_id"], r["owner"], r["title"], r["filename"], r["path"],
                           r["visibility"], r["status"], r["error"], r["chunks"], r["created_at"])

    def add_document(self, doc_id: str, owner_id: int | None, title: str, filename: str | None, path: str | None,
                     visibility: str, status: str, chunks: int = 0) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO documents (doc_id, owner_id, title, filename, path, visibility, status, chunks, "
                      "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (doc_id, owner_id, title, filename, path, visibility, status, chunks, now()))

    def update_document(self, doc_id: str, **fields) -> None:
        allowed = {"title", "status", "error", "chunks"}
        if not fields or set(fields) - allowed:
            raise ValueError(f"Dozwolone pola: {allowed}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE documents SET {cols} WHERE doc_id = ?", (*fields.values(), doc_id))

    def get_document(self, doc_id: str) -> DocumentRow | None:
        with self._conn() as c:
            r = c.execute(self._DOC_SELECT + "WHERE d.doc_id = ?", (doc_id,)).fetchone()
        return self._doc(r) if r else None

    def list_documents(self, owner_id: int | None = None, all_owners: bool = False) -> list[DocumentRow]:
        with self._conn() as c:
            if all_owners:
                rows = c.execute(self._DOC_SELECT + "ORDER BY d.created_at DESC").fetchall()
            else:
                rows = c.execute(self._DOC_SELECT + "WHERE d.owner_id = ? ORDER BY d.created_at DESC",
                                 (owner_id,)).fetchall()
        return [self._doc(r) for r in rows]

    def delete_document(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def fail_unfinished(self, message: str) -> int:
        """Po restarcie serwera dokumenty w trakcie przetwarzania nie dokończą się same."""
        with self._conn() as c:
            return c.execute("UPDATE documents SET status = 'error', error = ? WHERE status = 'processing'",
                             (message,)).rowcount
