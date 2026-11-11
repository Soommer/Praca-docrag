"""API HTTP (FastAPI): logowanie, dokumenty użytkownika i pytania.

  POST   /api/login              {"username", "password"} -> ciasteczko sesji (api.auth: local)
  POST   /api/logout
  GET    /api/me                 kto jest zalogowany (user = null, gdy nikt)
  GET    /api/documents          moje dokumenty (admin: wszystkie, także z CLI)
  POST   /api/documents          wgranie pliku (multipart: file, title?, public?) -> ingest w tle
  DELETE /api/documents/{id}     usunięcie: właściciel albo admin
  POST   /api/ask                {"question": "..."} -> {"answer": "...", "sources": [...]}
  POST   /api/ask/stream         to samo jako Server-Sent Events: sources, token..., done
  POST   /api/search             same źródła, bez LLM
  GET    /api/health             stan Qdranta, modeli i LLM
  GET    /                       frontend (src/docrag/web)

Uruchomienie: docrag serve  (albo: uvicorn --factory docrag.api:create_app, config z DOCRAG_CONFIG).
Modele na GPU są wywoływane pojedynczo (lock), generowanie LLM odbywa się poza lockiem.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .db import Database, DocumentRow
from .documents import Actor, DocumentError, DocumentService, user_group
from .llm import Answer, Answerer, LLMUnavailable
from .models import Config
from .pipeline import Pipeline
from .search import Source
from .store import SearchFilter

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
SESSION_COOKIE = "docrag_session"
NO_SOURCES_ANSWER = "Nie znalazłem odpowiedzi w dokumentach, do których masz dostęp."
LLM_DOWN = "jest chwilowo niedostępny (serwer Ollama nie odpowiada)"


# ------------------------------------------------------------------ schematy

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["Ile mam czasu na wniesienie apelacji?"])
    doc_ids: list[str] = Field(default_factory=list, description="ogranicz do tych dokumentów")
    doc_types: list[str] = Field(default_factory=list, description="ogranicz do tych typów dokumentów")


class SearchRequest(AskRequest):
    top_k: int | None = Field(None, ge=1, le=50)


class SourceOut(BaseModel):
    n: int = Field(description="numer źródła - odpowiada odwołaniom [n] w odpowiedzi")
    doc_id: str
    title: str
    unit: str | None
    section_path: list[str]
    version: str | None
    score: float
    related_to: str | None = Field(description="źródło pomocnicze: przywołane w źródle o tym unit_id")
    text: str


class AskResponse(BaseModel):
    answer: str
    llm: bool = Field(description="false: model językowy niepodłączony - odpowiedź to lista źródeł")
    sources: list[SourceOut]
    stats: dict


class SearchResponse(BaseModel):
    sources: list[SourceOut]
    related: list[SourceOut]
    stats: dict


class LoginRequest(BaseModel):
    username: str
    password: str


class DocumentOut(BaseModel):
    doc_id: str
    title: str
    filename: str | None
    owner: str | None = Field(description="null: dokument systemowy (wgrany przez CLI)")
    visibility: str = Field(description="private | public | restricted")
    status: str = Field(description="processing | ready | error")
    error: str | None
    chunks: int
    created_at: str
    can_delete: bool


@dataclass
class User:
    id: int | None                                   # None: niezalogowany (api.auth: none)
    name: str | None
    is_admin: bool = False
    groups: list[str] = field(default_factory=list)  # grupy AD z proxy (api.auth: header)

    @property
    def search_groups(self) -> list[str]:
        """Grupy ACL do filtra wyszukiwania: grupy AD + prywatna grupa użytkownika (+ "all" w SearchFilter)."""
        return self.groups + ([user_group(self.id)] if self.id is not None else [])

    @property
    def actor(self) -> Actor:
        return Actor(self.id, self.name, self.is_admin)


# ------------------------------------------------------------------ silnik

class Engine:
    def __init__(self, cfg: Config, answerer: Answerer | None = None, db: Database | None = None,
                 background: bool = True):
        self.cfg = cfg
        self.answerer = answerer or Answerer(cfg)
        self.gpu_lock = threading.Lock()
        self.db = db or Database(cfg.api.db_path)
        s = self.answerer.searcher
        # ingest korzysta z tych samych modeli i połączenia z Qdrantem co wyszukiwanie (leniwie)
        self.pipeline = Pipeline(cfg, embedder=_Lazy(lambda: s.embedder), store=_Lazy(lambda: s.store))
        self.docs = DocumentService(self.db, self.pipeline, _Lazy(lambda: s.store), cfg.api.upload_dir,
                                    cfg.api.max_upload_mb, self.gpu_lock, background)

    @property
    def llm(self):
        return self.answerer.llm

    def warmup(self) -> None:
        t0 = time.perf_counter()
        with self.gpu_lock:
            s = self.answerer.searcher
            s.embedder.embed_query("rozgrzewka")
            s.reranker.score("rozgrzewka", ["rozgrzewka"])
        log.info("Modele załadowane w %.1fs", time.perf_counter() - t0)

    def start(self) -> None:
        n = self.db.fail_unfinished("Przetwarzanie przerwane (restart serwera) - wgraj plik ponownie")
        if n:
            log.warning("%d dokumentów oznaczono jako błędne po restarcie", n)
        try:
            n = self.docs.sync_from_store()
            if n:
                log.info("Zarejestrowano %d dokumentów z Qdranta (wgranych przez CLI)", n)
        except Exception as e:
            log.warning("Nie udało się odczytać dokumentów z Qdranta: %s", e)

    def prepare(self, question: str, flt: SearchFilter) -> Answer:
        with self.gpu_lock:
            return self.answerer.prepare(question, flt)

    def search(self, question: str, flt: SearchFilter, top_k: int | None):
        with self.gpu_lock:
            return self.answerer.searcher.search(question, flt, top_k)


class _Lazy:
    """Odwołanie do obiektu tworzonego dopiero przy pierwszym użyciu (modele, klient Qdranta)."""

    def __init__(self, factory):
        self._factory = factory

    def __getattr__(self, name):
        return getattr(self._factory(), name)


def sources_out(sources: list[Source]) -> list[SourceOut]:
    return [SourceOut(n=i, doc_id=s.doc_id, title=s.title, unit=s.unit, section_path=s.section_path,
                      version=s.meta.get("version"), score=s.score, related_to=s.related_to, text=s.text)
            for i, s in enumerate(sources, 1)]


def document_out(d: DocumentRow, user: User) -> DocumentOut:
    return DocumentOut(doc_id=d.doc_id, title=d.title, filename=d.filename, owner=d.owner,
                       visibility=d.visibility, status=d.status, error=d.error, chunks=d.chunks,
                       created_at=d.created_at,
                       can_delete=user.is_admin or (user.id is not None and d.owner_id == user.id))


def fallback_answer(sources: list[Source], chars: int = 300, reason: str = "nie jest podłączony") -> str:
    """Odpowiedź bez LLM: najlepiej pasujące fragmenty z numerami źródeł."""
    if not sources:
        return NO_SOURCES_ANSWER
    lines = [f"Model językowy {reason}. Najlepiej pasujące fragmenty:"]
    for i, s in enumerate(sources, 1):
        text = " ".join(s.text.split())
        lines.append(f"[{i}] {s.label}: {text[:chars]}{' ...' if len(text) > chars else ''}")
    return "\n".join(lines)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------ aplikacja

def create_app(cfg: Config | None = None, engine: Engine | None = None) -> FastAPI:
    cfg = cfg or Config.load(os.environ.get("DOCRAG_CONFIG", "config.yaml"))
    api = cfg.api
    if api.auth not in ("local", "header", "none"):
        raise ValueError(f"Nieznany tryb api.auth: {api.auth}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = app.state.engine = engine or Engine(cfg)
        eng.start()
        if api.warmup and engine is None:
            eng.warmup()
        yield
        eng.docs.shutdown()

    app = FastAPI(title="docrag", version=__version__, lifespan=lifespan,
                  description="Pytania do dokumentów i aktów prawnych (RAG).")
    if api.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=api.cors_origins, allow_credentials=True,
                           allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

    def get_engine(request: Request) -> Engine:
        return request.app.state.engine

    def current_user(request: Request, engine: Engine = Depends(get_engine)) -> User | None:
        """Zalogowany użytkownik albo None (bez wyjątku - dla /api/me)."""
        if api.auth == "none":
            return User(None, None)
        if api.auth == "header":
            name = request.headers.get(api.user_header)
            if not name:
                return None
            row = engine.db.ensure_user(name, name in api.admin_users)
            raw = request.headers.get(api.groups_header, "")
            return User(row.id, row.username, row.is_admin,
                        [g.strip() for g in raw.split(api.groups_separator) if g.strip()])
        token = request.cookies.get(SESSION_COOKIE)
        row = engine.db.session_user(token) if token else None
        return User(row.id, row.username, row.is_admin) if row else None

    def get_user(user: User | None = Depends(current_user)) -> User:
        if user is None:
            raise HTTPException(401, "Zaloguj się")
        return user

    def question_of(req: AskRequest) -> str:
        q = req.question.strip()
        if not q:
            raise HTTPException(422, "Pytanie jest puste")
        if len(q) > api.max_question_chars:
            raise HTTPException(422, f"Pytanie dłuższe niż {api.max_question_chars} znaków")
        return q

    def flt_of(req: AskRequest, user: User) -> SearchFilter:
        return SearchFilter(groups=user.search_groups, doc_ids=req.doc_ids, doc_types=req.doc_types)

    def prepare(engine: Engine, req: AskRequest, user: User) -> Answer:
        q = question_of(req)
        try:
            ans = engine.prepare(q, flt_of(req, user))
        except Exception:
            log.exception("Błąd wyszukiwania")
            raise HTTPException(503, "Wyszukiwarka jest niedostępna")
        log.info("ask user=%s sources=%d %s", user.name, len(ans.sources), ans.stats)
        return ans

    # -------------------------------------------------------------- logowanie
    @app.post("/api/login", summary="Logowanie (api.auth: local) - ustawia ciasteczko sesji")
    def login(req: LoginRequest, response: Response, engine: Engine = Depends(get_engine)):
        if api.auth != "local":
            raise HTTPException(400, f"Logowanie hasłem jest wyłączone (api.auth: {api.auth})")
        row = engine.db.authenticate(req.username, req.password)
        if row is None:
            log.warning("nieudane logowanie: %s", req.username)
            raise HTTPException(401, "Błędny login lub hasło")
        token = engine.db.create_session(row.id, api.session_hours)
        response.set_cookie(SESSION_COOKIE, token, max_age=api.session_hours * 3600, httponly=True,
                            samesite="lax", secure=api.cookie_secure)
        return {"user": row.username, "is_admin": row.is_admin}

    @app.post("/api/logout", summary="Wylogowanie")
    def logout(request: Request, response: Response, engine: Engine = Depends(get_engine)):
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            engine.db.delete_session(token)
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @app.get("/api/me", summary="Zalogowany użytkownik (user = null, gdy nikt)")
    def me(user: User | None = Depends(current_user)):
        if user is None:
            return {"user": None, "is_admin": False, "auth": api.auth}
        return {"user": user.name, "is_admin": user.is_admin, "auth": api.auth,
                "groups": SearchFilter(groups=user.search_groups).allowed_groups()}

    # -------------------------------------------------------------- dokumenty
    def run(fn, *args):
        try:
            return fn(*args)
        except DocumentError as e:
            raise HTTPException(e.status, e.detail)

    @app.get("/api/documents", response_model=list[DocumentOut], summary="Moje dokumenty (admin: wszystkie)")
    def list_documents(user: User = Depends(get_user), engine: Engine = Depends(get_engine)):
        return [document_out(d, user) for d in run(engine.docs.list, user.actor)]

    @app.post("/api/documents", response_model=DocumentOut, status_code=202,
              summary="Wgranie dokumentu - przetwarzanie w tle (status: processing -> ready | error)")
    def upload_document(file: UploadFile = File(...), title: str = Form(""), public: bool = Form(False),
                        user: User = Depends(get_user), engine: Engine = Depends(get_engine)):
        d = run(engine.docs.upload, user.actor, file.filename, file.file, title, public)
        return document_out(d, user)

    @app.delete("/api/documents/{doc_id}", status_code=204, summary="Usunięcie: właściciel albo admin")
    def delete_document(doc_id: str, user: User = Depends(get_user), engine: Engine = Depends(get_engine)):
        run(engine.docs.delete, user.actor, doc_id)
        return Response(status_code=204)

    # -------------------------------------------------------------- pytania
    @app.post("/api/ask", response_model=AskResponse, summary="Pytanie -> odpowiedź ze źródłami")
    def ask(req: AskRequest, user: User = Depends(get_user), engine: Engine = Depends(get_engine)):
        ans = prepare(engine, req, user)
        used_llm = False
        if not ans.sources:
            text = NO_SOURCES_ANSWER
        elif engine.llm is None:
            text = fallback_answer(ans.sources)
        else:
            try:
                text = engine.answerer.complete(ans).text or ""
                used_llm = True
            except LLMUnavailable as e:
                log.warning("%s - odpowiedź samymi źródłami", e)
                text = fallback_answer(ans.sources, reason=LLM_DOWN)
            except Exception as e:
                log.exception("Błąd LLM")
                raise HTTPException(502, f"Model językowy nie odpowiedział: {e}")
        return AskResponse(answer=text, llm=used_llm, sources=sources_out(ans.sources), stats=ans.stats)

    @app.post("/api/ask/stream", summary="Jak /api/ask, odpowiedź strumieniowana (Server-Sent Events)",
              response_class=StreamingResponse)
    def ask_stream(req: AskRequest, user: User = Depends(get_user), engine: Engine = Depends(get_engine)):
        """Zdarzenia: `sources` (lista źródeł), `token` ({"text"}) wielokrotnie, `done` (statystyki)
        albo `error` ({"detail"})."""
        ans = prepare(engine, req, user)

        def events():
            yield _sse("sources", [s.model_dump() for s in sources_out(ans.sources)])
            used_llm = False
            try:
                if not ans.sources or engine.llm is None:
                    yield _sse("token", {"text": fallback_answer(ans.sources)})
                else:
                    t0 = time.perf_counter()
                    try:
                        for chunk in engine.llm.stream(ans.messages):
                            yield _sse("token", {"text": chunk})
                        used_llm = True
                        ans.stats["ms_llm"] = round((time.perf_counter() - t0) * 1000)
                    except LLMUnavailable as e:      # zgłaszane przed pierwszym fragmentem
                        log.warning("%s - odpowiedź samymi źródłami", e)
                        yield _sse("token", {"text": fallback_answer(ans.sources, reason=LLM_DOWN)})
                yield _sse("done", {"llm": used_llm, "stats": ans.stats})
            except Exception as e:
                log.exception("Błąd LLM (stream)")
                yield _sse("error", {"detail": f"Model językowy nie odpowiedział: {e}"})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/search", response_model=SearchResponse, summary="Same źródła, bez LLM")
    def search(req: SearchRequest, user: User = Depends(get_user), engine: Engine = Depends(get_engine)):
        q = question_of(req)
        try:
            res = engine.search(q, flt_of(req, user), req.top_k)
        except Exception:
            log.exception("Błąd wyszukiwania")
            raise HTTPException(503, "Wyszukiwarka jest niedostępna")
        main = sources_out(res.sources)
        related = [o.model_copy(update={"n": len(main) + o.n}) for o in sources_out(res.related)]
        return SearchResponse(sources=main, related=related, stats=res.stats)

    @app.get("/api/health", summary="Stan usług")
    def health(engine: Engine = Depends(get_engine)):
        out: dict = {"auth": api.auth}
        try:
            store = engine.answerer.searcher.store
            info = store.client.get_collection(store.name)
            out["qdrant"] = {"ok": True, "collection": store.name, "points": info.points_count}
        except Exception as e:
            out["qdrant"] = {"ok": False, "error": str(e)}
        out["llm"] = {"backend": cfg.llm.backend, **(engine.llm.health() if engine.llm else {"ok": True})}
        out["status"] = "ok" if out["qdrant"]["ok"] and out["llm"]["ok"] else "degraded"
        return out

    # frontend (web/index.html, app.js, style.css) - ostatni, żeby nie przesłaniał /api
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
