"""docrag - CLI.

  docrag inspect data/kpc.pdf --show 3 --out chunks.jsonl   # podgląd chunków, bez GPU i bazy
  docrag ingest data/ --acl GG-Prawny                       # chunking + embeddingi + zapis
  docrag delete --doc-id kpc
  docrag stats
  docrag search "termin na apelację" --groups GG-Prawny            # wyszukiwanie + reranker
  docrag ask "Jaki jest termin na wniesienie apelacji?"            # + LLM (albo podgląd promptu)
  docrag serve --port 8008                                         # API HTTP, dokumentacja: /docs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .models import Config
from .pipeline import Pipeline


def _overrides(a) -> dict:
    return {"profile": a.profile, "doc_type": a.doc_type, "doc_id": a.doc_id, "title": a.title,
            "acl": a.acl, "version": a.version, "valid_from": a.valid_from}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="docrag", description="Ingest dokumentów do bazy wektorowej")
    ap.add_argument("-c", "--config", default=os.environ.get("DOCRAG_CONFIG", "config.yaml"),
                    help="domyślnie $DOCRAG_CONFIG albo config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def doc_args(p):
        p.add_argument("paths", nargs="+")
        p.add_argument("--profile", choices=["auto", "legal", "generic"], default=None)
        p.add_argument("--doc-type")
        p.add_argument("--doc-id", help="tylko dla pojedynczego pliku")
        p.add_argument("--title")
        p.add_argument("--acl", nargs="+", help="grupy AD z dostępem, np. GG-Prawny GG-IT")
        p.add_argument("--version", help="np. 'Dz.U. 2024 poz. 1568'")
        p.add_argument("--valid-from")

    p_in = sub.add_parser("inspect", help="pokaż chunki bez embeddingów")
    doc_args(p_in)
    p_in.add_argument("--show", type=int, default=0, help="wypisz N pełnych chunków")
    p_in.add_argument("--out", type=Path, help="zapisz chunki do JSONL")

    p_ing = sub.add_parser("ingest", help="chunking + embeddingi + zapis do Qdrant")
    doc_args(p_ing)
    p_ing.add_argument("--force", action="store_true", help="przeindeksuj mimo braku zmian")

    p_del = sub.add_parser("delete", help="usuń dokument z bazy")
    p_del.add_argument("--doc-id", required=True)

    sub.add_parser("stats", help="statystyki kolekcji")

    p_user = sub.add_parser("user", help="konta użytkowników (logowanie w API, api.auth: local)")
    usub = p_user.add_subparsers(dest="user_cmd", required=True)
    u_add = usub.add_parser("add", help="nowe konto")
    u_add.add_argument("username")
    u_add.add_argument("--admin", action="store_true", help="administrator (usuwa dokumenty wszystkich)")
    u_add.add_argument("--password", help="hasło (bez tej opcji: zapytanie w terminalu)")
    u_pw = usub.add_parser("passwd", help="zmiana hasła")
    u_pw.add_argument("username")
    u_pw.add_argument("--password")
    u_adm = usub.add_parser("admin", help="nadanie / odebranie uprawnień admina")
    u_adm.add_argument("username")
    u_adm.add_argument("--off", action="store_true", help="odbierz uprawnienia")
    u_del = usub.add_parser("del", help="usunięcie konta (jego dokumenty zostają jako systemowe)")
    u_del.add_argument("username")
    usub.add_parser("list", help="lista kont")

    def query_args(p):
        p.add_argument("question")
        p.add_argument("--groups", nargs="+", default=[], help="grupy AD użytkownika (\"all\" zawsze dodawane)")
        p.add_argument("--no-acl", action="store_true", help="bez filtra uprawnień (diagnostyka)")
        p.add_argument("--doc-id", nargs="+", default=[], help="tylko te dokumenty")
        p.add_argument("--doc-type", nargs="+", default=[], help="tylko te typy dokumentów")
        p.add_argument("--json", action="store_true", help="wynik jako JSON")

    p_s = sub.add_parser("search", help="wyszukiwanie hybrydowe + reranker")
    query_args(p_s)
    p_s.add_argument("--top", type=int, help="liczba źródeł (domyślnie search.top_k)")
    p_s.add_argument("--chars", type=int, default=500, help="ile znaków treści pokazać")

    p_a = sub.add_parser("ask", help="odpowiedź LLM na podstawie źródeł")
    query_args(p_a)
    p_a.add_argument("--show-prompt", action="store_true", help="wypisz prompt także przy podłączonym LLM")

    p_srv = sub.add_parser("serve", help="API HTTP (FastAPI): /api/ask, /api/search, /api/health")
    p_srv.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = dostęp z sieci")
    p_srv.add_argument("--port", type=int, default=8008)

    a = ap.parse_args(argv)
    try:  # certyfikaty z magazynu systemu - proxy/antywirus przechwytujący TLS (pobieranie modeli z HF)
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "pdfminer", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.load(a.config)

    if a.cmd == "user":
        return _user_cmd(a, cfg)

    if a.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_config=None)
        return 0

    pipe = Pipeline(cfg)

    if a.cmd == "inspect":
        rows = pipe.inspect(a.paths, _overrides(a), a.out)
        for r in rows:
            toks = [p["token_count"] for p in r["payloads"]]
            print(f"\n== {r['path']}  doc_id={r['doc_id']}  profil={r['profile']}  tytuł={r['title']!r}")
            print(f"   {r['stats']}")
            if toks:
                print(f"   tokeny: min={min(toks)} max={max(toks)} śr={sum(toks) // len(toks)} "
                      f"(liczone {'tokenizerem' if pipe.count.exact else 'w przybliżeniu'})")
                print(f"   z rodzicem: {sum(1 for p in r['payloads'] if p['parent_id'])}, "
                      f"z odwołaniami: {sum(1 for p in r['payloads'] if p['refs'])}")
                depth = Counter(len(p["section_path"]) for p in r["payloads"])
                print(f"   głębokość ścieżki: {dict(sorted(depth.items()))}")
            for p in r["payloads"][: a.show]:
                print("\n---", p["chunk_id"], f"({p['token_count']} tok.)", p["unit"] or "")
                print("context:", p["context"].replace("\n", " | "))
                print("refs:", p["refs"][:10])
                print(p["text"][:1500])
        return 0

    if a.cmd == "ingest":
        if a.doc_id and (len(a.paths) > 1 or Path(a.paths[0]).is_dir()):
            ap.error("--doc-id można podać tylko dla pojedynczego pliku")
        summary = pipe.ingest(a.paths, _overrides(a), force=a.force)
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if summary["failed"] else 0

    if a.cmd == "delete":
        pipe.store.delete_doc(a.doc_id)
        from .db import Database
        from .documents import _remove

        db = Database(cfg.api.db_path)           # rejestr dokumentów API (jeśli był tam zapisany)
        rec = db.get_document(a.doc_id)
        if rec:
            db.delete_document(a.doc_id)
            if rec.path:
                _remove(Path(rec.path))
        print(f"usunięto {a.doc_id}")
        return 0

    if a.cmd in ("search", "ask"):
        from .search import Searcher
        from .store import SearchFilter

        flt = SearchFilter(groups=None if a.no_acl else a.groups, doc_ids=a.doc_id, doc_types=a.doc_type)
        searcher = Searcher(cfg, embedder=pipe.embedder, store=pipe.store)
        if a.cmd == "search":
            res = searcher.search(a.question, flt, top_k=a.top)
            if a.json:
                print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
                return 0
            print(f"{res.stats}")
            for i, s in enumerate(res.sources, 1):
                _print_source(f"[{i}] {s.score:.3f}", s, a.chars)
            for s in res.related:
                _print_source(f"[+] pomocnicze <- {s.related_to}", s, a.chars)
            if not res.sources:
                print("Brak wyników.")
            return 0

        from .llm import Answerer

        answerer = Answerer(cfg, searcher=searcher, count=pipe.count)
        ans = answerer.prepare(a.question, flt)
        if not a.json and (a.show_prompt or answerer.llm is None):
            if answerer.llm is None:
                print("LLM nie jest podłączony (llm.backend: none) - poniżej prompt, który dostałby model.\n")
            for m in ans.messages:
                print(f"===== {m['role']} =====\n{m['content']}\n")
        if answerer.llm is not None:
            answerer.complete(ans, on_token=None if a.json else lambda t: print(t, end="", flush=True))
        if a.json:
            print(json.dumps({"question": ans.question, "answer": ans.text, "stats": ans.stats,
                              "messages": ans.messages, "sources": [asdict(s) for s in ans.sources]},
                             ensure_ascii=False, indent=2))
        elif answerer.llm is not None:
            print("\n\nŹródła:")
            for i, s in enumerate(ans.sources, 1):
                print(f"  [{i}] {s.label}" + (f"  ({s.meta['source']})" if s.meta.get("source") else ""))
        return 0

    if a.cmd == "stats":
        print(json.dumps(pipe.store.stats(), ensure_ascii=False, indent=2))
        return 0
    return 2


def _user_cmd(a, cfg) -> int:
    from getpass import getpass

    from .db import Database

    db = Database(cfg.api.db_path)

    def password() -> str:
        pw = a.password
        if not pw:
            pw = getpass("Hasło: ")
            if pw != getpass("Powtórz hasło: "):
                raise ValueError("Hasła się różnią")
        if len(pw) < 8:
            raise ValueError("Hasło musi mieć co najmniej 8 znaków")
        return pw

    try:
        if a.user_cmd == "add":
            u = db.add_user(a.username, password(), a.admin)
            print(f"utworzono {u.username}{' (admin)' if u.is_admin else ''}")
        elif a.user_cmd == "passwd":
            db.set_password(a.username, password())
            print(f"zmieniono hasło {a.username}")
        elif a.user_cmd == "admin":
            db.set_admin(a.username, not a.off)
            print(f"{a.username}: admin = {not a.off}")
        elif a.user_cmd == "del":
            db.delete_user(a.username)
            print(f"usunięto {a.username}")
        elif a.user_cmd == "list":
            for u in db.list_users():
                print(f"{u['username']:20} {'admin' if u['is_admin'] else '     '}  dokumentów: {u['documents']:3}  "
                      f"od {u['created_at'][:10]}")
    except ValueError as e:
        print(f"błąd: {e}", file=sys.stderr)
        return 1
    print(f"(baza: {db.path})")
    return 0


def _print_source(head: str, s, chars: int) -> None:
    print(f"\n{head}  {s.label}")
    if s.section_path:
        print(f"    {' > '.join(s.section_path)}")
    text = s.text if len(s.text) <= chars else s.text[:chars] + " [...]"
    print("    " + text.replace("\n", "\n    "))


if __name__ == "__main__":
    sys.exit(main())
