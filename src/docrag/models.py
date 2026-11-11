"""Struktury danych: metadane dokumentu, chunk, konfiguracja."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DocMeta:
    doc_id: str
    title: str
    doc_type: str
    acl: list[str]
    source: str
    version: str | None = None
    valid_from: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section_path: list[str] = field(default_factory=list)
    unit: str | None = None          # np. "art. 369 § 1-2" albo nagłówek sekcji
    parent_id: str | None = None     # small-to-big: jednostka nadrzędna (artykuł, sekcja)
    parent_text: str | None = None
    refs: list[str] = field(default_factory=list)  # chunk_id/doc_id, do których tekst się odwołuje
    extra: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    meta_updates: dict[str, Any] = field(default_factory=dict)  # np. wykryty tytuł aktu
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- konfiguracja

@dataclass
class QdrantCfg:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection: str = "docs"
    path: str | None = None          # tryb lokalny bez serwera (testy): katalog na dane
    upsert_batch: int = 64


@dataclass
class EmbeddingCfg:
    backend: str = "bge-m3"          # bge-m3 | sentence-transformers
    model: str = "BAAI/bge-m3"
    device: str = "cuda:0"
    use_fp16: bool = True            # RTX 5070 (sm_120): fp16 i bf16 działają
    batch_size: int = 16
    max_length: int = 8192
    passage_prefix: str = ""         # niektóre modele (np. e5) wymagają prefiksu "passage: "
    query_prefix: str = ""           # ... i "query: " dla zapytań; BGE-M3 nie wymaga żadnego


@dataclass
class ChunkingCfg:
    max_tokens: int = 700
    min_tokens: int = 80
    overlap_sentences: int = 1
    parent_max_tokens: int = 4000    # większy rodzic nie trafia do payloadu
    use_model_tokenizer: bool = True


@dataclass
class LegalCfg:
    skip_repealed: bool = True
    drop_footnotes: bool = True


@dataclass
class DefaultsCfg:
    doc_type: str = "dokument"
    acl: list[str] = field(default_factory=lambda: ["all"])


@dataclass
class SearchCfg:
    prefetch_limit: int = 40         # kandydaci z każdego wektora (dense, sparse) przed fuzją RRF
    rerank_candidates: int = 40      # ile wyników fuzji trafia do rerankera
    top_k: int = 5                   # ile jednostek (artykułów/sekcji) zwraca wyszukiwanie
    reranker: str = "bge"            # bge | none
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda:0"
    reranker_fp16: bool = True
    reranker_batch_size: int = 16
    reranker_max_length: int = 1024
    min_score: float = 0.0           # próg znormalizowanego wyniku rerankera (0-1); 0 = bez progu
    expand_parents: bool = True      # small-to-big: fragment -> cały artykuł/sekcja
    max_refs: int = 3                # ile artykułów z refs dociągnąć jako kontekst pomocniczy
    context_max_tokens: int = 6000   # budżet tokenów na źródła w prompcie


@dataclass
class LLMCfg:
    backend: str = "none"            # none | ollama
    url: str = "http://localhost:11434"
    model: str = "Mistral-local:latest"
    temperature: float = 0.1
    num_ctx: int = 8192              # okno kontekstu Ollamy; musi pomieścić źródła + odpowiedź
    num_predict: int = 1024          # limit tokenów odpowiedzi (ochrona przed zapętleniem modelu)
    think: bool | None = False       # tryb "thinking" (Gemma 4, Qwen3...); None = domyślny modelu
    timeout: float = 300.0
    connect_timeout: float = 5.0     # Ollama na innym komputerze: wyłączony = źródła bez odpowiedzi po 5 s


@dataclass
class ApiCfg:
    # local  - logowanie loginem i hasłem (konta w SQLite, zakłada je admin: docrag user add)
    # header - użytkownik i grupy z nagłówków ustawianych przez reverse proxy po logowaniu w AD;
    #          API musi być wtedy dostępne WYŁĄCZNIE przez proxy (inaczej nagłówki można podrobić)
    # none   - bez logowania: każdy widzi tylko dokumenty z acl "all" i nie może wgrywać
    auth: str = "local"
    user_header: str = "X-Remote-User"
    groups_header: str = "X-Remote-Groups"
    groups_separator: str = ","
    admin_users: list[str] = field(default_factory=list)   # tryb header: kto jest adminem
    db_path: str = "state/docrag.db"                       # SQLite: użytkownicy, sesje, rejestr dokumentów
    upload_dir: str = "state/uploads"                      # wgrane pliki (podkatalog na użytkownika)
    max_upload_mb: int = 50
    session_hours: int = 168                               # ważność logowania (7 dni)
    cookie_secure: bool = False                            # true, gdy serwer jest za HTTPS
    cors_origins: list[str] = field(default_factory=list)
    max_question_chars: int = 2000
    warmup: bool = True              # załaduj modele przy starcie, a nie przy pierwszym pytaniu


@dataclass
class Config:
    qdrant: QdrantCfg = field(default_factory=QdrantCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)
    legal: LegalCfg = field(default_factory=LegalCfg)
    defaults: DefaultsCfg = field(default_factory=DefaultsCfg)
    search: SearchCfg = field(default_factory=SearchCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    api: ApiCfg = field(default_factory=ApiCfg)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            import yaml

            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            _merge(cfg, data)
        return cfg


def _merge(obj: Any, data: dict) -> None:
    names = {f.name for f in fields(obj)}
    for k, v in data.items():
        if k not in names:
            raise ValueError(f"Nieznany klucz konfiguracji: {k}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(obj, k, v)
