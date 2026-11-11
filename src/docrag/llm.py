"""Odpowiedź na pytanie: wyszukiwanie -> prompt ze źródłami [n] -> LLM.

Backend LLM jest wymienny (interfejs LLM). backend: none - bez modelu: `ask` zwraca gotowy
prompt i źródła, co pozwala ocenić wyszukiwanie zanim podłączymy model.
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from .models import Config, LLMCfg
from .search import Searcher, SearchResult, Source
from .splitting import CountFn, TokenCounter
from .store import SearchFilter

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """Jesteś asystentem odpowiadającym na pytania na podstawie dokumentów organizacji \
i aktów prawnych.
Zasady:
- Odpowiadaj wyłącznie na podstawie podanych źródeł. Nie korzystaj z własnej wiedzy o treści przepisów.
- Każde stwierdzenie opatrz odwołaniem do źródła w nawiasie kwadratowym, np. [1] lub [2][3].
- Przy aktach prawnych podawaj jednostkę redakcyjną (np. art. 505^1 § 1).
- Jeśli źródła nie zawierają odpowiedzi, napisz wprost, że nie znalazłeś jej w dokumentach.
- Źródła oznaczone jako pomocnicze to przepisy, do których odwołują się źródła główne.
- Odpowiadaj po polsku, zwięźle i rzeczowo."""

Message = dict[str, str]


# ------------------------------------------------------------------ prompt

def format_source(n: int, s: Source) -> str:
    head = f"[{n}] {s.label}"
    if s.meta.get("version"):
        head += f" ({s.meta['version']})"
    if s.section_path:
        head += f"\nŚcieżka: {' > '.join(s.section_path)}"
    if s.related_to:
        head += "\nŹródło pomocnicze (przywołane w innym źródle)"
    return f"{head}\n{s.text}"


def select_sources(result: SearchResult, count: CountFn, max_tokens: int) -> list[Source]:
    """Źródła główne w kolejności trafności, potem pomocnicze - dopóki mieszczą się w budżecie tokenów."""
    chosen, used = [], 0
    for s in result.sources + result.related:
        cost = count(format_source(len(chosen) + 1, s))
        if used + cost > max_tokens:
            if s.related_to is None and not chosen:
                log.warning("Najlepsze źródło (%s) przekracza budżet kontekstu - zostaje pominięte.", s.label)
            continue
        chosen.append(s)
        used += cost
    return chosen


def build_messages(question: str, sources: list[Source]) -> list[Message]:
    if sources:
        body = "\n\n".join(format_source(i, s) for i, s in enumerate(sources, 1))
    else:
        body = "(brak źródeł - nie znaleziono pasujących dokumentów)"
    user = f"Źródła:\n\n{body}\n\n---\nPytanie: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ LLM

class LLMUnavailable(RuntimeError):
    """Serwer LLM nieosiągalny (np. wyłączony komputer z Ollamą) - API odpowiada wtedy samymi źródłami."""


class LLM(ABC):
    @abstractmethod
    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Kolejne fragmenty odpowiedzi."""

    def chat(self, messages: list[Message]) -> str:
        return "".join(self.stream(messages))

    def health(self) -> dict:
        return {"ok": True}


class OllamaLLM(LLM):
    """Ollama /api/chat (lokalnie albo na serwerze z GPU)."""

    def __init__(self, cfg: LLMCfg):
        import httpx

        self.cfg = cfg
        self.httpx = httpx
        self.client = httpx.Client(base_url=cfg.url, timeout=httpx.Timeout(cfg.timeout, connect=cfg.connect_timeout))

    def stream(self, messages: list[Message]) -> Iterator[str]:
        body = {"model": self.cfg.model, "messages": messages, "stream": True,
                "options": {"temperature": self.cfg.temperature, "num_ctx": self.cfg.num_ctx,
                            "num_predict": self.cfg.num_predict}}
        if self.cfg.think is not None:
            body["think"] = self.cfg.think
        req = self.client.build_request("POST", "/api/chat", json=body)
        try:
            r = self.client.send(req, stream=True)
        except (self.httpx.ConnectError, self.httpx.ConnectTimeout) as e:
            raise LLMUnavailable(f"Ollama ({self.cfg.url}) niedostępna: {e}") from e
        try:
            if r.status_code != 200:
                r.read()
                raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise RuntimeError(f"Ollama: {msg['error']}")
                if chunk := msg.get("message", {}).get("content"):
                    yield chunk
                if msg.get("done"):
                    return
        finally:
            r.close()

    def health(self) -> dict:
        try:
            r = self.client.get("/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            name = self.cfg.model if ":" in self.cfg.model else self.cfg.model + ":latest"   # Ollama dopisuje tag
            return {"ok": name in models, "model": self.cfg.model,
                    **({} if name in models else {"error": "model niedostępny w Ollamie"})}
        except Exception as e:
            return {"ok": False, "model": self.cfg.model, "error": str(e)}


def build_llm(cfg: LLMCfg) -> LLM | None:
    if cfg.backend == "none":
        return None
    if cfg.backend == "ollama":
        return OllamaLLM(cfg)
    raise ValueError(f"Nieznany backend LLM: {cfg.backend}")


# ------------------------------------------------------------------ całość

@dataclass
class Answer:
    question: str
    sources: list[Source]            # źródła w prompcie; numeracja [n] = indeks + 1
    messages: list[Message]
    search: SearchResult
    text: str | None = None          # None, gdy LLM nie jest skonfigurowany
    stats: dict = field(default_factory=dict)


class Answerer:
    def __init__(self, cfg: Config, searcher: Searcher | None = None, llm: LLM | None = None,
                 count: CountFn | None = None):
        self.cfg = cfg
        self.searcher = searcher or Searcher(cfg)
        self.llm = llm if llm is not None else build_llm(cfg.llm)
        self.count = count or TokenCounter(cfg.embedding.model, cfg.chunking.use_model_tokenizer)

    def prepare(self, question: str, flt: SearchFilter) -> Answer:
        res = self.searcher.search(question, flt)
        sources = select_sources(res, self.count, self.cfg.search.context_max_tokens)
        return Answer(question, sources, build_messages(question, sources), res, stats=dict(res.stats))

    def complete(self, ans: Answer, on_token=None) -> Answer:
        """Wywołuje LLM na przygotowanym prompcie; on_token(str) - opcjonalny streaming fragmentów."""
        if self.llm is None:
            return ans
        t0 = time.perf_counter()
        parts = []
        for chunk in self.llm.stream(ans.messages):
            parts.append(chunk)
            if on_token:
                on_token(chunk)
        ans.text = "".join(parts)
        ans.stats["ms_llm"] = round((time.perf_counter() - t0) * 1000)
        return ans

    def ask(self, question: str, flt: SearchFilter, on_token=None) -> Answer:
        return self.complete(self.prepare(question, flt), on_token)
