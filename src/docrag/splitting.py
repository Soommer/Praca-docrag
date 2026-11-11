"""Liczenie tokenów i dzielenie tekstu z limitem rozmiaru."""
from __future__ import annotations

import logging
from typing import Callable

from .utils import split_sentences

log = logging.getLogger(__name__)

CountFn = Callable[[str], int]


class TokenCounter:
    """Liczy tokeny tokenizerem modelu embeddingów; bez transformers - przybliżenie."""

    CHARS_PER_TOKEN = 3.3  # ostrożne przybliżenie dla polszczyzny (tokenizer XLM-R)

    def __init__(self, model_name: str | None = None, use_tokenizer: bool = True):
        self._tok = None
        if model_name and use_tokenizer:
            try:
                from transformers import AutoTokenizer

                self._tok = AutoTokenizer.from_pretrained(model_name)
                self._tok.model_max_length = 10**9  # wyłącza ostrzeżenia o długości
            except Exception as e:  # brak sieci/biblioteki -> przybliżenie
                log.warning("Tokenizer %s niedostępny (%s) - używam przybliżenia.", model_name, e)

    @property
    def exact(self) -> bool:
        return self._tok is not None

    def __call__(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return max(1, int(len(text) / self.CHARS_PER_TOKEN))


def hard_split(text: str, count: CountFn, max_tokens: int) -> list[str]:
    """Ostateczność: cięcie po słowach, gdy nawet pojedyncze zdanie jest za długie."""
    words = text.split()
    out, start = [], 0
    while start < len(words):
        best, end, step = start + 1, start + 1, 32
        while end <= len(words) and count(" ".join(words[start:end])) <= max_tokens:
            best, end = end, end + step
        end = best + 1
        while end <= len(words) and count(" ".join(words[start:end])) <= max_tokens:
            best, end = end, end + 1
        out.append(" ".join(words[start:best]))
        start = best
    return out


def split_text(text: str, count: CountFn, max_tokens: int, overlap_sentences: int = 1) -> list[str]:
    """Dzieli tekst po zdaniach (z zakładką), gdy przekracza max_tokens."""
    if count(text) <= max_tokens:
        return [text]
    sents = split_sentences(text)
    if len(sents) <= 1:
        return hard_split(text, count, max_tokens)

    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_tok = 0
    for s in sents:
        t = count(s)
        if t > max_tokens:
            if cur:
                chunks.append(" ".join(x for x, _ in cur))
                cur, cur_tok = [], 0
            chunks.extend(hard_split(s, count, max_tokens))
            continue
        if cur and cur_tok + t + 1 > max_tokens:
            chunks.append(" ".join(x for x, _ in cur))
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_tok = sum(n + 1 for _, n in cur)
            if cur_tok + t + 1 > max_tokens:
                cur, cur_tok = [], 0
        cur.append((s, t))
        cur_tok += t + 1
    if cur:
        chunks.append(" ".join(x for x, _ in cur))
    return chunks


def pack_blocks(blocks: list[str], count: CountFn, max_tokens: int, sep: str = "\n\n",
                overlap_sentences: int = 1) -> list[str]:
    """Skleja kolejne bloki (akapity, tabele, jednostki redakcyjne) do limitu tokenów."""
    pieces: list[str] = []
    for b in blocks:
        pieces.extend(split_text(b, count, max_tokens, overlap_sentences))
    out: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    sep_tok = 1
    for p in pieces:
        t = count(p)
        if cur and cur_tok + sep_tok + t > max_tokens:
            out.append(sep.join(cur))
            cur, cur_tok = [], 0
        cur.append(p)
        cur_tok += t + (sep_tok if len(cur) > 1 else 0)
    if cur:
        out.append(sep.join(cur))
    return out
