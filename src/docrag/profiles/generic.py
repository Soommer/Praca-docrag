"""Profil ogólny: dowolny dokument -> Markdown -> drzewo nagłówków -> chunki.

* tnie po sekcjach, nigdy nie skleja tekstu z niepowiązanych sekcji,
* tabele, bloki kodu i listy traktuje jako niepodzielne bloki (o ile mieszczą się w limicie),
* za duże sekcje dzieli po blokach, potem po zdaniach (z zakładką), ostatecznie po słowach,
* małe sąsiednie sekcje skleja w jeden chunk (ze wspólną ścieżką nagłówków),
* sekcja podzielona na kilka chunków staje się rodzicem (small-to-big).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..converters import to_markdown
from ..models import Chunk, ChunkResult, DocMeta
from ..splitting import pack_blocks
from ..utils import slugify
from .base import Profile

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
LINK_RE = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
WIKI_RE = re.compile(r"\[\[([^\]|#]+)")


@dataclass
class Section:
    path: list[str]
    heading: str | None
    level: int
    blocks: list[str] = field(default_factory=list)


def parse_markdown(md: str) -> tuple[str | None, list[Section]]:
    """Zwraca (tytuł z pierwszego H1, lista sekcji w kolejności)."""
    sections = [Section([], None, 0)]
    stack: list[tuple[int, str]] = []
    title = None
    buf: list[str] = []
    kind = None  # para | table | list | code

    def flush():
        nonlocal buf, kind
        if buf and any(x.strip() for x in buf):
            sections[-1].blocks.append("\n".join(buf).strip("\n"))
        buf, kind = [], None

    for line in md.splitlines():
        if kind == "code":
            buf.append(line)
            if line.strip().startswith("```"):
                flush()
            continue
        if line.strip().startswith("```"):
            flush()
            kind = "code"
            buf.append(line)
            continue
        m = HEADING_RE.match(line)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2).strip()
            if title is None and level == 1:
                title = text
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            sections.append(Section([t for _, t in stack], text, level))
            continue
        if not line.strip():
            if kind != "list":
                flush()
            continue
        new_kind = "table" if line.lstrip().startswith("|") else "list" if LIST_RE.match(line) else "para"
        if kind == "list" and new_kind == "para" and line.startswith((" ", "\t")):
            new_kind = "list"  # wcięta kontynuacja punktu listy
        if kind and new_kind != kind:
            flush()
        kind = new_kind
        buf.append(line)
    flush()
    return title, [s for s in sections if s.blocks]


class GenericProfile(Profile):
    name = "generic"

    @classmethod
    def detect(cls, sample: str, path: Path) -> float:
        return 0.1  # zawsze możliwy fallback

    def chunk(self, path: Path, meta: DocMeta) -> ChunkResult:
        md = to_markdown(path)
        title, sections = parse_markdown(md)
        return self.chunk_sections(sections, meta, title, md)

    def chunk_sections(self, sections: list[Section], meta: DocMeta, title: str | None, md: str) -> ChunkResult:
        cnt = self.count
        # 1. sekcje -> jednostki (tekst sekcji z nagłówkiem)
        units = []
        for s in sections:
            head = f"{'#' * s.level} {s.heading}" if s.heading else None
            blocks = ([head] if head else []) + s.blocks
            text = "\n\n".join(blocks)
            units.append({"path": s.path, "heading": s.heading, "blocks": blocks, "text": text,
                          "tokens": cnt(text)})

        # 2. sklejanie małych sąsiednich sekcji
        merged = []
        for u in units:
            if merged:
                prev = merged[-1]
                common = _common_prefix(prev["path"], u["path"])
                if (prev["tokens"] < self.cfg.min_tokens or u["tokens"] < self.cfg.min_tokens) \
                        and prev["tokens"] + u["tokens"] <= self.cfg.max_tokens \
                        and (common or (not prev["path"] or not u["path"])):
                    prev["path"] = common
                    prev["heading"] = common[-1] if common else None
                    prev["blocks"] += u["blocks"]
                    prev["text"] += "\n\n" + u["text"]
                    prev["tokens"] = cnt(prev["text"])
                    continue
            merged.append(dict(u))

        # 3. jednostki -> chunki
        chunks: list[Chunk] = []
        used_ids: dict[str, int] = {}
        for u in merged:
            base = slugify("-".join(u["path"]))[:80] if u["path"] else "wstep"
            n = used_ids.get(base, 0)
            used_ids[base] = n + 1
            sec_id = f"{meta.doc_id}#{base}" + (f"-{n + 1}" if n else "")
            refs = _refs(u["text"], meta.doc_id)
            unit_name = u["heading"]
            if u["tokens"] <= self.cfg.max_tokens:
                chunks.append(Chunk(sec_id, u["text"], u["path"], unit_name, refs=refs, token_count=u["tokens"]))
                continue
            parts = pack_blocks(u["blocks"], cnt, self.cfg.max_tokens, overlap_sentences=self.cfg.overlap_sentences)
            parent_text = self.parent_or_none(u["text"])
            for i, p in enumerate(parts, 1):
                chunks.append(Chunk(f"{sec_id}-{i}", p, u["path"], unit_name, parent_id=sec_id,
                                    parent_text=parent_text, refs=refs, token_count=cnt(p)))

        meta_updates = {"detected_title": title} if title else {}
        stats = {"sections": len(sections), "chunks": len(chunks), "markdown_chars": len(md)}
        return ChunkResult(chunks, meta_updates, stats)


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def _refs(text: str, own_doc: str) -> list[str]:
    refs = []
    for target in LINK_RE.findall(text):
        if "://" in target or target.startswith("mailto:"):
            continue
        refs.append(slugify(Path(target).stem))
    refs += [slugify(t.strip()) for t in WIKI_RE.findall(text)]
    return sorted({r for r in refs if r and r != own_doc})
