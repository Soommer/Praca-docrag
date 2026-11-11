"""Profil dla polskich aktów prawnych (ustawy, kodeksy, rozporządzenia).

Budowę aktów narzucają "Zasady techniki prawodawczej", więc jeden parser obsługuje
większość aktów z ISAP:
  * hierarchia: CZĘŚĆ > KSIĘGA > TYTUŁ > DZIAŁ > Rozdział > Oddział
  * jednostka podstawowa: Art. N  (w rozporządzeniach: § N)
  * jednostki wewnętrzne: § N (kodeksy) lub N. (ustępy), punkty N), litery a), tiret
Chunk = artykuł; za długi artykuł dzielony po jednostkach wewnętrznych,
z artykułem jako rodzicem (small-to-big).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..converters import read_legal_lines
from ..models import Chunk, ChunkResult, DocMeta, LegalCfg
from ..splitting import split_text
from .base import Profile

log = logging.getLogger(__name__)

SUP = r"(?:\^\d+[a-z]{0,2})?"          # indeks górny, też z literą: 505^1a
NUM = rf"\d+[a-z]{{0,2}}{SUP}"
ART_RE = re.compile(rf"^Art\.\s*({NUM})(?:\s*[–-]\s*({NUM}))?\.\s*(.*)$")
PAR_TOP_RE = re.compile(rf"^§\s*({NUM})(?:\s*[–-]\s*({NUM}))?\.\s*(.*)$")
PAR_UNIT_RE = re.compile(rf"^§\s*({NUM})\.(?:\s|$)")
UST_UNIT_RE = re.compile(rf"^({NUM})\.(?:\s|$)")
ITEM_RE = re.compile(rf"^(?:{NUM}\)|[a-z]{SUP}\)|[–—-]\s)")
REPEALED_RE = re.compile(
    r"^\((?:uchylon[yaei]|pominięt[yaei]|utracił[a]? moc|utraciły moc|wygasł[a]?|skreślon[yaei])\)\.?$", re.I)
NOISE_RES = [re.compile(p) for p in (
    r"^©\s*Kancelaria Sejmu",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^s\.\s*\d+\s*/\s*\d+$",
)]

_ROMAN = rf"[IVXLC]+[a-z]?{SUP}"
_ARAB = rf"\d+[a-z]?{SUP}"
HEADINGS = [(lvl, re.compile(rf"^({pat})(?:\s*[.:–-]?\s+(.*))?$")) for lvl, pat in (
    (1, r"(?:CZĘŚĆ|Część)\s+[A-ZĄĆĘŁŃÓŚŹŻ][A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż]*"),
    (2, r"(?:KSIĘGA|Księga)\s+[A-ZĄĆĘŁŃÓŚŹŻ][A-ZĄĆĘŁŃÓŚŹŻa-ząćęłńóśźż]*"),
    (3, rf"(?:TYTUŁ|Tytuł)\s+(?:WSTĘPNY|Wstępny|{_ROMAN})"),
    (4, rf"(?:DZIAŁ|Dział)\s+(?:{_ROMAN}|{_ARAB})"),
    (5, rf"(?:ROZDZIAŁ|Rozdział)\s+(?:{_ARAB}|{_ROMAN})"),
    (6, rf"(?:ODDZIAŁ|Oddział)\s+(?:{_ARAB}|{_ROMAN})"),
)]


def num_key(num: str) -> tuple[int, str, int, str]:
    m = re.fullmatch(r"(\d+)([a-z]{0,2})(?:\^(\d+)([a-z]{0,2}))?", num)
    if not m:
        return (0, "", 0, "")
    return (int(m.group(1)), m.group(2), int(m.group(3) or 0), m.group(4) or "")


def match_heading(line: str):
    for lvl, rx in HEADINGS:
        m = rx.match(line)
        if m:
            rest = (m.group(2) or "").strip()
            if rest and rest[0].islower():
                return None          # "Rozdział 2 stosuje się..." - zawinięty tekst, nie nagłówek
            return lvl, m.group(1), rest
    return None


def join_lines(lines: list[str]) -> str:
    """Skleja zawinięte linie: dzielenie wyrazów, nowe linie przed punktami i literami."""
    out = ""
    for ln in (x.strip() for x in lines):
        if not ln:
            continue
        if not out:
            out = ln
        elif re.search(r"[A-Za-ząćęłńóśźżĄĆĘŁŃÓŚŹŻ]-$", out) and ln[:1].islower():
            out = out[:-1] + ln
        elif ITEM_RE.match(ln):
            out += "\n" + ln
        else:
            out += " " + ln
    return out


# ------------------------------------------------------------------ odwołania

_REF_START = re.compile(r"\bart\.?\s*(?=\d)", re.I)
_REF_TOK = re.compile(rf"§|ust\.|pkt|lit\.|zd\.|zdanie|{NUM}|[–-]|,|[^\W\d]+\.?|\S")
_SUB_MARKERS = {"§", "ust.", "pkt", "lit.", "zd.", "zdanie"}
_CONNECTORS = {",", "i", "oraz", "lub", "albo", "–", "-", "a", "w", "z", "do", "pierwsze", "drugie",
               "trzecie", "zdaniu", "zdania", "in", "fine"}
_EXTERNAL = re.compile(r"ustaw|kodeks|k\.\s?[a-z]\.|rozporządz|konstytuc|konwencj|dyrektyw", re.I)


def extract_refs(text: str) -> list[str]:
    """Numery artykułów, do których odwołuje się tekst (tylko wewnątrz tego samego aktu)."""
    found: list[str] = []
    for m in _REF_START.finditer(text):
        window = text[m.end(): m.end() + 160]
        toks = _REF_TOK.findall(window)
        arts: list[str] = []
        mode_sub, pending_range, stop_idx = False, False, len(toks)
        for i, tok in enumerate(toks):
            low = tok.lower()
            if tok in _SUB_MARKERS:
                mode_sub = True
            elif low in ("art.", "art"):
                mode_sub = False
            elif re.fullmatch(NUM, tok):
                if not mode_sub:
                    if pending_range and arts:
                        arts.append("RANGE")
                    arts.append(tok)
                pending_range = False
            elif tok in ("–", "-"):
                pending_range = True
            elif low in _CONNECTORS:
                continue
            else:
                stop_idx = i
                break
        tail = " ".join(toks[stop_idx: stop_idx + 6])
        if _EXTERNAL.search(tail) and not re.search(r"niniejsz", tail, re.I):
            continue
        found.extend(_expand(arts))
    seen, out = set(), []
    for a in found:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _expand(arts: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(arts):
        if arts[i] == "RANGE" and out and i + 1 < len(arts):
            a, b = out[-1], arts[i + 1]
            if a.isdigit() and b.isdigit() and 0 < int(b) - int(a) <= 30:
                out.extend(str(n) for n in range(int(a) + 1, int(b) + 1))
            else:
                out.append(b)
            i += 2
            continue
        if arts[i] != "RANGE":
            out.append(arts[i])
        i += 1
    return out


# ------------------------------------------------------------------ parser

@dataclass
class _Heading:
    level: int
    marker: str
    title_lines: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        title = " ".join(self.title_lines).strip()
        return f"{self.marker}. {title}" if title else self.marker


@dataclass
class _Article:
    num: str
    end: str | None
    lines: list[str]
    path: list[_Heading]


class LegalActProfile(Profile):
    name = "legal"
    default_doc_type = "akt_prawny"

    def __init__(self, chunking, count, legal: LegalCfg | None = None):
        super().__init__(chunking, count)
        self.legal = legal or LegalCfg()

    @classmethod
    def detect(cls, sample: str, path: Path) -> float:
        lines = sample.splitlines()
        arts = sum(1 for ln in lines if ART_RE.match(ln.strip()))
        pars = sum(1 for ln in lines if PAR_TOP_RE.match(ln.strip()))
        header = re.search(r"\b(USTAWA|ROZPORZĄDZENIE|KODEKS|Kodeks|OBWIESZCZENIE)\b", sample[:5000])
        score = min(1.0, max(arts, pars) / 10)
        return score if header else score * 0.6

    # --- etap 1: linie -> artykuły z hierarchią
    def parse(self, lines: list[str]) -> tuple[list[str], list[_Article], str]:
        clean = [ln.strip() for ln in lines]
        clean = [ln for ln in clean if ln and not any(r.match(ln) for r in NOISE_RES)]
        n_art = sum(1 for ln in clean if ART_RE.match(ln))
        n_par = sum(1 for ln in clean if PAR_TOP_RE.match(ln))
        mode = "art" if n_art >= 3 or n_art >= n_par else "par"
        top_re = ART_RE if mode == "art" else PAR_TOP_RE

        preamble: list[str] = []
        articles: list[_Article] = []
        levels: dict[int, _Heading] = {}
        collecting: _Heading | None = None
        cur: _Article | None = None
        last_key = (-1, "", 0, "")

        for ln in clean:
            h = match_heading(ln)
            if h:
                lvl, marker, rest = h
                heading = _Heading(lvl, marker, [rest] if rest else [])
                for k in [k for k in levels if k >= lvl]:
                    del levels[k]
                levels[lvl] = heading
                collecting = heading
                if cur:
                    articles.append(cur)
                    cur = None
                continue
            m = top_re.match(ln)
            if m and num_key(m.group(1)) > last_key:
                if cur:
                    articles.append(cur)
                num, end, rest = m.group(1), m.group(2), m.group(3)
                last_key = num_key(end or num)
                path = [levels[k] for k in sorted(levels)]
                cur = _Article(num, end, [rest] if rest else [], path)
                collecting = None
                continue
            if collecting is not None and len(collecting.title_lines) < 3:
                collecting.title_lines.append(ln)
            elif cur is not None:
                cur.lines.append(ln)
            else:
                preamble.append(ln)
        if cur:
            articles.append(cur)
        return preamble, articles, mode

    # --- etap 2: artykuł -> jednostki (§ / ustępy)
    @staticmethod
    def split_units(lines: list[str], mode: str) -> list[tuple[str | None, str]]:
        units: list[tuple[str | None, list[str]]] = []
        last_ust = 0
        for ln in lines:
            label = None
            if mode == "art" and (m := PAR_UNIT_RE.match(ln)):
                label = f"§ {m.group(1)}"
            elif (m := UST_UNIT_RE.match(ln)):
                k = num_key(m.group(1))
                if k[0] == last_ust + 1 or (k[0] == last_ust and (k[1] or k[2] or k[3])) or (not units and k[0] == 1):
                    label = f"ust. {m.group(1)}"
                    last_ust = k[0]
            if label:
                units.append((label, [ln]))
            elif units:
                units[-1][1].append(ln)
            else:
                units.append((None, [ln]))
        return [(lab, join_lines(ls)) for lab, ls in units if join_lines(ls)]

    # --- etap 3: chunki
    def chunk(self, path: Path, meta: DocMeta) -> ChunkResult:
        lines = read_legal_lines(path, drop_footnotes=self.legal.drop_footnotes)
        preamble, articles, mode = self.parse(lines)
        prefix, word = ("art", "Art.") if mode == "art" else ("par", "§")
        chunks: list[Chunk] = []
        repealed = 0

        for art in articles:
            num_label = f"{art.num}–{art.end}" if art.end else art.num
            art_id = f"{meta.doc_id}#{prefix}-{num_label}"
            units = self.split_units(art.lines, mode)
            body = "\n".join(t for _, t in units)
            is_repealed = bool(REPEALED_RE.match(body.strip())) or not body.strip()
            if is_repealed:
                repealed += 1
                if self.legal.skip_repealed:
                    continue
            header = f"{word} {num_label}."
            full = f"{header} {body}".strip()
            section_path = [h.label for h in art.path]
            unit_name = f"{word.lower().rstrip('.')} {num_label}".replace("art ", "art. ")
            refs = [f"{meta.doc_id}#{prefix}-{r}" for r in extract_refs(body)]
            refs = [r for r in refs if r != art_id]
            extra = {"article": num_label, "repealed": is_repealed}

            if self.count(full) <= self.cfg.max_tokens:
                chunks.append(Chunk(art_id, full, section_path, unit_name, refs=refs, extra=extra,
                                    token_count=self.count(full)))
                continue

            # za długi artykuł: pakowanie jednostek, artykuł jako rodzic
            groups: list[list[tuple[str | None, str]]] = []
            cur: list[tuple[str | None, str]] = []
            budget = self.cfg.max_tokens - self.count(header) - 4
            for lab, txt in units:
                pieces = split_text(txt, self.count, budget, self.cfg.overlap_sentences)
                for p in pieces:
                    if cur and self.count("\n".join(t for _, t in cur + [(lab, p)])) > budget:
                        groups.append(cur)
                        cur = []
                    cur.append((lab, p))
            if cur:
                groups.append(cur)
            parent_text = self.parent_or_none(full)
            for i, g in enumerate(groups, 1):
                labels = [lab for lab, _ in g if lab]
                rng = (labels[0] if len(set(labels)) == 1 else f"{labels[0]}–{labels[-1].split()[-1]}") if labels else ""
                head = header if i == 1 else f"{word} {num_label} (cd.)"
                text = f"{head} " + "\n".join(t for _, t in g)
                chunks.append(Chunk(f"{art_id}-{i}", text, section_path, f"{unit_name} {rng}".strip(),
                                    parent_id=art_id, parent_text=parent_text, refs=refs, extra=dict(extra),
                                    token_count=self.count(text)))

        meta_updates = {}
        title = _detect_title(preamble)
        if title:
            meta_updates["detected_title"] = title
        stats = {"mode": mode, "articles": len(articles), "repealed": repealed, "chunks": len(chunks)}
        if not articles:
            log.warning("%s: nie znaleziono żadnego artykułu - czy to na pewno akt prawny?", path.name)
        return ChunkResult(chunks, meta_updates, stats)


def _detect_title(preamble: list[str]) -> str | None:
    for i, ln in enumerate(preamble):
        if re.match(r"^z dnia \d{1,2} \w+ \d{4} r\.?$", ln) and i + 1 < len(preamble):
            nxt = preamble[i + 1]
            if not nxt.startswith(("(", "Dz.", "Opracowano")):
                return nxt.strip()
    return None
