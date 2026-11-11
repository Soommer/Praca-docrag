"""Konwersja plików do tekstu.

* read_legal_lines  - linie tekstu aktu prawnego; z PDF wyciągane znak po znaku,
                      żeby zachować indeksy górne (Art. 505^1) i odrzucić przypisy.
* to_markdown       - dowolny dokument -> Markdown z nagłówkami (Docling, z fallbackiem).
* sample_text       - szybka próbka tekstu do automatycznego wyboru profilu.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from .utils import normalize_superscripts

log = logging.getLogger(__name__)

TEXT_EXT = {".txt", ".md", ".markdown"}
DOCLING_EXT = {".pdf", ".docx", ".pptx", ".html", ".htm", ".xlsx"}
SUPPORTED_EXT = TEXT_EXT | DOCLING_EXT


# ----------------------------------------------------------------- PDF aktów prawnych

def _pdf_body_size(pdf) -> float:
    sizes: Counter = Counter()
    for page in pdf.pages[: min(15, len(pdf.pages))]:
        for c in page.chars:
            if c["text"].strip():
                sizes[round(c["size"], 1)] += 1
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _page_lines(page, body: float, drop_footnotes: bool) -> list[str]:
    # jawne spacje z PDF zostają - przy justowaniu odstęp bywa mniejszy niż próg geometryczny
    chars = [c for c in page.chars if c["text"].strip() or c["text"] == " "]
    normal = sorted((c for c in chars if c["size"] >= body * 0.85), key=lambda c: (round(c["bottom"]), c["x0"]))
    small = [c for c in chars if c["size"] < body * 0.85 and c["text"] != " "]

    # 1. linie z tekstu w rozmiarze podstawowym (grupowanie po linii bazowej)
    lines: list[dict] = []
    for c in normal:
        for ln in lines:
            if abs(ln["bottom"] - c["bottom"]) <= body * 0.3:
                ln["chars"].append((c, False))
                ln["top"] = min(ln["top"], c["top"])
                ln["x0"], ln["x1"] = min(ln["x0"], c["x0"]), max(ln["x1"], c["x1"])
                break
        else:
            lines.append({"bottom": c["bottom"], "top": c["top"], "x0": c["x0"], "x1": c["x1"],
                          "chars": [(c, False)]})

    # 2. mały tekst: indeks górny w linii albo osobna linia (przypis)
    orphan: list[dict] = []
    for c in small:
        host = None
        for ln in lines:
            vertical = ln["top"] - body * 0.4 <= c["top"] and c["bottom"] <= ln["bottom"] + 0.5
            horizontal = ln["x0"] - body <= c["x0"] <= ln["x1"] + body * 2
            if vertical and horizontal:
                host = ln
                break
        if host is not None:
            raised = c["bottom"] < host["bottom"] - body * 0.15
            host["chars"].append((c, raised))
        else:
            for ln in orphan:
                if abs(ln["bottom"] - c["bottom"]) <= body * 0.3:
                    ln["chars"].append((c, False))
                    break
            else:
                orphan.append({"bottom": c["bottom"], "chars": [(c, False)]})

    if not drop_footnotes:
        lines.extend(orphan)
    lines.sort(key=lambda ln: ln["bottom"])
    return [_render_line(ln["chars"], body, drop_footnotes) for ln in lines]


def _render_line(items: list[tuple[dict, bool]], body: float, drop_footnotes: bool) -> str:
    items = sorted(items, key=lambda it: it[0]["x0"])
    out: list[str] = []
    prev_x1 = None
    i = 0
    while i < len(items):
        c, sup = items[i]
        if sup:
            run = []
            while i < len(items) and items[i][1]:
                run.append(items[i][0])
                i += 1
            run_text = "".join(r["text"] for r in run)
            before = out[-1][-1:] if out else ""
            is_marker = bool(re.fullmatch(r"(?:\d+\)|\*+)+", run_text))   # też serie: 2)3)
            if not is_marker and re.fullmatch(r"\d+", run_text) and (
                    before.isdigit() or (before.islower() and _is_unit_tail(out))):
                out.append("^" + run_text)                     # Art. 505^1, art. 50^5
            elif drop_footnotes and (is_marker or re.fullmatch(r"\d+", run_text)):
                continue                                       # odnośnik do przypisu; odstęp od tekstu przed nim
            else:
                out.append(run_text)
            prev_x1 = run[-1]["x1"]
            continue
        if c["text"] == " ":
            if out and out[-1] != " ":
                out.append(" ")
            i += 1
            continue
        gap = prev_x1 is not None and c["x0"] - prev_x1 > body * 0.2
        if gap and c["text"] not in ",.;:)" and out[-1] != " ":
            out.append(" ")
        out.append(c["text"])
        prev_x1 = c["x1"]
        i += 1
    return "".join(out).strip()


def _is_unit_tail(out: list[str]) -> bool:
    """Litera+indeks górny po numerze (np. 'art. 3a^1') - rzadkie, ale występuje."""
    tail = "".join(out)[-6:]
    return bool(re.search(r"\d[a-z]$", tail))


def read_legal_lines(path: Path, drop_footnotes: bool = True) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            body = _pdf_body_size(pdf)
            lines: list[str] = []
            for page in pdf.pages:
                lines.extend(_page_lines(page, body, drop_footnotes))
        return [normalize_superscripts(ln) for ln in lines]
    if ext in TEXT_EXT:
        return normalize_superscripts(path.read_text(encoding="utf-8")).splitlines()
    md = to_markdown(path)
    return normalize_superscripts(md).splitlines()


# ----------------------------------------------------------------- dokumenty ogólne

def to_markdown(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return path.read_text(encoding="utf-8")
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        if ext == ".pdf":
            log.warning("Brak docling - %s czytany przez pdfplumber, bez struktury nagłówków.", path.name)
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                return "\n\n".join((p.extract_text() or "") for p in pdf.pages)
        raise RuntimeError(f"Do konwersji {ext} potrzebny jest pakiet docling (pip install docling).")
    result = DocumentConverter().convert(str(path))
    return result.document.export_to_markdown()


def sample_text(path: Path, max_chars: int = 30000) -> str:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    if ext == ".pdf":
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages[:5])
        return text[:max_chars]
    return ""  # docx/html itp. - profil ogólny, chyba że wskazano inaczej
