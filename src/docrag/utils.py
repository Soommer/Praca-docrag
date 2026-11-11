"""Drobne narzędzia wspólne dla całego pipeline'u."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

_SUP_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUP_TRANS = str.maketrans(_SUP_DIGITS, "0123456789")
_SUP_RE = re.compile(rf"([0-9a-z])([{_SUP_DIGITS}]+)")
_BRACKET_SUP_RE = re.compile(r"(\d[a-z]{0,2})\[(\d+[a-z]{0,2})\](?=([\w§]))?")


def normalize_superscripts(text: str) -> str:
    """'Art. 505¹' i 'Art. 505[1]' (nowsze PDF-y ISAP) -> 'Art. 505^1' (jednolity zapis indeksów górnych)."""
    # w PDF-ach ISAP po "]" często brakuje spacji: "art. 63[1]do", "art. 117[3]§ 2"
    text = _BRACKET_SUP_RE.sub(lambda m: f"{m.group(1)}^{m.group(2)}" + (" " if m.group(3) else ""), text)
    return _SUP_RE.sub(lambda m: m.group(1) + "^" + m.group(2).translate(_SUP_TRANS), text)


def slugify(value: str) -> str:
    value = value.replace("ł", "l").replace("Ł", "L")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or "doc"


def sha256_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Skróty, po których kropka NIE kończy zdania (polszczyzna prawnicza i urzędowa).
_ABBR = {
    "art", "ust", "pkt", "lit", "par", "poz", "nr", "np", "tj", "tzw", "m.in", "r",
    "dz", "u", "zm", "ww", "ok", "godz", "str", "s", "al", "ul", "prof", "dr", "inż",
    "mgr", "tel", "wg", "jw", "tzn", "zd", "kpc", "k.p.c", "k.c", "k.k", "sygn", "akt",
}
_SENT_END = re.compile(r"(?<=[.!?…;])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9„\"(§–-])")


def split_sentences(text: str) -> list[str]:
    parts = _SENT_END.split(text.strip())
    out: list[str] = []
    for p in parts:
        if out:
            last = out[-1].rsplit(None, 1)[-1].rstrip(".").lower()
            if last in _ABBR or re.fullmatch(r"\d+", last):
                out[-1] += " " + p
                continue
        out.append(p)
    return [s for s in out if s.strip()]
