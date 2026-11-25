"""Step 1 of ingestion: file -> list of Pages (clean text + page number).

We keep the page number so every answer can say "see page 12".
"""
import hashlib
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from app.models import Page


def make_doc_id(path: Path) -> str:
    """Same file content -> same id. Re-ingesting a file won't create duplicates."""
    digest = hashlib.sha1(path.read_bytes()).hexdigest()[:10]
    return f"{path.stem[:40]}-{digest}"


def parse_pdf(path: str | Path) -> list[Page]:
    path = Path(path)
    doc_id = make_doc_id(path)
    with fitz.open(path) as pdf:
        raw_pages = [p.get_text("text") for p in pdf]

    repeated = _find_repeated_lines(raw_pages)
    pages = []
    for i, text in enumerate(raw_pages, start=1):
        cleaned = _clean(text, repeated)
        # skip empty / image-only pages, and Hindi pages of bilingual gazettes
        if cleaned.strip() and not is_mostly_devanagari(cleaned):
            pages.append(Page(doc_id=doc_id, doc_name=path.name, page=i, text=cleaned))
    return pages


def is_mostly_devanagari(text: str) -> bool:
    """Gazettes print the Hindi version next to the English one. Our embedding and
    reranker models are English-only, so Hindi pages are just noise in search.
    """
    letters = [c for c in text if c.isalpha()]
    hindi = sum("ऀ" <= c <= "ॿ" for c in letters)
    return bool(letters) and hindi / len(letters) > 0.5


def _find_repeated_lines(pages: list[str]) -> set[str]:
    """Headers/footers appear on most pages (e.g. 'RBI/2025-26/12'). Find and drop them."""
    if len(pages) < 4:
        return set()
    counts = Counter()
    for text in pages:
        lines = {ln.strip() for ln in text.splitlines() if ln.strip()}
        counts.update(lines)
    threshold = len(pages) * 0.6
    # Short lines only: a long repeated line is probably real content.
    return {ln for ln, c in counts.items() if c >= threshold and len(ln) < 80}


def _clean(text: str, repeated: set[str]) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() not in repeated]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # join hyphen-broken words: "regu-\nlation"
    text = re.sub(r"^\s*Page \d+( of \d+)?\s*$", "", text, flags=re.M | re.I)  # page numbers
    text = re.sub(r"[ \t]+", " ", text)  # many spaces -> one
    text = re.sub(r"\n{3,}", "\n\n", text)  # many blank lines -> one
    return text.strip()
