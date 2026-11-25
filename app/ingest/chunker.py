"""Step 2 of ingestion: Pages -> Chunks.

Why chunk? A 100-page PDF is too big to send to the LLM for every question.
We cut it into small pieces and later send only the few relevant pieces.

Strategy (section-aware):
  1. Legal documents are organised as "CHAPTER II", "8. Obligations ...", "Section 3".
     We detect these headings so each chunk knows which section it belongs to.
  2. Inside a section, we cut into windows of ~chunk_size words with some overlap,
     so a sentence on the border is not lost.
We measure size in words (simple, no extra downloads). Rough rule: 1 word ~ 1.3 tokens.
"""
import re

from app.config import settings
from app.models import Chunk, Page

# A line that starts a new section, e.g.
#   "CHAPTER II"   "Chapter 3"   "8. (1) A Data Fiduciary ..."   "Section 12"   "Annex I"
HEADING = re.compile(
    r"^\s*("
    r"(CHAPTER|Chapter)\s+[IVXLC\d]+"
    r"|(SECTION|Section)\s+\d+"
    r"|(THE\s+)?(ANNEX|Annex|SCHEDULE|Schedule|APPENDIX|Appendix)(\s+[IVXLC\d]+)?\s*$"
    r"|\d{1,3}\.\s+(\(\d+\)\s*)?[A-Z][^\n]{0,80}"
    r")"
)


def detect_heading(line: str) -> str | None:
    m = HEADING.match(line)
    if not m:
        return None
    return line.strip()[:80]


def chunk_pages(
    pages: list[Page],
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or settings.chunk_size_words
    overlap = overlap if overlap is not None else settings.chunk_overlap_words
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    # 1) Walk through all lines, tagging each with (page, section).
    current_section = "Introduction"
    tagged: list[tuple[int, str, str]] = []  # (page, section, line)
    for page in pages:
        for line in page.text.splitlines():
            heading = detect_heading(line)
            if heading:
                current_section = heading
            if line.strip():
                tagged.append((page.page, current_section, line.strip()))

    # 2) Group consecutive lines of the same section.
    groups: list[list[tuple[int, str, str]]] = []
    for item in tagged:
        if groups and groups[-1][-1][1] == item[1]:
            groups[-1].append(item)
        else:
            groups.append([item])

    # 3) Cut each section into word windows with overlap.
    chunks: list[Chunk] = []
    if not pages:
        return chunks
    doc_id, doc_name = pages[0].doc_id, pages[0].doc_name
    step = size - overlap
    for group in groups:
        words: list[tuple[str, int]] = []  # (word, page it came from)
        for page_no, _, line in group:
            words.extend((w, page_no) for w in line.split())
        section = group[0][1]
        for start in range(0, len(words), step):
            window = words[start : start + size]
            # Tiny tail already covered by the previous window's overlap -> skip.
            if start > 0 and len(window) <= overlap:
                break
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=window[0][1],
                    section=section,
                    chunk_index=idx,
                    text=" ".join(w for w, _ in window),
                )
            )
    return chunks
