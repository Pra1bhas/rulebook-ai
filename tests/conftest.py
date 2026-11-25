"""Test setup: no internet, no API keys, no model downloads.

We switch to fake embedder / fake LLM / no reranker BEFORE the app is imported,
and keep all storage in a temporary folder.
"""
import os
import tempfile
from pathlib import Path

import pymupdf as fitz
import pytest

_tmp = tempfile.mkdtemp()
os.environ.update(
    EMBEDDER="fake",
    RERANKER="none",
    LLM_PROVIDER="fake",
    STORAGE_DIR=_tmp,
    QDRANT_PATH=f"{_tmp}/qdrant",
    QDRANT_URL="",
    AUTO_INGEST="false",
    DATA_DIR=f"{_tmp}/raw",
)

PAGES = [
    "CHAPTER I\nPRELIMINARY\n1. Short title. This Act may be called the Sample Data Act.",
    "CHAPTER II\nOBLIGATIONS\n8. (1) A Data Fiduciary shall protect personal data.\n"
    "(6) In the event of a personal data breach, the Data Fiduciary shall inform the Board "
    "and each affected Data Principal.",
    "9. Processing of children's data. The Data Fiduciary shall obtain verifiable consent "
    "of the parent before processing personal data of a child.",
    "CHAPTER III\nRIGHTS\n11. Right to access. A Data Principal may obtain a summary of "
    "personal data being processed.",
]


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    path = tmp_path / "sample_act.pdf"
    doc = fitz.open()
    for i, text in enumerate(PAGES, start=1):
        page = doc.new_page()
        page.insert_text((50, 40), "THE GAZETTE OF INDIA")  # repeated header
        page.insert_text((50, 80), text)
        page.insert_text((50, 800), f"Page {i}")
    doc.save(path)
    return path
