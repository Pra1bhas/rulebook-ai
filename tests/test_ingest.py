from app.ingest.chunker import chunk_pages, detect_heading
from app.ingest.parser import parse_pdf
from app.models import Page


def test_parser_reads_pages_and_drops_header(sample_pdf):
    pages = parse_pdf(sample_pdf)
    assert len(pages) == 4
    assert pages[1].page == 2
    assert "personal data breach" in pages[1].text
    assert all("GAZETTE" not in p.text for p in pages)  # repeated header removed
    assert all("Page " not in p.text for p in pages)  # page numbers removed


def test_same_file_gives_same_doc_id(sample_pdf):
    assert parse_pdf(sample_pdf)[0].doc_id == parse_pdf(sample_pdf)[0].doc_id


def test_detect_heading():
    assert detect_heading("CHAPTER II") == "CHAPTER II"
    assert detect_heading("8. (1) A Data Fiduciary shall") is not None
    assert detect_heading("the data fiduciary shall") is None
    assert detect_heading("(a) in clause 3") is None


def test_chunks_keep_section_and_page(sample_pdf):
    chunks = chunk_pages(parse_pdf(sample_pdf))
    breach = next(c for c in chunks if "breach" in c.text)
    assert breach.section.startswith("8.")
    assert breach.page == 2
    assert chunks[0].chunk_index == 0


def test_chunk_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    pages = [Page(doc_id="d", doc_name="d.pdf", page=1, text=text)]
    chunks = chunk_pages(pages, size=40, overlap=10)
    assert all(len(c.text.split()) <= 40 for c in chunks)
    # overlap: last 10 words of chunk 0 == first 10 words of chunk 1
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    # every word is covered
    covered = set(w for c in chunks for w in c.text.split())
    assert len(covered) == 100


def test_schedule_heading_only_when_standalone():
    assert detect_heading("THE SCHEDULE") is not None
    assert detect_heading("Annex II") is not None
    assert detect_heading("Schedule to the Constitution and providing the contact") is None


def test_hindi_pages_of_bilingual_gazette_are_skipped():
    from app.ingest.parser import is_mostly_devanagari

    assert is_mostly_devanagari("डिजिटल व्यक्तिगत डेटा संरक्षण नियम, 2025")
    assert not is_mostly_devanagari("Digital Personal Data Protection Rules, 2025 (असाधारण)")
