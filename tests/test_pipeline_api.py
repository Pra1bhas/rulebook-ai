from fastapi.testclient import TestClient

from app.generation.answer import NOT_FOUND, answer_question
from app.pipeline import RuleBook


def test_ingest_is_idempotent_and_search_works(sample_pdf):
    rb = RuleBook(vector_location=":memory:")
    first = rb.ingest_file(sample_pdf)
    rb.ingest_file(sample_pdf)  # again: must not duplicate
    assert len(rb.chunks) == first["chunks"]
    assert rb.vectors.count() == first["chunks"]
    top = rb.search("personal data breach inform the Board")
    assert "breach" in top[0].chunk.text


def test_chunks_file_roundtrip_non_ascii(sample_pdf):
    # Bug: on Windows, open() defaults to cp1252 and crashed saving "≥" or Telugu text.
    rb = RuleBook(vector_location=":memory:")
    rb.ingest_file(sample_pdf)
    rb.chunks[0].text += " ≥ ≤ → తెలుగు"
    rb._save_chunks()
    assert rb._load_chunks()[0].text.endswith("≥ ≤ → తెలుగు")


def test_answer_has_valid_citations(sample_pdf):
    rb = RuleBook(vector_location=":memory:")
    rb.ingest_file(sample_pdf)
    ans = rb.ask("What happens on a data breach?")
    assert ans.found
    assert ans.citations and ans.citations[0].page >= 1


def test_no_results_means_not_found():
    ans = answer_question("anything", [])
    assert not ans.found and ans.answer == NOT_FOUND


def test_api_end_to_end(sample_pdf):
    from app.api.main import app

    me = {"X-Session-Id": "test-session-end-to-end-0001"}
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        with open(sample_pdf, "rb") as f:
            r = client.post("/upload", headers=me, files=[("files", ("sample_act.pdf", f, "application/pdf"))])
        assert r.status_code == 200 and r.json()[0]["ok"], r.text
        r = client.post("/ask", headers=me, json={"question": "children personal data consent"})
        assert r.status_code == 200
        assert r.json()["found"] is True
        mine = [d for d in client.get("/documents", headers=me).json() if d["mine"]]
        assert mine[0]["doc_name"] == "sample_act.pdf"
        bad = client.post("/upload", headers=me, files=[("files", ("x.txt", b"hi", "text/plain"))])
        assert bad.json()[0]["ok"] is False
        assert "RuleBook AI" in client.get("/").text
