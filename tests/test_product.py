"""Tests for the 'real world' features: private uploads, problem solver, cache, rate limit."""
import time

from fastapi.testclient import TestClient

from app.api.guard import RateLimiter, TTLCache, normalize
from app.config import settings
from app.generation.answer import make_citations
from app.models import Chunk, SearchResult
from app.pipeline import RuleBook

ALICE, BOB = "alice-session-0000000001", "bob-session-00000000002"


def _upload(client, sid, pdf_path, name="sample_act.pdf"):
    with open(pdf_path, "rb") as f:
        return client.post(
            "/upload",
            headers={"X-Session-Id": sid},
            files=[("files", (name, f, "application/pdf"))],
        )


# ---------------- private uploads ----------------
def test_uploads_are_private_per_user(sample_pdf):
    rb = RuleBook(vector_location=":memory:")
    rb.chunks = []  # start from an empty library for this test
    rb.ingest_file(sample_pdf, owner=ALICE)
    q = "personal data breach inform the Board"
    assert rb.search(q, owner=ALICE), "Alice must find her own document"
    assert rb.search(q, owner=BOB) == [], "Bob must NOT see Alice's document"
    assert rb.search(q, owner="") == [], "anonymous users see only the public library"
    assert [d["mine"] for d in rb.documents(ALICE)] == [True]
    assert rb.documents(BOB) == []


def test_same_pdf_two_users_are_separate_copies(sample_pdf):
    rb = RuleBook(vector_location=":memory:")
    a = rb.ingest_file(sample_pdf, owner=ALICE)["doc_id"]
    b = rb.ingest_file(sample_pdf, owner=BOB)["doc_id"]
    assert a != b
    assert rb.delete_document(a, BOB) is False, "Bob can't delete Alice's copy"
    assert rb.delete_document(a, ALICE) is True
    assert rb.user_doc_ids(BOB) == [b]


def test_public_library_cannot_be_deleted_by_users(sample_pdf):
    rb = RuleBook(vector_location=":memory:")
    doc_id = rb.ingest_file(sample_pdf)["doc_id"]  # owner "" = public
    assert rb.delete_document(doc_id, ALICE) is False
    assert rb.delete_document(doc_id, "") is False


def test_expired_uploads_are_purged(sample_pdf, monkeypatch):
    rb = RuleBook(vector_location=":memory:")
    rb.chunks = []  # ignore uploads saved by other tests
    rb.ingest_file(sample_pdf, owner=ALICE)
    monkeypatch.setattr(time, "time", lambda: 10**12)  # far in the future
    assert rb.purge_expired() == 1
    assert rb.user_doc_ids(ALICE) == []


def test_upload_limits(sample_pdf, monkeypatch):
    from app.api.main import app

    monkeypatch.setattr(settings, "max_user_docs", 1)
    with TestClient(app) as client:
        sid = "limit-session-000000000001"
        assert _upload(client, sid, sample_pdf).json()[0]["ok"]
        assert _upload(client, sid, sample_pdf, "second.pdf").status_code == 400
        assert client.post("/upload", files=[("files", ("a.pdf", b"x", "application/pdf"))]).status_code == 400  # no session
        fake = client.post(
            "/upload",
            headers={"X-Session-Id": "other-session-00000000001"},
            files=[("files", ("evil.pdf", b"<html>not a pdf</html>", "application/pdf"))],
        )
        assert fake.json()[0] == {"ok": False, "doc": "evil.pdf", "error": "evil.pdf: this is not a real PDF file"}


def test_admin_ingest_needs_key(sample_pdf, monkeypatch):
    from app.api.main import app

    monkeypatch.setattr(settings, "admin_key", "secret-admin")
    with TestClient(app) as client, open(sample_pdf, "rb") as f:
        files = {"file": ("sample_act.pdf", f, "application/pdf")}
        assert client.post("/admin/ingest", files=files, headers={"X-Admin-Key": "wrong"}).status_code == 403


# ---------------- problem solver ----------------
def test_solve_returns_grounded_plan(sample_pdf):
    rb = RuleBook(vector_location=":memory:")
    rb.ingest_file(sample_pdf)
    plan = rb.solve("A company leaked my personal data in a breach. What should I do?")
    assert plan.found and plan.steps and plan.draft_letter
    texts = [p.text for p in plan.rights]
    assert "Fake point with an invented source." not in texts, "points with fake source ids are dropped"
    for point in plan.rights + plan.steps:
        assert all(1 <= r <= len(plan.citations) for r in point.refs)


def test_solve_api(sample_pdf):
    from app.api.main import app

    with TestClient(app) as client:
        _upload(client, ALICE, sample_pdf)
        r = client.post(
            "/solve",
            headers={"X-Session-Id": ALICE},
            json={"problem": "My child's data was processed without asking me. What can I do?"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["found"] is True


# ---------------- citations ----------------
def _result(cid, page, section):
    c = Chunk(chunk_id=cid, doc_id="d", doc_name="act.pdf", page=page, section=section,
              chunk_index=0, text="text")
    return SearchResult(chunk=c, score=1.0, source="rerank")


def test_citations_from_same_section_are_merged():
    cits, by_chunk = make_citations([_result("a", 5, "6. Consent"), _result("b", 5, "6. Consent"),
                                     _result("c", 8, "9. Children")])
    assert len(cits) == 2
    assert by_chunk == {"a": 1, "b": 1, "c": 2}
    assert cits[0].url == "files/act.pdf#page=5"


# ---------------- protecting the free tier ----------------
def test_cache_returns_saved_answer_and_expires(monkeypatch):
    cache = TTLCache(max_items=2, ttl_seconds=60)
    cache.set("q", "answer")
    assert cache.get("q") == "answer"
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now + 61)
    assert cache.get("q") is None


def test_cache_drops_least_recently_used():
    cache = TTLCache(max_items=2, ttl_seconds=60)
    cache.set("a", 1); cache.set("b", 2)
    cache.get("a")  # 'a' used recently
    cache.set("c", 3)  # -> 'b' is dropped
    assert cache.get("b") is None and cache.get("a") == 1


def test_normalize_makes_same_question_same_key():
    assert normalize("Can I withdraw  my consent?") == normalize("can i withdraw my consent")


def test_rate_limiter_per_minute():
    rl = RateLimiter(per_minute=2, per_day=100)
    assert rl.check("me") == 0 and rl.check("me") == 0
    assert rl.check("me") > 0, "3rd question in a minute must wait"
    assert rl.check("someone-else") == 0, "limits are per user"


def test_ask_api_rate_limited_but_cache_is_free(sample_pdf, monkeypatch):
    from app.api.main import app

    monkeypatch.setattr(settings, "rate_per_minute", 1)
    with TestClient(app) as client:
        sid = {"X-Session-Id": "rate-session-0000000001"}
        assert client.post("/ask", headers=sid, json={"question": "first question here"}).status_code == 200
        assert client.post("/ask", headers=sid, json={"question": "First question here!"}).status_code == 200  # cached
        assert client.post("/ask", headers=sid, json={"question": "a different question"}).status_code == 429
