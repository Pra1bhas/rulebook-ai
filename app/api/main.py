"""FastAPI server: exposes the pipeline over HTTP and serves the web UI.

Endpoints:
  GET    /                  -> chat web page
  GET    /health            -> is the server alive? how many chunks indexed?
  GET    /documents         -> public library + MY uploads
  POST   /ask               -> {"question": "..."} -> answer + citations
  POST   /solve             -> {"problem": "..."}  -> action plan + citations
  POST   /search            -> {"question": "..."} -> raw top chunks (for debugging)
  POST   /upload            -> upload 1..N PDFs, PRIVATE to my session
  DELETE /documents/{id}    -> delete one of MY uploads
  POST   /admin/ingest      -> add a PUBLIC document (needs X-Admin-Key)
  GET    /files/<name>.pdf  -> the public PDFs, so citations can link to the page

Who is "me"? The browser makes a random id once and sends it as the X-Session-Id
header. No login needed. (It's a private key, not a password: good enough for a
free public tool; real accounts would come later.)
Swagger docs (try every endpoint in the browser): /docs
"""
import logging
import re
import secrets
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import llm
from app.api.guard import RateLimiter, TTLCache, normalize
from app.config import settings
from app.models import ActionPlan, Answer, SearchResult
from app.pipeline import RuleBook

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
STATIC = Path(__file__).parent.parent / "static"
SESSION_RE = re.compile(r"^[A-Za-z0-9-]{16,64}$")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE at startup. Loading per request would add seconds to every question.
    rb = RuleBook()
    if settings.auto_ingest and Path(settings.data_dir).exists():
        added = rb.sync_folder(settings.data_dir)  # index any NEW default PDFs
        log.info("Public library: %d new document(s) indexed", len(added))
    rb.purge_expired()
    state.update(
        rb=rb,
        cache=TTLCache(settings.cache_size, settings.cache_ttl_hours * 3600),
        user_limit=RateLimiter(settings.rate_per_minute, settings.rate_per_day),
        # Per network address too, with more room (a family/office shares one address),
        # so making new session ids doesn't get around the limit.
        ip_limit=RateLimiter(settings.rate_per_minute * 3, settings.rate_per_day * 3),
        last_purge=time.time(),
    )
    yield
    state.clear()


app = FastAPI(title="RuleBook AI", lifespan=lifespan)
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=settings.data_dir, check_dir=False), name="files")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Problem(BaseModel):
    problem: str = Field(min_length=10, max_length=1500)


# ------------------------------------------------------------ helpers
def session_id(x_session_id: str | None = Header(default=None)) -> str:
    """'' = anonymous (public library only). Invalid ids are treated as anonymous."""
    return x_session_id if x_session_id and SESSION_RE.match(x_session_id) else ""


def client_ip(request: Request) -> str:
    # Behind Hugging Face / a proxy, the real address is the first X-Forwarded-For entry.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def rulebook() -> RuleBook:
    rb: RuleBook = state["rb"]
    if time.time() - state["last_purge"] > 600:  # at most every 10 minutes
        state["last_purge"] = time.time()
        rb.purge_expired()
    return rb


def check_rate(sid: str, ip: str) -> None:
    for limiter, key in ((state["user_limit"], sid or ip), (state["ip_limit"], ip)):
        wait = limiter.check(key)
        if wait:
            raise HTTPException(
                429, f"Too many questions. Please wait {wait} seconds (free service limits)."
            )


def cache_key(kind: str, text: str, rb: RuleBook, sid: str) -> tuple:
    # If I have private uploads, my answers depend on them: cache per user + doc set.
    mine = tuple(rb.user_doc_ids(sid))
    return (kind, normalize(text), sid if mine else "", mine)


# ------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    rb: RuleBook = state["rb"]
    return {"status": "ok", "chunks": len(rb.chunks), "llm": settings.llm_provider}


@app.get("/documents")
def documents(sid: str = Depends(session_id)):
    return rulebook().documents(sid)


# ------------------------------------------------------------ questions
@app.post("/ask", response_model=Answer)
def ask(q: Question, request: Request, sid: str = Depends(session_id)):
    rb = rulebook()
    key = cache_key("ask", q.question, rb, sid)
    if (hit := state["cache"].get(key)) is not None:
        return hit.model_copy(update={"question": q.question})
    check_rate(sid, client_ip(request))
    try:
        ans = rb.ask(q.question, sid)
    except llm.LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    state["cache"].set(key, ans)
    return ans


@app.post("/solve", response_model=ActionPlan)
def solve(p: Problem, request: Request, sid: str = Depends(session_id)):
    rb = rulebook()
    key = cache_key("solve", p.problem, rb, sid)
    if (hit := state["cache"].get(key)) is not None:
        return hit.model_copy(update={"problem": p.problem})
    check_rate(sid, client_ip(request))
    try:
        plan = rb.solve(p.problem, sid)
    except llm.LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    state["cache"].set(key, plan)
    return plan


@app.post("/search", response_model=list[SearchResult])
def search(q: Question, sid: str = Depends(session_id)):
    return rulebook().search(q.question, sid)


# ------------------------------------------------------------ documents
def _save_upload(file: UploadFile, folder: Path) -> Path:
    """Validate and save one upload. Raises HTTPException with a clear message."""
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, f"{name or 'file'}: only PDF files are supported")
    if file.size is not None and file.size > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"{name}: bigger than {settings.max_upload_mb} MB")
    if file.file.read(5) != b"%PDF-":  # check the content, not just the file name
        raise HTTPException(400, f"{name}: this is not a real PDF file")
    file.file.seek(0)
    dest = folder / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


@app.post("/upload")
def upload(files: list[UploadFile] = File(...), sid: str = Depends(session_id)):
    if not sid:
        raise HTTPException(400, "Missing X-Session-Id header (the web page sends it)")
    rb = rulebook()
    have = len(rb.user_doc_ids(sid))
    if have + len(files) > settings.max_user_docs:
        raise HTTPException(
            400, f"You can keep up to {settings.max_user_docs} PDFs. Delete one first."
        )
    results = []
    # The PDF itself is deleted right after indexing: we keep only the chunks, and those
    # are auto-deleted after user_doc_ttl_hours.
    with tempfile.TemporaryDirectory(dir=settings.storage_dir) as tmp:
        for file in files:
            try:
                path = _save_upload(file, Path(tmp))
                results.append({"ok": True, **rb.ingest_file(path, owner=sid)})
            except HTTPException as e:
                results.append({"ok": False, "doc": file.filename, "error": e.detail})
            except ValueError as e:
                results.append({"ok": False, "doc": file.filename, "error": str(e)})
    return results


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, sid: str = Depends(session_id)):
    if not rulebook().delete_document(doc_id, sid):
        raise HTTPException(404, "Not found, or not your document")
    return {"deleted": doc_id}


@app.post("/admin/ingest")
def admin_ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    if not settings.admin_key or not secrets.compare_digest(x_admin_key, settings.admin_key):
        raise HTTPException(403, "Admin only")
    dest = _save_upload(file, Path(settings.data_dir))
    try:
        return rulebook().ingest_file(dest)
    except ValueError as e:
        raise HTTPException(400, str(e))
