"""The full RAG pipeline in one class. Read this file first: it shows the whole flow.

INGEST (once per document):
    PDF -> parse_pdf -> pages -> chunk_pages -> chunks
        -> embedder -> vectors -> Qdrant
        -> chunks.jsonl (for BM25)
    Public library: PDFs in data/raw (owner "").
    User uploads: private, tagged with the user's session id (owner), auto-deleted after 24h.

ASK (a question):
    question -> rewrite to formal English
             -> BM25 + vector for BOTH the rewrite and the original (public + my docs only)
             -> RRF merge -> top 20 -> reranker -> top 5
             -> LLM with sources -> answer + citations (or "not found")

SOLVE (a problem: "UPI fraud lo 20k poyayi, em cheyyali?"):
    problem -> LLM splits it into 1-3 search queries
            -> search each query, keep the best few of each -> top 8
            -> LLM writes an action plan: rights, steps, where to complain, draft letter
            -> every point keeps only citations that really exist
"""
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

from app.config import settings
from app.generation.answer import answer_question, rewrite_query
from app.generation.solver import plan_queries, solve_problem
from app.ingest.chunker import chunk_pages
from app.ingest.parser import make_doc_id, parse_pdf
from app.models import ActionPlan, Answer, Chunk, SearchResult
from app.retrieval.bm25_index import BM25Index
from app.retrieval.embedder import get_embedder
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorStore

log = logging.getLogger(__name__)


def owner_prefix(owner: str) -> str:
    """Short, non-reversible tag so two users uploading the same PDF get separate copies."""
    return "u" + hashlib.sha1(owner.encode()).hexdigest()[:8]


class RuleBook:
    def __init__(self, vector_location: str | None = None):
        Path(settings.storage_dir).mkdir(parents=True, exist_ok=True)
        log.info("Loading embedding model (first run downloads it)...")
        self.embedder = get_embedder()
        self.vectors = VectorStore(self.embedder.dim, location=vector_location)
        log.info("Loading reranker...")
        self.reranker = get_reranker()
        self.lock = threading.Lock()  # FastAPI runs requests in threads: one writer at a time
        self.chunks = self._load_chunks()
        self.bm25 = BM25Index(self.chunks)
        self._ensure_vectors()

    def _ensure_vectors(self) -> None:
        """Self-heal: if Qdrant doesn't hold exactly one vector per chunk (new embedder,
        fresh in-memory store, crashed ingest), rebuild the vectors from chunks.jsonl."""
        if not self.chunks or self.vectors.count() == len(self.chunks):
            return
        log.info("Rebuilding %d vectors (embedder or store changed)", len(self.chunks))
        for owner in sorted({c.owner for c in self.chunks}):
            mine = [c for c in self.chunks if c.owner == owner]
            texts = [f"{c.section}\n{c.text}" for c in mine]
            self.vectors.upsert(mine, self.embedder.embed_documents(texts, cache=not owner))

    # ------------------------------------------------------------ ingest
    def ingest_file(self, path: str | Path, owner: str = "") -> dict:
        t0 = time.time()
        pages = parse_pdf(path)
        if not pages:
            raise ValueError(f"No text found in {Path(path).name}. Is it a scanned PDF? (needs OCR)")
        if owner and len(pages) > settings.max_upload_pages:
            raise ValueError(f"{Path(path).name} has {len(pages)} pages (max {settings.max_upload_pages})")
        if owner:
            prefix = owner_prefix(owner)
            pages = [p.model_copy(update={"doc_id": f"{prefix}-{p.doc_id}"}) for p in pages]
        now = time.time()
        chunks = [c.model_copy(update={"owner": owner, "created_at": now}) for c in chunk_pages(pages)]
        doc_id = chunks[0].doc_id
        texts = [f"{c.section}\n{c.text}" for c in chunks]
        vectors = self.embedder.embed_documents(texts, cache=not owner)  # never cache private docs

        with self.lock:
            # Remove the old version of this document (idempotent re-ingest).
            self.vectors.delete_doc(doc_id)
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id] + chunks
            self.vectors.upsert(chunks, vectors)
            self._save_chunks()
            self.bm25 = BM25Index(self.chunks)  # rebuild keyword index (fast)

        stats = {
            "doc_id": doc_id,
            "doc": chunks[0].doc_name,
            "pages": len(pages),
            "chunks": len(chunks),
            "seconds": round(time.time() - t0, 1),
        }
        log.info("Ingested %s", stats)
        return stats

    def sync_folder(self, folder: str | Path) -> list[dict]:
        """Index every public PDF in the folder that isn't indexed yet (new default docs)."""
        known = {c.doc_id for c in self.chunks if not c.owner}
        # Suffix check instead of glob("*.pdf") + glob("*.PDF"): on Windows those two
        # globs match the SAME files (case-insensitive), so every PDF was indexed twice.
        files = sorted(f for f in Path(folder).iterdir() if f.suffix.lower() == ".pdf")
        return [self.ingest_file(f) for f in files if make_doc_id(f) not in known]

    ingest_folder = sync_folder  # old name, still used by scripts/ingest_folder.py

    def delete_document(self, doc_id: str, owner: str) -> bool:
        """Users can delete only their own uploads, never the public library."""
        if not owner or not any(c.doc_id == doc_id and c.owner == owner for c in self.chunks):
            return False
        self._remove(lambda c: c.doc_id == doc_id)
        return True

    def purge_expired(self) -> int:
        """Delete user uploads older than user_doc_ttl_hours (privacy + free disk space)."""
        cutoff = time.time() - settings.user_doc_ttl_hours * 3600
        old = {c.doc_id for c in self.chunks if c.owner and c.created_at < cutoff}
        if old:
            self._remove(lambda c: c.doc_id in old)
            log.info("Purged %d expired user document(s)", len(old))
        return len(old)

    def _remove(self, match) -> None:
        with self.lock:
            for doc_id in {c.doc_id for c in self.chunks if match(c)}:
                self.vectors.delete_doc(doc_id)
            self.chunks = [c for c in self.chunks if not match(c)]
            self._save_chunks()
            self.bm25 = BM25Index(self.chunks)

    def documents(self, owner: str = "") -> list[dict]:
        """Public library + this user's own uploads (never other users' uploads)."""
        docs: dict[str, dict] = {}
        for c in self.chunks:
            if c.owner and c.owner != owner:
                continue
            d = docs.setdefault(
                c.doc_id,
                {
                    "doc_id": c.doc_id,
                    "doc_name": c.doc_name,
                    "chunks": 0,
                    "pages": set(),
                    "mine": bool(c.owner),
                    "expires_in_hours": None,
                },
            )
            d["chunks"] += 1
            d["pages"].add(c.page)
            if c.owner:
                left = c.created_at + settings.user_doc_ttl_hours * 3600 - time.time()
                d["expires_in_hours"] = round(max(left, 0) / 3600, 1)
        return [{**d, "pages": len(d["pages"])} for d in docs.values()]

    def user_doc_ids(self, owner: str) -> list[str]:
        return sorted({c.doc_id for c in self.chunks if owner and c.owner == owner})

    # ------------------------------------------------------------ search
    def search(
        self, query: str, owner: str = "", also: list[str] | None = None, k: int | None = None
    ) -> list[SearchResult]:
        """Hybrid search over the public library + this user's uploads.

        `also` = extra phrasings of the same question (e.g. the original Tenglish text).
        Their candidates join the RRF merge, so one bad rewrite can't hide the right chunk.
        The reranker scores with `query` (the clean English one).
        """
        owners = {"", owner}
        n = settings.candidates_k
        lists: list[list[SearchResult]] = []
        for q in dict.fromkeys([query, *(also or [])]):
            lists.append(self.bm25.search(q, n, owners))
            lists.append(self.vectors.search(self.embedder.embed_query(q), n, owners))
        merged = reciprocal_rank_fusion(lists, k=settings.rrf_k, limit=n)
        return self.reranker.rerank(query, merged, k or settings.final_k)

    def ask(self, question: str, owner: str = "") -> Answer:
        query = rewrite_query(question) if settings.rewrite_query else question
        results = self.search(query, owner, also=[question])
        return answer_question(question, results, search_query=query)

    def solve(self, problem: str, owner: str = "") -> ActionPlan:
        queries = plan_queries(problem)
        # Keep the best few chunks for EACH sub-problem, taking turns, so one topic
        # (e.g. "fraud") can't push out another (e.g. "bank not replying").
        per_query = [self.search(q, owner, k=settings.solve_per_query_k) for q in queries]
        results: list[SearchResult] = []
        seen: set[str] = set()
        for rank in range(settings.solve_per_query_k):
            for lst in per_query:
                if rank < len(lst) and lst[rank].chunk.chunk_id not in seen:
                    seen.add(lst[rank].chunk.chunk_id)
                    results.append(lst[rank])
        return solve_problem(problem, results[: settings.solve_k], queries)

    # ------------------------------------------------------------ storage
    def _load_chunks(self) -> list[Chunk]:
        f = settings.chunks_file
        if not f.exists():
            return []
        # Always UTF-8: Windows defaults to cp1252, which can't store symbols like "≥" or Telugu.
        text = f.read_text(encoding="utf-8")
        return [Chunk(**json.loads(line)) for line in text.splitlines() if line]

    def _save_chunks(self) -> None:
        with settings.chunks_file.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
