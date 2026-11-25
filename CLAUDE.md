# RuleBook AI

RAG assistant for Indian regulations (RBI, DPDP Act, SEBI). Answers in English or Telugu-English
with citations (document, page, section) and says "not found" instead of guessing.

## About me
I'm Prabhas, an AI engineer preparing for job interviews. The code is already built.
I'm learning it by reading, running and changing it. My concentration is low, so:
- Explain in simple English mixed with Telugu (Tenglish), with real-life examples.
- Keep explanations short (5-6 lines), then show real output by running code.
- When I ask "explain X", explain the concept first, then the code, line by line if I ask.
- When I make a change, run the tests and the eval script and tell me what changed and WHY.
- If I say `confused`, re-explain with a simpler example.
- Never silently rewrite big parts of the project. Small changes, explained.

LEARN.md has my learning rounds. Follow it if I say "start round N".

## Architecture
See the docstring at the top of `app/pipeline.py` for the full flow.
- `app/config.py`: all settings (env / .env)
- `app/llm.py`: the only place that calls an LLM (gemini | groq | fake)
- `app/ingest/`: parser (PyMuPDF), section-aware chunker
- `app/retrieval/`: BM25, fastembed embeddings, Qdrant (server if QDRANT_URL else embedded), RRF, cross-encoder reranker
- `app/generation/answer.py`: query rewrite, prompt, citation validation + dedupe
- `app/generation/solver.py`: "Solve my problem" mode: problem → 1-3 queries → cited action plan + draft letter
- `app/api/main.py`: FastAPI + serves `app/static/index.html`
- `app/api/guard.py`: answer cache + per-user rate limit (protects the free LLM quota)
- Public library = PDFs in `data/raw` (owner ""), indexed automatically at startup.
  User uploads = private per browser (`X-Session-Id` header, `Chunk.owner`), auto-deleted after 24 h.
  Adding public docs on a deployed server: `POST /admin/ingest` with `X-Admin-Key` (= `ADMIN_KEY`).

## Commands
- Run: `uvicorn app.api.main:app --reload --port 7860`
- Tests (no internet/key needed): `pytest -q`
- Eval: `python -m scripts.evaluate` (add `--llm` for not-found checks)
- Ingest a folder: `python -m scripts.ingest_folder data/raw`
- Docker: `docker compose up --build` → http://localhost:8000
- After changing chunking settings: delete `storage/` so the index is rebuilt.

## Rules
- Free tier only: Gemini (`GEMINI_API_KEY`), Groq backup. Keep LLM calls minimal.
- Only public documents. Never commit `.env`.
- Keep tests passing; add a test for every bug fix.
