# 🧠 LEARN.md: Ee project ni ela nerchukovali

Code motham ready ga undi. Nee pani: **chadavadam, run cheyyadam, maarchi chudadam, explain cheyyagaladam.**
Interview lo "ee project explain cheyyi" ani adigithe 5 min matladagaligithe chaalu, neeku job ready.

Ee file lo 6 rounds unnayi. Roju 2-3 rounds chaalu. Prathi round 30-45 min.

---

## Round 0: Run it first (30 min)

Mundu app ni run chesi chudu. Pani chese app ni chusthe code chadavadaniki interest vasthundi.

```bash
cp .env.example .env
```
`.env` lo `GEMINI_API_KEY` pettu. `data/raw/` lo DPDP Act PDF pettu (meity.gov.in). Tarvatha:
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
(Windows lo `.venv\Scripts\activate`)
```bash
pip install -r requirements-dev.txt
```
```bash
uvicorn app.api.main:app --reload --port 7860
```
Browser lo `http://localhost:7860` open chesi 5 questions adugu. Tarvatha `http://localhost:7860/docs` lo `/search` try cheyyi. Adi LLM ki pampe chunks ni chupisthundi.

✅ **Check:** "not found" answer vachela oka question adugu (ex: "What is the repo rate?").

---

## Round 1: Big picture (30 min)

**`app/pipeline.py` okkate chadavu.** Top lo unna comment lo motham flow undi:

```
INGEST (PDF vachinappudu okkasari):
PDF → pages → chunks → vectors → Qdrant
                    └→ BM25 index

ASK (prathi question ki):
question → rewrite → BM25 top 20 + vector top 20 → RRF merge → rerank → top 5 → LLM → answer + citations
```

**Real-life example:** Library lo oka book vethakadam anukondi.
- **Chunking** = book ni chapters/pages ga vibhajinchadam
- **BM25** = index lo "KYC" ane word ekkada undo chudadam
- **Embeddings** = librarian ki meaning ardham avthundi: "data share" ani adigithe "disclosure" page chupisthadu
- **RRF** = iddaru librarians suggestions ni kalapadam
- **Reranker** = senior librarian top 20 pages chadivi best 5 select cheyyadam
- **LLM** = aa 5 pages chadivi neeku simple ga cheppe friend

✅ **Check:** Ee flow ni paper meeda code chudakunda draw cheyyi. Cheyyagaligithe next round ki vellu.

---

## Round 2: Ingest side (45 min)

Ee order lo chadavu:

**1. `app/ingest/parser.py`**
- `make_doc_id`: same file ni malli upload chesthe duplicate raadu. Enduku ante file content hash vaaduthunnam.
- `_find_repeated_lines`: prathi page lo vache "GAZETTE OF INDIA" laanti headers ni remove chesthundi. Leni ante prathi chunk lo garbage untundi.

**2. `app/ingest/chunker.py`** ⭐ interview favourite
- `HEADING` regex: "CHAPTER II", "8. (1) A Data Fiduciary..." laanti lines ni section start ga gurthisthundi.
- Prathi chunk ki `section` + `page` untayi. Citations ki ivi kavali.
- **Overlap:** 220 words chunk, 40 words overlap. Oka sentence rendu chunks border lo padithe, overlap valla rendintilo vasthundi.

🧪 **Experiment:** `.env` lo `CHUNK_SIZE_WORDS=80` pettu, `storage/` folder delete chesi restart cheyyi. Answers ela maaripoyayo chudu. Tarvatha `600` tho try cheyyi.
> Chinna chunks: precise, kani context miss avthundi. Pedda chunks: context untundi, kani noise kooda ekkuva, LLM ki ekkuva tokens.

**Interview questions:**
- Chunk size ela decide chesav? → *"Legal sections are 100-400 words. I started at 220 words with 40 overlap and tuned it using my eval script's hit rate."*
- Scanned PDF vasthe? → *"PyMuPDF returns no text, so we raise an error. The fix is OCR (Tesseract), which I've used before in my TekVision internship."*

---

## Round 3: Search side (45 min) ⭐⭐

**1. `app/retrieval/bm25_index.py`**: keyword search. `tokenize` lo "8(6)" ni okka token ga unchutham.

**2. `app/retrieval/embedder.py`**: text → 384 numbers. Same meaning unna sentences ki vectors daggaraga untayi.
- `embed_query` vs `embed_documents`: bge models questions ki prefix add chesthayi.

**3. `app/retrieval/vector_store.py`**: Qdrant.
- `QDRANT_URL` set chesthe server mode (docker-compose), lekapothe folder lo embedded mode (deploy ki easy).
- `uuid5(chunk_id)`: same chunk ki same id vasthundi, so duplicates raavu.

**4. `app/retrieval/hybrid.py`**: RRF. File top lo hand-worked example undi, adi pen tho chesi chudu.

**5. `app/retrieval/reranker.py`**: Bi-encoder vs cross-encoder difference file top comment lo undi.

🧪 **Experiments** (ivi resume numbers kooda isthayi):
```bash
python -m scripts.evaluate
```
`RERANKER=none python -m scripts.evaluate` tho run chesi compare cheyyi. Hit rate, MRR numbers note chesko.

(Windows lo: `set RERANKER=none` run chesi tarvatha `python -m scripts.evaluate`)

**Interview questions:**
- Hybrid search enduku? → *"BM25 catches exact terms like section numbers; embeddings catch paraphrases. Each misses what the other finds."*
- Scores add cheyyakunda RRF enduku? → *"BM25 scores are unbounded (like 12.7), cosine is between -1 and 1. RRF uses only ranks, so no normalisation is needed."*
- Reranker motham database meeda enduku run cheyyamu? → *"A cross-encoder runs a full model forward pass per (query, chunk) pair. That's too slow for thousands of chunks, so we only rerank the top 20."*
- HNSW enti? → *"Qdrant's graph index. It finds approximate nearest neighbours without comparing against every vector."*

---

## Round 4: Answer side (45 min) ⭐⭐

**1. `app/llm.py`**
- Okka wrapper: Gemini ↔ Groq switch cheyyadaniki `.env` lo okka line maarchithe chaalu.
- 429 vasthe 2s, 4s, 8s, 16s wait chesi retry chesthundi (exponential backoff).
- `schema=LLMAnswer`: Gemini ni JSON format lo reply cheyyamani force chestham (structured output).

**2. `app/generation/answer.py`**: 3 hallucination guards (file top comment chudu):
1. Prompt lo "sources nunchi matrame answer cheyyi" ani rule
2. `found` true/false flag
3. LLM ichina `S3` laanti source ids nijamga mana sources lo unnayo check chestham. Fake ids vasthe drop chestham.

- `rewrite_query`: "Bank naa data share cheyyocha?" ni "Can a bank disclose customer personal data?" ga maarchi search chesthundi. UI lo "Searched for:" ani chupisthundi.

🧪 **Experiment:** `.env` lo `REWRITE_QUERY=false` pettu, Telugu question adugu. Rewrite lekapothe search results enta worse avthayo chudu.

**Interview questions:**
- Hallucination ela control chesav? → 3 guards cheppu.
- Structured output enduku? → *"So the code can trust the shape: found flag and source ids. No fragile text parsing."*
- Free tier rate limit ela handle chesav? → backoff + Groq fallback + one LLM call for the answer (plus one for rewrite).

---

## Round 5: API, UI, Docker, Deploy (45 min)

**1. `app/api/main.py`**
- `lifespan`: models ni startup lo **okkasari** load chestham. Prathi request lo load chesthe prathi question ki seconds waste avthayi.
- Index empty ga unte `data/raw` ni auto-ingest chesthundi. Deploy lo idi useful.

**2. `app/static/index.html`**: Simple HTML + JS chat. `esc()` function LLM output ni HTML ga run avvakunda aaputhundi (XSS protection).

**3. `Dockerfile`**: comments chadavu. Important points:
- Libraries mundu copy chestham, code tarvatha. Code maarinappudu Docker library layer ni cache nunchi vaaduthundi, so rebuild fast.
- Models ni build time lo download chestham, so app fast ga start avthundi.

**4. `docker-compose.yml`**: app + Qdrant server rendu separate containers. Company setup laaga.
```bash
docker compose up --build
```
`http://localhost:6333/dashboard` lo nee chunks ni chudu.

**5. `DEPLOY.md`** follow chesi Hugging Face lo deploy cheyyi. 🎉

**Interview questions:**
- Docker image chinnaga ela chesthav? → slim base image, `--no-cache-dir`, `.dockerignore`, layer order.
- Deploy lo API key ela safe? → platform secrets, never in code/image; `.env` is in `.gitignore`.

---

## Round 6: Final boss 🏆

Claude Code open chesi ila adugu:
```
Act as a senior AI engineer interviewing me about this project. Ask 10 questions one at a time, from easy to hard. Wait for my answer, then tell me what was good and what was missing.
```
8/10 correct ga cheppagaligithe nuvvu ready.

---

## Resume bullets (nee real numbers tho fill cheyyi)

> **RuleBook AI: RAG assistant for Indian regulations (RBI, DPDP Act)** | Python, FastAPI, Qdrant, Gemini, Docker · [Live demo] · [GitHub]
> - Built hybrid retrieval (BM25 + bge embeddings, fused with Reciprocal Rank Fusion) and cross-encoder reranking; section-aware chunking preserves document, page and section for every citation.
> - Reduced hallucinations with source-only prompting, structured JSON outputs and citation validation; the system answers "not found" for out-of-scope questions (X/Y in tests).
> - Built a retrieval eval (20 questions, 5 regulations): measured that a general cross-encoder reranker *hurt* legal retrieval (hit@5 90%→75%), and shipped Gemini embeddings without it: 95% hit@5, MRR 0.70.
> - Supports Telugu-English questions via LLM query rewriting; provider-agnostic LLM layer with retry/backoff (Gemini, Groq).
> - Fit a 512 MB free tier (measured 960 MB → 400 MB) with API embeddings + a committed vector cache; Docker, 39 pytest tests, GitHub Actions CI, deployed on Render.

⚠️ **Numbers nuvvu run chesi chusinave pettu.** Interview lo "how did you measure?" ani adigithe `scripts/evaluate.py` chupinchu.

---

## Claude Code tho explore cheyyadaniki useful prompts
- `Trace what happens step by step when I ask "Can I withdraw consent?" — show me which function runs and what data it returns at each step.`
- `Explain app/retrieval/hybrid.py line by line in simple Telugu-English.`
- `Add a print in pipeline.search so I can see BM25, vector, and reranked results side by side. Run it.`
- `I changed X and the results got worse. Why?`
