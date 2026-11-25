# 🚀 Deploy for ₹0: Render (free, no credit card)

## Why Render, and why "lite" mode? (interview story ⭐)
- **Hugging Face Docker Spaces stopped being free in July 2026** (402: "requires a PRO subscription").
  Google Cloud Run / Oracle need a card. Render's free plan needs **no card**: 512 MB RAM, 0.1 CPU.
- I **measured** the full app: ~530 MB with no reranker and ~960 MB with even a small reranker, because
  the local models + `onnxruntime` need that much RAM *while running* (model file size ≠ memory used).
- So production runs in **lite mode** (`Dockerfile.render`):
  - embeddings from the **Gemini API** (`EMBEDDER=gemini`) → no local model, no onnxruntime
  - **no reranker** (`RERANKER=none`); the full pipeline still runs locally (`Dockerfile`, `uvicorn`)
  - the public library's vectors are **cached in `data/embeddings/*.npz`** (committed), so the
    index is built during `docker build` with **0 API calls and no key**, and startup is fast

Kitchen example 🍳: the full restaurant has a big oven (the reranker). The free food truck has no
room for it, so we serve a simpler menu, and measured how much taste we lose (`scripts/evaluate.py`).

### Measured: what does lite mode cost? (20 questions, 5 documents, Nov 2026)
| Config | Hit@5 | MRR |
|---|---|---|
| Local bge-small + bge-reranker-base | 75% | 0.45 |
| ...same, reranker also sees the section heading (fixed) | 80% | 0.55 |
| Local bge-small, **no reranker** | 90% | 0.59 |
| Gemini embeddings + reranker (with heading) | 80% | 0.51 |
| **Lite = production: Gemini embeddings, no reranker** | **95%** | **0.70** |

Surprise: on this eval set **lite mode is the best**, so the free deploy loses nothing. The general-purpose
cross-encoder (trained on web search, MS MARCO) re-orders legal sections worse than the embeddings do.
Honest caveats: only 20 questions, and a hit counts only the expected section in the expected PDF
(an equally good answer from the DPDP *Rules* counts as a miss). Next step: a bigger eval set.

## Step 1: Render account (2 min)
1. https://render.com → **Get Started** → **Sign in with GitHub** (no card needed)
2. Allow Render to see the `rulebook-ai` repository

## Step 2: Create the service from the Blueprint (`render.yaml`)
1. Render dashboard → **New +** → **Blueprint**
2. Pick the `rulebook-ai` repo → Render reads `render.yaml` (free plan, `Dockerfile.render`)
3. It asks for the secret values: paste `GEMINI_API_KEY`, `GROQ_API_KEY` (backup LLM),
   and optionally `ADMIN_KEY` (any long random password). They are stored as secrets, never in git.
4. **Apply**. The first build takes ~5 min. The link looks like `https://rulebook-ai-xxxx.onrender.com`

## Step 3: Check it
- `https://<your-app>.onrender.com/health` → `{"status":"ok","chunks":494,...}`
- Ask "Can I withdraw my consent?" and try 🛠️ Solve my problem.

## Free plan limits (be honest in interviews)
| Limit | What it means | What we did |
|---|---|---|
| Sleeps after 15 min idle | First visit after a break waits ~1 min | Index pre-built in the image, so wake-up is just loading |
| 512 MB RAM | Big local models don't fit | Lite mode (API embeddings, no reranker) |
| Disk is temporary | Uploads vanish on restart/redeploy | That's fine: uploads auto-delete after 24 h anyway |
| Gemini free tier: 100 embeddings/min | Heavy traffic can hit limits | Query-embedding cache + answer cache + per-user rate limit |

## Adding a new public PDF
1. Put it in `data/raw/`
2. Run the app **locally once with** `EMBEDDER=gemini` (in `.env`), so `data/embeddings/*.npz` gets its vectors
   (the free tier allows 100 per minute, so a big PDF can take a few minutes; it waits automatically)
3. Commit + push → Render redeploys automatically. If you forget step 2, the build fails with a clear message.

## Full version (laptop / paid hosting)
`docker compose up --build` or `uvicorn app.api.main:app --port 7860` runs the full pipeline:
local bge-small embeddings + bge-reranker-base cross-encoder.
