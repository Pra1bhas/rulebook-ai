# 🚀 Deploy (free) — Hugging Face Spaces

Hugging Face Spaces free CPU machine lo 16 GB RAM untundi. Mana embedding + reranker models ki adi saripothundi.
Mana `Dockerfile` ne vaaduthundi, so code maarchalsina avasaram ledu.

## Step 0: PDFs ready cheyyi
`data/raw/` lo public PDFs pettu (ex: DPDP Act 2023, RBI KYC Master Direction).
App start ayyinappudu index empty ga unte, ee PDFs ni automatic ga index chesthundi.

## Step 1: Space create cheyyi (5 min)
1. huggingface.co lo account create cheyyi (free)
2. Top right lo **New → Space**
3. Name: `rulebook-ai` · SDK: **Docker** · Template: **Blank** · Hardware: **CPU basic (free)** · Public
4. **Create Space**

## Step 2: API key ni secret ga pettu (important!)
Space lo **Settings → Variables and secrets → New secret**:
- Name: `GEMINI_API_KEY` · Value: nee key

Key ni code lo kaani, `.env` file lo kaani upload cheyyaku. Secret ga pettina value container ki environment variable ga vasthundi, kani bayata evariki kanipinchadu.

(Optional) **New variable**: `GEMINI_MODEL` = AI Studio lo ippudu free ga unna model name.

## Step 3: Files upload cheyyi
Easiest way (git avasaram ledu): Space page lo **Files → Add file → Upload files**. Ivi drag & drop cheyyi:
```
Dockerfile   README.md   requirements.txt   .dockerignore
app/         scripts/    data/
```
❌ Ivi upload cheyyaku: `.env`, `storage/`, `models/`, `.venv/`

`README.md` top lo unna `sdk: docker` / `app_port: 7860` lines ni Hugging Face chaduvuthundi. Vaatini delete cheyyaku.

## Step 4: Build chudu
**Logs** tab open cheyyi. First build ki ~5-10 min padutundi (libraries + models download).
"Application startup complete" kanipisthe ready. Nee app link idi:
```
https://huggingface.co/spaces/<your-username>/rulebook-ai
```

## Problems?
| Problem | Fix |
|---|---|
| `GEMINI_API_KEY is missing` | Step 2 secret name exact ga check cheyyi, tarvatha Space ni **Restart** cheyyi |
| `Gemini error 404` model not found | `GEMINI_MODEL` variable lo AI Studio lo unna current model name pettu |
| `Rate limit hit` | Free tier limit. 1 min wait cheyyi, leda `GROQ_API_KEY` secret + `LLM_PROVIDER=groq` variable pettu |
| Uploaded PDF restart tarvatha kanipinchadu | Free Spaces lo storage temporary. Permanent ga undalante PDF ni `data/raw/` lo pettu |
| Space "sleeping" | 48 hrs evaru open cheyyakapothe free Spaces sleep avthayi. Link open chesthe malli start avthundi (~1 min) |

## Code update chesthe
Changed files ni malli upload cheyyi. Space automatic ga rebuild avthundi.

---

## GitHub ki kooda push cheyyi (resume kosam must)
```bash
git init
```
```bash
git add .
```
```bash
git status
```
(`.env` list lo ledu ani confirm cheyyi!)
```bash
git commit -m "RuleBook AI: hybrid RAG over Indian regulations"
```
GitHub lo `rulebook-ai` ane empty repo create chesi, akkada chupinche `git remote add origin ...` and `git push -u origin main` commands run cheyyi.
README lo top lo live demo link pettu.
