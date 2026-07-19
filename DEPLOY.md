# Deploying Mohafiz to Hugging Face Spaces (free)

The app is one FastAPI container (Gradio UI at `/`, JSON+SSE API under `/api`) on port
7860. Retrieval reads from **Qdrant Cloud** (the local disk index can't ship to Spaces).
Everything below is free-tier.

## 1. Qdrant Cloud — host the vector index (free 1 GB)

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) → create a **free 1 GB cluster**.
2. Copy the **cluster URL** and an **API key**.
3. Locally, put them in `.env`:
   ```
   QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
   QDRANT_API_KEY=xxxxxxxx
   ```
4. Push the index to the cloud (reuses the local embedding artifacts — no re-embedding):
   ```powershell
   .venv\Scripts\python ingestion\index_v2.py
   ```
   `config.qdrant_client()` sees `QDRANT_URL` and targets the cloud automatically. Confirm
   it prints `indexed 2082 hybrid points`.

## 2. Create the Space

1. [huggingface.co/new-space](https://huggingface.co/new-space) → **SDK: Docker**, name
   e.g. `mohafiz-climate-rag`, visibility Public.
2. Give the Space's `README.md` this frontmatter (Spaces requires it):
   ```yaml
   ---
   title: Mohafiz
   emoji: 🛡️
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

## 3. Set Space secrets (Settings → Variables and secrets)

| Secret | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | ✅ | LLM (generation + routing/grading) |
| `QDRANT_URL` | ✅ | cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | cloud API key |
| `GEMINI_API_KEY` | optional | failover (not yet wired) |
| `RELIEFWEB_APPNAME` | optional | enables the sitreps tool |
| `DAILY_LLM_BUDGET` | optional | default 300 answers/day |

## 4. Push the code to the Space

```powershell
git remote add space https://huggingface.co/spaces/<you>/mohafiz-climate-rag
git push space master:main
```

The Space builds the `Dockerfile` (bakes BGE-M3 + reranker into the image, ~6 GB — first
build is slow) and starts uvicorn. When the build finishes, the chat UI is live at the
Space URL.

## Notes
- **Cold start:** the free CPU Space sleeps after inactivity; first request after a wake
  can take ~1 min (models load into RAM). The UI already warns users about this.
- **CPU inference:** BGE-M3 + reranker run on CPU on the free tier — a few seconds per
  query. Fine for a demo.
- **Quota state** (`data/quota.sqlite`, cache + daily budget) is ephemeral across Space
  restarts — acceptable for a demo.
- **Keep it free:** rate limiting (10 q/hr/IP), response cache, and the daily budget cap
  (graceful degradation to retrieval-only) protect the Groq quota from a public URL.
