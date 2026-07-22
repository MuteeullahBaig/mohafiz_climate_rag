# Deploying Mohafiz to Hugging Face Spaces (free Gradio Space)

The free HF tier hosts **Gradio** Spaces (Docker Spaces now require a paid plan). Mohafiz
therefore ships as a Gradio Space: `app.py` launches the chat UI directly, retrieval reads
from **Qdrant Cloud**, and the LLM is Groq. Everything below is free.

> Self-hosting alternative: the repo also has a `Dockerfile` + FastAPI service
> (`api/main.py`, JSON + SSE) for running anywhere you control (a VM, a paid Space, etc.).

## 1. Qdrant Cloud — host the vector index (free 1 GB)  ✅ done

1. [cloud.qdrant.io](https://cloud.qdrant.io) → free 1 GB cluster → copy the endpoint URL + an API key.
2. In `.env`: `QDRANT_URL=https://...:6333` and `QDRANT_API_KEY=...`
3. Push the index (reuses local embedding artifacts, no re-embedding):
   `.venv\Scripts\python ingestion\index_v2.py` → expect `indexed 2082 hybrid points`.

## 2. Create the Space

1. [huggingface.co/new-space](https://huggingface.co/new-space)
2. **SDK: Gradio** · template **Blank** · visibility **Public**
3. **Hardware:** pick **CPU Basic** (free, 16 GB — the app runs on CPU as-is). If only
   **ZeroGPU** is offered free, that works too (the CPU code still runs; ask me to add the
   `spaces` GPU decorator if you want it to use the GPU).
4. Name it e.g. `mohafiz-climate-rag`.

The Space's `README.md` already carries the required YAML frontmatter (`sdk: gradio`,
`app_file: app.py`) — no edit needed once you push this repo.

## 3. Set Space secrets (Settings → Variables and secrets)

| Secret | Required | Notes |
|---|---|---|
| `GROQ_API_KEY` | ✅ | generation + routing/grading |
| `QDRANT_URL` | ✅ | cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | cloud API key |
| `RELIEFWEB_APPNAME` | optional | enables the sitreps tool |
| `DAILY_LLM_BUDGET` | optional | default 300 answers/day |

## 4. Push the code to the Space

```powershell
git remote add space https://huggingface.co/spaces/<you>/mohafiz-climate-rag
git push space master:main
```
Authenticate with an HF **write** token ([huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))
when prompted. The Space then installs `requirements.txt`, downloads BGE-M3 + the reranker
on first boot, and launches `app.py`.

## Notes
- **First boot / cold start:** the models (~4.6 GB) download at startup — the first load can
  take several minutes; the UI warns users. While the Space stays warm, queries are quick.
- **CPU inference:** a few seconds per query on the free tier — fine for a demo.
- **Keeping it free under a public URL:** the daily LLM budget (degrades to retrieval-only),
  the response cache, and Gradio's bounded queue protect the Groq quota. (Per-IP rate
  limiting exists only on the FastAPI/self-host path.)
- **Quota/cache state** (`data/quota.sqlite`) is ephemeral across Space restarts — fine for a demo.
