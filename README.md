# Mohafiz (محافظ)

Bilingual (English + Urdu) agentic RAG assistant for Pakistan's climate reality — disaster
preparedness, climate policy, and climate-smart agriculture — grounded in official documents
(NDMA, Ministry of Climate Change, PMD Agromet) with citations, plus live-data tools for
weather, earthquakes, and disaster alerts.

> Week 1 status: corpus ingestion, baseline dense retrieval, and the eval harness.
> Architecture, eval tables, and the demo link land here as the build progresses.

## Week 1 baseline (2026-07-17)

Corpus: 6 official documents (NDMA / MoCC), 607 pages → **1,306 section-aware chunks**
(Docling HybridChunker, BGE-M3 tokenizer, max 512 tokens, metadata: domain/doc/year/pages/headings).

| Metric | Naive dense RAG (BGE-M3 → Qdrant, top-5) |
|---|---|
| hit@5 (25-question golden set) | **1.000** |
| MRR@10 | **0.883** |
| RAGAS faithfulness / precision / recall | pending — see note |

**Notes (eval honesty log):**
- Golden v1 questions were drafted close to source phrasing → easy for the retriever;
  hit@5 is saturated and has no headroom for W2 ablations. W2 adds paraphrased,
  multi-hop, and unanswerable items to unsaturate the metric.
- First RAGAS pass exhausted the Groq free tier: 25 generations ≈ 80K tokens of the
  70B model's 100K/day budget, starving the judge phase (3-4/25 valid scores — not
  reportable). Fix implemented: judge moved to `llama-3.1-8b-instant` (separate
  per-model quota) and answers cached to disk (`--eval-only` reruns are
  generation-free). Full pass rescheduled post quota reset.

## Pipeline (W1)

```
ingestion/download.py     manifest.json → data/raw/*.pdf
ingestion/parse.py        Docling → data/parsed/*.json (+ .md preview)
ingestion/chunk.py        section-level chunks + metadata → data/chunks/chunks.jsonl
ingestion/embed_index.py  BGE-M3 dense → Qdrant (embedded local mode)
retrieval/search.py       dense top-k baseline
evals/                    golden set + retrieval metrics (hit@5, MRR@10) + RAGAS harness
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # add GROQ_API_KEY
```
