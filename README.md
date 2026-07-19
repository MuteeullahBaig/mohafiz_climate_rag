# Mohafiz (محافظ)

Bilingual (English + Urdu) **agentic RAG** assistant for Pakistan's climate reality — disaster
preparedness, climate policy, and climate-smart agriculture — grounded in official documents
(NDMA, Ministry of Climate Change) with citations, plus live-data tools for weather,
earthquakes, and disaster alerts.

> Status: Weeks 1–4 in place — ingestion + eval harness, hybrid retrieval with reranking,
> the LangGraph agent (router, CRAG, groundedness, live tools), and the climate-smart
> agriculture module with domain routing and a bulletin-freshness pipeline. Public UI and
> the deployed demo link land in Week 5.
>
> Corpus: 10 official documents (NDMA, Ministry of Climate Change, PMD Agromet, Punjab
> Agriculture) across three domains — disaster, policy, agriculture — → 2,082 section-aware chunks.

## Agent flow

```
classify ─┬─ emergency ── helplines ─────────────────────────► answer
          ├─ refuse ─────────────────────────────────────────► answer
          ├─ live ──── weather/quake/alert/sitrep tool ──┐
          ├─ retrieve ─ hybrid+rerank ─ CRAG grade ─┐    │
          │                    ▲ (rewrite, max 1) ◄──┘    │
          └─ both ──── retrieve ──────────────────────────┼─► generate (70B)
                                                          ►│      │
                                          Self-RAG groundedness ◄─┘
                                                    │
                                        grounded? ──┴── no ─► regenerate / abstain+cite
```

Cheap-first cascade: the 8B model runs routing, CRAG grading, and groundedness; the 70B
is reserved for final answer generation only (protects the scarce 70B daily token budget).

## Results

### Retrieval — W2 ablation matrix (BGE-M3 → Qdrant, cross-encoder = BGE-reranker-v2-m3)

The **easy** golden set (questions phrased near the source) saturates hit@5, so the **hard**
set (25 colloquial paraphrases, same page-level ground truth, low lexical overlap) is the
one that discriminates.

| Config | easy hit@5 | easy MRR@10 | hard hit@5 | hard MRR@10 |
|---|---|---|---|---|
| dense | 1.000 | 0.883 | 0.760 | 0.641 |
| **dense + rerank** | 1.000 | 0.903 | **0.800** | **0.673** |
| sparse | 0.960 | 0.835 | 0.600 | 0.386 |
| sparse + rerank | 1.000 | 0.903 | 0.680 | 0.607 |
| hybrid (RRF) | 1.000 | 0.883 | 0.760 | 0.614 |
| **hybrid + rerank** | 1.000 | 0.900 | **0.800** | 0.649 |

**Findings:**
1. **Reranking is the decisive lever** — it lifts hard-set hit@5 for every mode and rescues
   ranking dramatically (sparse MRR 0.386 → 0.607, +57%).
2. **Naive RRF fusion does not beat dense** on paraphrased queries — sparse's lexical noise
   offsets its gains. An honest negative result: fusion was not the win, reranking was.
3. Production config: **hybrid + rerank** — ties the best hit@5 (0.800) and stays robust to
   keyword-style queries where pure dense is weaker.
4. Remaining hard misses are queries whose gold chunk falls outside the top-20 prefetch
   pool (reranking can't recover what retrieval never surfaced) — a prefetch-depth / query-
   expansion item for later.

### Generation — W1 RAGAS baseline (naive dense RAG, 25 questions, 8B judge)

| faithfulness | context precision | context recall |
|---|---|---|
| 0.920 | 0.930 | 0.893 |

Coverage 25/25 (recall 24/25). Judge = `llama-3.1-8b-instant` on its own token quota;
answers cached so re-judging costs no generation. (First attempt scored only 4/25 before
the 70B daily budget was exhausted mid-run — the split-model design is the fix.)

### Agent — W3/W4 routing (golden sets)

| route accuracy (10) | tool-selection (5) | domain accuracy (12) |
|---|---|---|
| 10/10 (1.000) | 5/5 (1.000) | 11/12 (0.917) |

All five routes correct (retrieve / live / both / emergency / refuse), including Roman-Urdu
emergency detection and travel-safety ("both") classification. Domain routing classifies
disaster / agriculture / policy for metadata-aware retrieval; the one miss is a borderline
institutional-vs-disaster case.

### Agriculture module — W4 retrieval (agri golden set, hybrid+rerank)

| hit@5 | MRR@10 |
|---|---|
| 8/9 (0.889) | 0.685 |

Agri corpus: 2 PMD agromet bulletins + Punjab Rabi crops book + crops calendar. A
`refresh_bulletins.py` freshness job scrapes the NAMC listing for the newest weekly
bulletin and re-ingests it — the "latest data" half of a historical+latest corpus.

**Honest coverage note:** free Pakistani agri sources skew quantitative (agromet metrics,
crop-production estimates) rather than agronomic "how-to" prose — the package-of-practices
guides were only on Indian sites (excluded). The crops calendar is a visual grid that OCR
could not turn into usable text. Agri Q&A therefore leans on bulletin advisories and crop
statistics.

## Live-data tools (all keyless/free)

| Tool | Source | Status |
|---|---|---|
| weather + soil/ET₀ | Open-Meteo | ✅ |
| earthquakes (PK bbox) | USGS FDSN | ✅ |
| disaster alerts | GDACS RSS | ✅ |
| situation reports | ReliefWeb V2 | ⚙️ needs an approved `RELIEFWEB_APPNAME` (degrades gracefully) |

## Repo layout

```
ingestion/   download · parse (Docling) · chunk · embed_sparse · index_v2 · refresh_bulletins
retrieval/   search.py — DenseSearcher (v1) + Retriever (dense/sparse/hybrid ±rerank)
agent/       graph.py (LangGraph) · llm · prompts · state · helplines · tools/live_data
evals/       golden/ · run_retrieval_eval (ablations+W&B) · run_ragas · run_routing_eval
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # add GROQ_API_KEY (required); WANDB / RELIEFWEB optional
# ingest → index → evaluate
.venv\Scripts\python ingestion\download.py
.venv\Scripts\python ingestion\parse.py
.venv\Scripts\python ingestion\chunk.py
.venv\Scripts\python ingestion\embed_sparse.py
.venv\Scripts\python ingestion\index_v2.py
.venv\Scripts\python evals\run_retrieval_eval.py --mode hybrid --rerank --set hard
.venv\Scripts\python agent\graph.py "what does the National Adaptation Plan say about groundwater?"
```
