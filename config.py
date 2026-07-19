"""Shared paths and constants for the Mohafiz pipeline."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PARSED = DATA / "parsed"
CHUNKS = DATA / "chunks"
CHUNKS_FILE = CHUNKS / "chunks.jsonl"
MANIFEST = ROOT / "ingestion" / "manifest.json"

# Qdrant embedded local mode — no server/Docker needed for dev
QDRANT_PATH = str(DATA / "qdrant_local")
COLLECTION = "mohafiz_v1"          # W1: dense-only
COLLECTION_V2 = "mohafiz_v2"       # W2: named vectors (dense + sparse) for hybrid

# W2 embedding artifacts (reusable across re-indexing)
DENSE_NPY = DATA / "embeddings_dense.npy"
SPARSE_JSONL = DATA / "embeddings_sparse.jsonl"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

# Chunking
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
