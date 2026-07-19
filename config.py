"""Shared paths and constants for the Mohafiz pipeline."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
RAW = DATA / "raw"
PARSED = DATA / "parsed"
CHUNKS = DATA / "chunks"
CHUNKS_FILE = CHUNKS / "chunks.jsonl"
MANIFEST = ROOT / "ingestion" / "manifest.json"

# Qdrant: embedded local mode for dev (no server needed); Qdrant Cloud for the
# deployed demo. Set QDRANT_URL (+ QDRANT_API_KEY) to switch to cloud.
QDRANT_PATH = str(DATA / "qdrant_local")
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
COLLECTION = "mohafiz_v1"          # W1: dense-only
COLLECTION_V2 = "mohafiz_v2"       # W2: named vectors (dense + sparse) for hybrid


def qdrant_client():
    """One client factory used everywhere — local path unless QDRANT_URL is set."""
    from qdrant_client import QdrantClient
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None, timeout=30)
    return QdrantClient(path=QDRANT_PATH)

# W2 embedding artifacts (reusable across re-indexing)
DENSE_NPY = DATA / "embeddings_dense.npy"
SPARSE_JSONL = DATA / "embeddings_sparse.jsonl"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

# Chunking
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
