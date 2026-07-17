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
COLLECTION = "mohafiz_v1"

EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

# Chunking
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
