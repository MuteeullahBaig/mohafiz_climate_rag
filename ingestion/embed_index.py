"""Embed chunks with BGE-M3 (dense) and index into Qdrant embedded local mode.

W1 baseline is dense-only; sparse vectors join in W2 for hybrid ablations.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models


def load_chunks() -> list[dict]:
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    print(f"{len(chunks)} chunks to embed")

    use_gpu = torch.cuda.is_available()
    model = BGEM3FlagModel(config.EMBED_MODEL, use_fp16=use_gpu)
    print(f"BGE-M3 loaded ({'GPU' if use_gpu else 'CPU'})")

    t0 = time.time()
    out = model.encode(
        [c["embed_text"] for c in chunks],
        batch_size=16 if use_gpu else 4,
        max_length=config.MAX_CHUNK_TOKENS,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"]
    print(f"embedded in {time.time()-t0:.0f}s")

    client = QdrantClient(path=config.QDRANT_PATH)
    if client.collection_exists(config.COLLECTION):
        client.delete_collection(config.COLLECTION)
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE),
    )
    BATCH = 256
    for start in range(0, len(chunks), BATCH):
        end = min(start + BATCH, len(chunks))
        client.upsert(
            collection_name=config.COLLECTION,
            points=[
                models.PointStruct(id=i, vector=dense[i].tolist(), payload=chunks[i])
                for i in range(start, end)
            ],
        )
    print(f"indexed {len(chunks)} points into '{config.COLLECTION}' at {config.QDRANT_PATH}")


if __name__ == "__main__":
    main()
