"""Build the v2 hybrid collection: named dense vector + sparse vector per point.

Reads the disk artifacts from embed_sparse.py — run that first. Requires the
Qdrant local-mode lock to be free (no other process using data/qdrant_local).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import numpy as np
from qdrant_client import QdrantClient, models


def main():
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    dense = np.load(config.DENSE_NPY)
    with open(config.SPARSE_JSONL, encoding="utf-8") as f:
        sparse = [json.loads(line) for line in f]
    assert len(chunks) == len(dense) == len(sparse), "artifact length mismatch — rerun embed_sparse.py"
    for c, s in zip(chunks, sparse):
        assert c["chunk_id"] == s["chunk_id"], f"chunk order mismatch at {c['chunk_id']}"

    client = QdrantClient(path=config.QDRANT_PATH)
    if client.collection_exists(config.COLLECTION_V2):
        client.delete_collection(config.COLLECTION_V2)
    client.create_collection(
        collection_name=config.COLLECTION_V2,
        vectors_config={"dense": models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    BATCH = 128
    for start in range(0, len(chunks), BATCH):
        end = min(start + BATCH, len(chunks))
        points = []
        for i in range(start, end):
            w = sparse[i]["weights"]
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "dense": dense[i].tolist(),
                        "sparse": models.SparseVector(
                            indices=[int(k) for k in w.keys()], values=list(w.values())
                        ),
                    },
                    payload=chunks[i],
                )
            )
        client.upsert(collection_name=config.COLLECTION_V2, points=points)
    print(f"indexed {len(chunks)} hybrid points into '{config.COLLECTION_V2}'")


if __name__ == "__main__":
    main()
