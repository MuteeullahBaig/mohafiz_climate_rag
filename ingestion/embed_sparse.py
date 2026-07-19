"""Re-encode all chunks with BGE-M3 dense + sparse, saving artifacts to disk.

Separated from indexing on purpose: the RAGAS job may hold the Qdrant local-mode
lock, and embeddings are reusable across collection rebuilds/ablations.
Outputs: data/embeddings_dense.npy, data/embeddings_sparse.jsonl (chunk order
matches chunks.jsonl line order).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


def main():
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    print(f"{len(chunks)} chunks to encode (dense + sparse)")

    use_gpu = torch.cuda.is_available()
    model = BGEM3FlagModel(config.EMBED_MODEL, use_fp16=use_gpu)

    t0 = time.time()
    out = model.encode(
        [c["embed_text"] for c in chunks],
        batch_size=16 if use_gpu else 4,
        max_length=config.MAX_CHUNK_TOKENS,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    np.save(config.DENSE_NPY, out["dense_vecs"])
    with open(config.SPARSE_JSONL, "w", encoding="utf-8") as f:
        for cid, weights in zip((c["chunk_id"] for c in chunks), out["lexical_weights"]):
            f.write(json.dumps({"chunk_id": cid, "weights": {str(k): float(v) for k, v in weights.items()}}) + "\n")
    print(f"encoded in {time.time()-t0:.0f}s -> {config.DENSE_NPY.name}, {config.SPARSE_JSONL.name}")


if __name__ == "__main__":
    main()
