"""Dense retrieval baseline over the Qdrant local index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient


class DenseSearcher:
    def __init__(self):
        self.model = BGEM3FlagModel(config.EMBED_MODEL, use_fp16=torch.cuda.is_available())
        self.client = QdrantClient(path=config.QDRANT_PATH)

    def search(self, query: str, k: int = 5) -> list[dict]:
        vec = self.model.encode(
            [query], return_dense=True, return_sparse=False, return_colbert_vecs=False
        )["dense_vecs"][0]
        try:
            hits = self.client.query_points(
                collection_name=config.COLLECTION, query=vec.tolist(), limit=k
            ).points
        except AttributeError:  # older qdrant-client
            hits = self.client.search(
                collection_name=config.COLLECTION, query_vector=vec.tolist(), limit=k
            )
        return [{"score": h.score, **h.payload} for h in hits]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What should communities do before a flood?"
    searcher = DenseSearcher()
    for r in searcher.search(q, k=5):
        heading = " > ".join(r["headings"][-2:]) if r["headings"] else "(no heading)"
        print(f"  {r['score']:.3f}  [{r['doc_id']} p.{r['pages']}] {heading}")
        print(f"         {r['text'][:160].replace(chr(10),' ')}...")
