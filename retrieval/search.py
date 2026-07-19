"""Retrieval over the Qdrant local index.

W1: DenseSearcher (dense-only, v1 collection) — kept for compatibility.
W2: Retriever — modes dense / sparse / hybrid (server-side RRF fusion via the
    Query API) on the v2 named-vector collection, with optional BGE cross-encoder
    reranking of the prefetch pool.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

import torch
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, models

_USE_GPU = torch.cuda.is_available()


class DenseSearcher:
    """W1 baseline searcher (v1 collection, dense only)."""

    def __init__(self):
        self.model = BGEM3FlagModel(config.EMBED_MODEL, use_fp16=_USE_GPU)
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


class Retriever:
    """W2 configurable retriever over the v2 hybrid collection."""

    def __init__(self, mode: str = "hybrid", rerank: bool = False):
        assert mode in ("dense", "sparse", "hybrid"), mode
        self.mode = mode
        self.rerank = rerank
        self.model = BGEM3FlagModel(config.EMBED_MODEL, use_fp16=_USE_GPU)
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self._reranker = None  # lazy — 2.3GB model

    @property
    def reranker(self):
        # Driven via transformers directly rather than FlagEmbedding's FlagReranker,
        # which calls tokenizer.prepare_for_model() — removed from the slow tokenizer
        # in transformers 5.x. Fast tokenizer + sigmoid(logits) is the robust path.
        if self._reranker is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(config.RERANKER_MODEL, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(config.RERANKER_MODEL)
            model.eval()
            if _USE_GPU:
                model = model.half().cuda()
            self._reranker = (tok, model)
        return self._reranker

    def _rerank_scores(self, query: str, texts: list[str]) -> list[float]:
        tok, model = self.reranker
        pairs = [[query, t] for t in texts]
        with torch.no_grad():
            inputs = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt")
            if _USE_GPU:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = model(**inputs).logits.view(-1).float()
            return torch.sigmoid(logits).cpu().tolist()

    def _encode_query(self, query: str):
        out = self.model.encode(
            [query],
            return_dense=self.mode in ("dense", "hybrid"),
            return_sparse=self.mode in ("sparse", "hybrid"),
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"][0].tolist() if self.mode in ("dense", "hybrid") else None
        sparse = None
        if self.mode in ("sparse", "hybrid"):
            w = out["lexical_weights"][0]
            sparse = models.SparseVector(indices=[int(k) for k in w.keys()], values=list(w.values()))
        return dense, sparse

    def search(self, query: str, k: int = 5, prefetch_k: int = 20) -> list[dict]:
        dense, sparse = self._encode_query(query)
        fetch = prefetch_k if self.rerank else k

        if self.mode == "dense":
            res = self.client.query_points(
                collection_name=config.COLLECTION_V2, query=dense, using="dense", limit=fetch
            )
        elif self.mode == "sparse":
            res = self.client.query_points(
                collection_name=config.COLLECTION_V2, query=sparse, using="sparse", limit=fetch
            )
        else:  # hybrid: server-side RRF over both prefetch lists
            res = self.client.query_points(
                collection_name=config.COLLECTION_V2,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=prefetch_k),
                    models.Prefetch(query=sparse, using="sparse", limit=prefetch_k),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=fetch,
            )
        hits = [{"score": h.score, **h.payload} for h in res.points]

        if self.rerank and hits:
            scores = self._rerank_scores(query, [h["text"] for h in hits])
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:k]
        return hits[:k]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid"])
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    r = Retriever(mode=args.mode, rerank=args.rerank)
    for h in r.search(" ".join(args.query), k=5):
        heading = " > ".join(h["headings"][-2:]) if h["headings"] else "(no heading)"
        score = h.get("rerank_score", h["score"])
        print(f"  {score:.3f}  [{h['doc_id']} p.{h['pages']}] {heading}")
        print(f"         {h['text'][:160].replace(chr(10), ' ')}...")
