"""Retrieval metrics on the golden set: hit@5 and MRR@10, ablation-ready.

Ground truth = (relevant_doc_id, relevant_pages); page-level so re-chunking never
invalidates the set. Modes/flags select the retriever config; every run appends
one row to ablation_results.jsonl and (with --wandb) logs to the mohafiz-rag project.

Examples:
  python evals/run_retrieval_eval.py --retriever v1                 # W1 baseline (v1 collection)
  python evals/run_retrieval_eval.py --mode dense                   # dense on v2
  python evals/run_retrieval_eval.py --mode hybrid --rerank --set hard --wandb
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GOLDEN_DIR = Path(__file__).parent / "golden"
RESULTS = Path(__file__).parent / "ablation_results.jsonl"

# "all" deliberately matches only retrieval sets — routing_/unanswerable_ files
# have different schemas and are consumed by their own eval scripts.
SETS = {"easy": "en_v1.jsonl", "hard": "en_v1_hard.jsonl", "all": "en_v1*.jsonl"}


def load_golden(pattern: str = "*.jsonl") -> list[dict]:
    items = []
    for f in sorted(GOLDEN_DIR.glob(pattern)):
        with open(f, encoding="utf-8") as fh:
            items += [json.loads(line) for line in fh if line.strip()]
    return items


def is_relevant(hit: dict, item: dict) -> bool:
    if hit["doc_id"] != item["relevant_doc_id"]:
        return False
    return bool(set(item["relevant_pages"]) & set(hit["pages"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="v2", choices=["v1", "v2"],
                    help="v1 = W1 DenseSearcher on the old collection; v2 = configurable Retriever")
    ap.add_argument("--mode", default="dense", choices=["dense", "sparse", "hybrid"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--prefetch-k", type=int, default=20)
    ap.add_argument("--set", default="easy", choices=list(SETS))
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    golden = load_golden(SETS[args.set])
    if not golden:
        print(f"No golden items for set '{args.set}'")
        sys.exit(1)

    if args.retriever == "v1":
        from retrieval.search import DenseSearcher
        searcher = DenseSearcher()
        run_name = f"v1-dense-{args.set}"
        cfg = {"retriever": "v1-dense", "set": args.set, "k": args.k}
    else:
        from retrieval.search import Retriever
        searcher = Retriever(mode=args.mode, rerank=args.rerank)
        rr = "+rerank" if args.rerank else ""
        run_name = f"{args.mode}{rr}-{args.set}"
        cfg = {"retriever": "v2", "mode": args.mode, "rerank": args.rerank,
               "k": args.k, "prefetch_k": args.prefetch_k, "set": args.set}

    hits_at_k, rr_sum = 0, 0.0
    misses = []
    for item in golden:
        results = searcher.search(item["question"], k=10)
        ranks = [i for i, r in enumerate(results, start=1) if is_relevant(r, item)]
        first = ranks[0] if ranks else None
        if first and first <= args.k:
            hits_at_k += 1
        rr_sum += (1.0 / first) if first else 0.0
        if not first or first > args.k:
            misses.append((item["qid"], first))

    n = len(golden)
    metrics = {f"hit@{args.k}": round(hits_at_k / n, 4), "MRR@10": round(rr_sum / n, 4)}
    print(f"\n{run_name} on {n} questions:")
    for name, val in metrics.items():
        print(f"  {name} = {val}")
    if misses:
        print(f"  missed@{args.k} ({len(misses)}): " + ", ".join(
            f"{q}(rank {r})" for q, r in misses))

    row = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "run": run_name, **cfg, "n": n, **metrics,
           "misses": [q for q, _ in misses]}
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"appended -> {RESULTS.name}")

    if args.wandb:
        import wandb
        run = wandb.init(project="mohafiz-rag", name=run_name, config=cfg)
        run.log(metrics)
        run.finish()
        print("logged to W&B")


if __name__ == "__main__":
    main()
