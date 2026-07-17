"""Retrieval metrics on the golden set: hit@5 and MRR@10.

Ground truth = (relevant_doc_id, relevant_pages). A retrieved chunk counts as
relevant when it comes from that document AND overlaps at least one golden page.
Page-level (not chunk-id) ground truth survives re-chunking across W2 ablations.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from retrieval.search import DenseSearcher

GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden() -> list[dict]:
    items = []
    for f in sorted(GOLDEN_DIR.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            items += [json.loads(line) for line in fh if line.strip()]
    return items


def is_relevant(hit: dict, item: dict) -> bool:
    if hit["doc_id"] != item["relevant_doc_id"]:
        return False
    golden_pages = set(item["relevant_pages"])
    return bool(golden_pages & set(hit["pages"]))


def main():
    golden = load_golden()
    if not golden:
        print("No golden items found in evals/golden/*.jsonl")
        sys.exit(1)
    searcher = DenseSearcher()

    hits_at_5, rr_sum = 0, 0.0
    misses = []
    for item in golden:
        results = searcher.search(item["question"], k=10)
        ranks = [i for i, r in enumerate(results, start=1) if is_relevant(r, item)]
        first = ranks[0] if ranks else None
        if first and first <= 5:
            hits_at_5 += 1
        rr_sum += (1.0 / first) if first else 0.0
        if not first or first > 5:
            misses.append((item["qid"], first))

    n = len(golden)
    print(f"\nRetrieval baseline on {n} golden questions (dense-only, BGE-M3):")
    print(f"  hit@5  = {hits_at_5}/{n} = {hits_at_5/n:.3f}")
    print(f"  MRR@10 = {rr_sum/n:.3f}")
    if misses:
        print(f"\n  missed@5 ({len(misses)}):")
        for qid, rank in misses:
            print(f"    {qid}  (first relevant rank: {rank})")

    out = {"n": n, "hit@5": round(hits_at_5 / n, 4), "MRR@10": round(rr_sum / n, 4),
           "config": {"retriever": "dense", "model": config.EMBED_MODEL,
                      "max_chunk_tokens": config.MAX_CHUNK_TOKENS}}
    report = Path(__file__).parent / "baseline_results.json"
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {report}")


if __name__ == "__main__":
    main()
