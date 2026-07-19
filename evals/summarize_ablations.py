"""Print the deduped ablation matrix (latest run per config) as a table."""
import json
from pathlib import Path

rows = [json.loads(l) for l in (Path(__file__).parent / "ablation_results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
seen = {}
for r in rows:
    if r.get("retriever") == "v2":
        seen[(r["mode"], r.get("rerank", False), r["set"])] = r

hdr = f'{"config":<16}{"easy hit@5":>12}{"easy MRR":>10}{"hard hit@5":>12}{"hard MRR":>10}'
print(hdr)
print("-" * len(hdr))
for mode in ["dense", "sparse", "hybrid"]:
    for rr in [False, True]:
        e = seen.get((mode, rr, "easy"))
        h = seen.get((mode, rr, "hard"))
        name = mode + ("+rerank" if rr else "")
        eh = f'{e["hit@5"]:.3f}' if e else "-"
        em = f'{e["MRR@10"]:.3f}' if e else "-"
        hh = f'{h["hit@5"]:.3f}' if h else "-"
        hm = f'{h["MRR@10"]:.3f}' if h else "-"
        print(f"{name:<16}{eh:>12}{em:>10}{hh:>12}{hm:>10}")
