"""Router accuracy on golden sets — roadmap W3/W4 metrics.

Runs only the classify node (cheap 8B model, no Qdrant), safe to run while the
retrieval index is locked. Checks whichever of expected_route / expected_tool /
expected_domain each item declares, so one script serves both the routing set
(routes + tools) and the domain set (domain classification).

  python evals/run_routing_eval.py --set routing   # routes + tools
  python evals/run_routing_eval.py --set domain     # domain classification
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import classify

SETS = {"routing": "routing_v1.jsonl", "domain": "domain_v1.jsonl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="routing", choices=list(SETS))
    args = ap.parse_args()

    path = Path(__file__).parent / "golden" / SETS[args.set]
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    tally = {"route": [0, 0], "tool": [0, 0], "domain": [0, 0]}  # [correct, applicable]
    rows = []
    for it in items:
        out = classify({"question": it["question"]})
        checks = {}
        for field, got_key in [("route", "route"), ("tool", "tool"), ("domain", "domain")]:
            exp_key = f"expected_{field}"
            if exp_key in it and it[exp_key] is not None:
                ok = out.get(got_key) == it[exp_key]
                tally[field][0] += ok
                tally[field][1] += 1
                checks[field] = (it[exp_key], out.get(got_key), ok)
        allok = all(c[2] for c in checks.values())
        flag = "OK " if allok else "XX "
        detail = " ".join(f"{f}:{exp}->{got}{'' if ok else '!!'}"
                          for f, (exp, got, ok) in checks.items())
        print(f"  {flag}{it['qid']}: {detail}")
        rows.append({"qid": it["qid"], "checks": {f: {"expected": e, "got": g, "ok": o}
                                                  for f, (e, g, o) in checks.items()}})

    print()
    summary = {}
    for field, (c, n) in tally.items():
        if n:
            summary[f"{field}_accuracy"] = round(c / n, 4)
            print(f"{field} accuracy = {c}/{n} = {c/n:.3f}")

    out_file = Path(__file__).parent / f"routing_results_{args.set}.json"
    out_file.write_text(json.dumps({"set": args.set, **summary, "detail": rows}, indent=2), encoding="utf-8")
    print(f"saved -> {out_file.name}")


if __name__ == "__main__":
    main()
