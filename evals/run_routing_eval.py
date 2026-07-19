"""Routing accuracy on the routing golden set — a roadmap W3 metric.

Runs only the classify node (cheap 8B model, no Qdrant), so it's safe to run while
the retrieval index is locked by an ablation job.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import classify

GOLDEN = Path(__file__).parent / "golden" / "routing_v1.jsonl"


def main():
    items = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    correct_route = correct_tool = tool_applicable = 0
    rows = []
    for it in items:
        out = classify({"question": it["question"]})
        route_ok = out["route"] == it["expected_route"]
        correct_route += route_ok
        tool_ok = None
        if it.get("expected_tool"):
            tool_applicable += 1
            tool_ok = out.get("tool") == it["expected_tool"]
            correct_tool += bool(tool_ok)
        rows.append((it["qid"], it["expected_route"], out["route"], route_ok,
                     it.get("expected_tool"), out.get("tool"), tool_ok))
        flag = "OK " if route_ok else "XX "
        print(f"  {flag}{it['qid']}: expected={it['expected_route']:<9} got={out['route']:<9}"
              f" tool={out.get('tool')}")

    n = len(items)
    print(f"\nrouting accuracy = {correct_route}/{n} = {correct_route/n:.3f}")
    if tool_applicable:
        print(f"tool accuracy    = {correct_tool}/{tool_applicable} = {correct_tool/tool_applicable:.3f}")

    out_file = Path(__file__).parent / "routing_results.json"
    out_file.write_text(json.dumps(
        {"n": n, "route_accuracy": round(correct_route / n, 4),
         "tool_accuracy": round(correct_tool / tool_applicable, 4) if tool_applicable else None,
         "detail": [{"qid": r[0], "expected": r[1], "got": r[2], "route_ok": r[3],
                     "expected_tool": r[4], "got_tool": r[5], "tool_ok": r[6]} for r in rows]},
        indent=2), encoding="utf-8")
    print(f"saved -> {out_file.name}")


if __name__ == "__main__":
    main()
