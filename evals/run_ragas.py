"""RAGAS harness: naive-RAG answers + LLM-judge metrics on the golden set.

Quota design (measured 2026-07-17: one naive pass = ~2.4x the 70B's 100K TPD free budget):
  - Generator: llama-3.3-70b-versatile (the model being benchmarked)
  - Judge:     llama-3.1-8b-instant by default — separate per-model TPD quota
  - Answers are cached to answers_baseline.jsonl; --eval-only reruns judging at
    zero generation cost. --gen-only does the opposite.

W1 metrics (LLM-judge only): faithfulness, context_precision (w/ reference), context_recall.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: F401  (path setup)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GENERATOR_MODEL = "llama-3.3-70b-versatile"
DEFAULT_JUDGE = "llama-3.1-8b-instant"
ANSWERS_FILE = Path(__file__).parent / "answers_baseline.jsonl"

SYSTEM = (
    "You are Mohafiz, an assistant for Pakistan disaster preparedness and climate policy. "
    "Answer ONLY from the provided context passages. Cite the source in brackets like "
    "[doc_id p.PAGE] after each claim. If the context does not contain the answer, say "
    "'The provided documents do not cover this.' Keep answers under 150 words."
)


def generate_answers(golden: list[dict], k: int) -> list[dict]:
    from groq import Groq
    from retrieval.search import DenseSearcher

    searcher = DenseSearcher()
    client = Groq()
    rows = []
    for item in golden:
        contexts = searcher.search(item["question"], k=k)
        ctx_block = "\n\n".join(f"[{c['doc_id']} p.{c['pages']}] {c['text']}" for c in contexts)
        resp = client.chat.completions.create(
            model=GENERATOR_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{ctx_block}\n\nQuestion: {item['question']}"},
            ],
        )
        rows.append(
            {
                "qid": item["qid"],
                "user_input": item["question"],
                "retrieved_contexts": [c["text"] for c in contexts],
                "response": resp.choices[0].message.content,
                "reference": item["reference_answer"],
            }
        )
        print(f"  answered {item['qid']}")
        time.sleep(1)  # free-tier RPM
    ANSWERS_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    print(f"cached {len(rows)} answers -> {ANSWERS_FILE}")
    return rows


def evaluate_answers(rows: list[dict], judge_model: str):
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig

    judge = LangchainLLMWrapper(ChatGroq(model=judge_model, temperature=0))
    dataset = EvaluationDataset.from_list([{k: v for k, v in r.items() if k != "qid"} for r in rows])
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=judge),
            LLMContextPrecisionWithReference(llm=judge),
            LLMContextRecall(llm=judge),
        ],
        run_config=RunConfig(max_workers=1, timeout=300, max_retries=15, max_wait=90),
    )
    print(f"\nRAGAS (judge={judge_model}):")
    print(result)
    out = Path(__file__).parent / "ragas_baseline.json"
    df = result.to_pandas()
    df.insert(0, "qid", [r["qid"] for r in rows])
    df.to_json(out, orient="records", indent=2)
    import math
    for col in ["faithfulness", "llm_context_precision_with_reference", "context_recall"]:
        if col in df:
            valid = df[col].notna().sum()
            print(f"  coverage {col}: {valid}/{len(df)}")
    print(f"per-question scores -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge", default=DEFAULT_JUDGE)
    ap.add_argument("--gen-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="judge cached answers, no generation")
    args = ap.parse_args()

    if args.eval_only:
        rows = [json.loads(l) for l in ANSWERS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        print(f"loaded {len(rows)} cached answers")
    else:
        from run_retrieval_eval import load_golden
        golden = load_golden()
        if args.limit:
            golden = golden[: args.limit]
        rows = generate_answers(golden, args.k)

    if not args.gen_only:
        evaluate_answers(rows, args.judge)


if __name__ == "__main__":
    main()
