# Golden evaluation set

One JSON object per line in `*.jsonl` files. Schema:

```json
{
  "qid": "dis-en-001",
  "lang": "en",
  "domain": "disaster",
  "question": "What are the pre-monsoon preparedness actions NDMA assigns to PDMAs?",
  "reference_answer": "Short human-verified answer, grounded in the source.",
  "relevant_doc_id": "monsoon-cp-2025",
  "relevant_pages": [14, 15]
}
```

Rules:
- `relevant_pages` is page-level ground truth (not chunk IDs) so re-chunking in W2
  ablations never invalidates the set.
- `reference_answer` must be verifiable in the named pages — no outside knowledge.
- qid prefixes: `dis` disaster, `pol` policy, `agr` agriculture (W4), `rte` routing (W3),
  `una` unanswerable (W3). Language tag: `en` / `ur`.
- Every item is human-verified before it counts — LLM-drafted items are marked
  `"draft": true` until reviewed.
