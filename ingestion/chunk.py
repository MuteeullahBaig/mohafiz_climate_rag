"""Section-aware chunking of parsed documents via Docling's HybridChunker.

Emits data/chunks/chunks.jsonl — one record per chunk:
  chunk_id, doc_id, title, domain, year, lang, publisher, source_url,
  headings (section path), pages, text (raw), embed_text (heading-contextualized)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer


def chunk_pages(chunk) -> list[int]:
    pages = set()
    for item in chunk.meta.doc_items:
        for prov in getattr(item, "prov", []) or []:
            pages.add(prov.page_no)
    return sorted(pages)


def main():
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in manifest["documents"]}
    config.CHUNKS.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    chunker = HybridChunker(
        tokenizer=tokenizer, max_tokens=config.MAX_CHUNK_TOKENS, merge_peers=True
    )

    n_total = 0
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as out:
        for parsed in sorted(config.PARSED.glob("*.json")):
            if parsed.name.startswith("_"):
                continue
            doc_id = parsed.stem
            meta = by_id.get(doc_id, {})
            doc = DoclingDocument.model_validate_json(parsed.read_text(encoding="utf-8"))
            n_doc = 0
            for i, chunk in enumerate(chunker.chunk(doc)):
                try:
                    embed_text = chunker.contextualize(chunk)
                except AttributeError:  # older docling_core
                    embed_text = chunk.text
                record = {
                    "chunk_id": f"{doc_id}::{i:04d}",
                    "doc_id": doc_id,
                    "title": meta.get("title", doc_id),
                    "domain": meta.get("domain", "unknown"),
                    "year": meta.get("year"),
                    "lang": meta.get("lang", "en"),
                    "publisher": meta.get("publisher"),
                    "source_url": meta.get("url"),
                    "headings": list(chunk.meta.headings or []),
                    "pages": chunk_pages(chunk),
                    "text": chunk.text,
                    "embed_text": embed_text,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_doc += 1
            n_total += n_doc
            print(f"  [{doc_id}] {n_doc} chunks")
    print(f"\n{n_total} chunks -> {config.CHUNKS_FILE}")


if __name__ == "__main__":
    main()
