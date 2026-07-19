"""Parse corpus PDFs with Docling into structured JSON (+ markdown preview).

Writes data/parsed/{id}.json (DoclingDocument) and {id}.md.
Flags likely-scanned PDFs (low text density) for the OCR fallback path.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

from docling.document_converter import DocumentConverter

LOW_TEXT_CHARS_PER_PAGE = 200  # below this, the PDF is probably scanned


def main():
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    config.PARSED.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    report = []

    for entry in manifest["documents"]:
        pdf = config.RAW / f"{entry['id']}.pdf"
        out_json = config.PARSED / f"{entry['id']}.json"
        if not pdf.exists():
            print(f"  [{entry['id']}] no PDF, skipping")
            continue
        if out_json.exists():
            print(f"  [{entry['id']}] already parsed")
            continue
        t0 = time.time()
        result = converter.convert(str(pdf))
        doc = result.document
        out_json.write_text(json.dumps(doc.export_to_dict()), encoding="utf-8")
        (config.PARSED / f"{entry['id']}.md").write_text(
            doc.export_to_markdown(), encoding="utf-8"
        )
        n_pages = len(doc.pages)
        n_chars = sum(len(t.text) for t in doc.texts)
        density = n_chars / max(n_pages, 1)
        flag = "  [!] LOW TEXT - likely scanned, needs OCR" if density < LOW_TEXT_CHARS_PER_PAGE else ""
        print(
            f"  [{entry['id']}] {n_pages} pages, {n_chars:,} chars "
            f"({density:.0f}/page) in {time.time()-t0:.0f}s{flag}"
        )
        report.append({"id": entry["id"], "pages": n_pages, "chars": n_chars,
                       "chars_per_page": round(density), "scanned_suspect": density < LOW_TEXT_CHARS_PER_PAGE})

    (config.PARSED / "_parse_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nDone. Report: data/parsed/_parse_report.json")


if __name__ == "__main__":
    main()
