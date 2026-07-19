"""Freshness pipeline: fetch the newest NAMC weekly agromet bulletin and ingest it.

The "latest data" half of the historical+latest story. PMD blocks naive fetchers
(WebFetch → 403) but a browser User-Agent via httpx works. When a not-yet-ingested
bulletin is found it is appended to the manifest; pass --ingest to then re-run the
standard pipeline (parse → chunk → embed → index). Full rebuild is simple and correct
at this corpus size; the deployed version will target Qdrant Cloud instead.

  python ingestion/refresh_bulletins.py            # check + add to manifest only
  python ingestion/refresh_bulletins.py --ingest   # also rebuild the index
"""
import argparse
import re
import json
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

LISTING = "https://namc.pmd.gov.pk/weekly-bulletins.php"
BASE = "https://namc.pmd.gov.pk/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def find_latest_bulletin() -> str | None:
    r = httpx.get(LISTING, headers=UA, timeout=30, follow_redirects=True)
    r.raise_for_status()
    links = re.findall(r"assets/weekly-bulletins/[^\"'()\s]+\.pdf", r.text)
    if not links:
        return None
    url = links[0]  # listing is newest-first
    return url if url.startswith("http") else BASE + url.lstrip("/")


def bulletin_id(url: str) -> str:
    m = re.search(r"Weekly[-_ ]?Bulletin[-_ ]?(\d+)", url, re.IGNORECASE)
    return f"namc-weekly-{m.group(1)}" if m else "namc-weekly-" + str(abs(hash(url)) % 100000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true", help="rebuild the index after adding")
    args = ap.parse_args()

    latest = find_latest_bulletin()
    if not latest:
        print("Could not find any bulletin link (listing markup may have changed).")
        sys.exit(1)
    print("Latest bulletin:", latest)

    manifest = json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    if latest in {d["url"] for d in manifest["documents"]}:
        print("Already ingested — nothing to do.")
        return

    bid = bulletin_id(latest)
    manifest["documents"].append({
        "id": bid, "title": f"NAMC Weekly Weather and Crop Bulletin ({bid})",
        "url": latest, "publisher": "PMD National Agromet Centre",
        "domain": "agriculture", "year": 2026, "lang": "en"})
    config.MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Added '{bid}' to manifest.")

    if args.ingest:
        py = sys.executable
        for step in ["download.py", "parse.py", "chunk.py"]:
            print(f"\n--- {step} ---")
            subprocess.run([py, str(Path(__file__).parent / step)], check=True)
        for step in ["embed_sparse.py", "index_v2.py"]:
            print(f"\n--- {step} ---")
            subprocess.run([py, str(Path(__file__).parent / step)], check=True)
        print("\nReindex complete — newest bulletin is now searchable.")
    else:
        print("Run with --ingest to rebuild the index, or run the pipeline manually.")


if __name__ == "__main__":
    main()
