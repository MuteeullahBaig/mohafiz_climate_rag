"""Download the corpus PDFs listed in manifest.json into data/raw/.

Stdlib-only on purpose: runs before the heavy dependencies are installed.
Skips entries marked "manual" (bot-protected sources) and files already present.
"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def download(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        while True:
            block = resp.read(1024 * 256)
            if not block:
                break
            f.write(block)
    return dest.stat().st_size


def main():
    manifest = json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    config.RAW.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, []
    for doc in manifest["documents"]:
        dest = config.RAW / f"{doc['id']}.pdf"
        if doc.get("manual"):
            status = "present (manual)" if dest.exists() else "SKIPPED - manual download needed"
            print(f"  [{doc['id']}] {status}")
            continue
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"  [{doc['id']}] already downloaded ({dest.stat().st_size:,} bytes)")
            ok += 1
            continue
        try:
            size = download(doc["url"], dest)
            magic = dest.open("rb").read(5)
            if magic != b"%PDF-":
                failed.append((doc["id"], f"not a PDF (starts with {magic!r})"))
                dest.unlink()
                continue
            print(f"  [{doc['id']}] downloaded {size:,} bytes")
            ok += 1
        except Exception as e:
            failed.append((doc["id"], str(e)))
    print(f"\n{ok} downloaded/present, {len(failed)} failed")
    for doc_id, err in failed:
        print(f"  FAILED {doc_id}: {err}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
