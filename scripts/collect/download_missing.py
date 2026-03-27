#!/usr/bin/env python3
"""Download missing PDFs for papers in bamboo_curated.json.

Handles both OpenReview and arXiv sources. Tries arXiv first (faster),
falls back to OpenReview with browser headers.

Usage:
    python scripts/collect/download_missing.py
    python scripts/collect/download_missing.py --workers 8
"""
import json
import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False

BASE = Path(__file__).resolve().parent.parent.parent
PDF_DIR = BASE / "data" / "paper_pdfs"
CURATED = BASE / "data" / "bamboo_curated.json"


def is_valid(path):
    if not path.exists() or path.stat().st_size < 10000:
        return False
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            return False
    if HAS_PDFIUM:
        try:
            doc = pdfium.PdfDocument(str(path))
            doc.close()
            return True
        except Exception:
            return False
    return True


def download_one(paper, max_retries=3):
    pid = paper["paper_id"]
    pp = PDF_DIR / f"{pid}.pdf"

    if is_valid(pp):
        return pid, True, "already_valid"

    urls = []

    # Try arXiv first (faster, no anti-bot)
    if paper.get("arxiv_id"):
        urls.append(("arxiv", f"https://arxiv.org/pdf/{paper['arxiv_id']}"))

    # OpenReview PDF
    paper_url = paper.get("paper_url", "")
    if "openreview.net" in paper_url and "?id=" in paper_url:
        or_id = paper_url.split("?id=")[1].split("&")[0]
        urls.append(("openreview", f"https://openreview.net/pdf?id={or_id}"))

    if not urls:
        return pid, False, "no_url"

    for source, url in urls:
        headers = [
            "-H", "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        ]
        if source == "openreview":
            headers += [
                "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "-H", "Referer: https://openreview.net/",
            ]

        for attempt in range(max_retries):
            try:
                subprocess.run(
                    ["curl", "-sL", "-o", str(pp), "--max-time", "180",
                     "--retry", "2", "--retry-delay", "3"] + headers + [url],
                    capture_output=True,
                    timeout=240,
                )
                if is_valid(pp):
                    return pid, True, source
                pp.unlink(missing_ok=True)
            except Exception:
                pp.unlink(missing_ok=True)

            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))

    return pid, False, "all_failed"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(CURATED) as f:
        papers = json.load(f)

    pdf_files = set(
        f.replace(".pdf", "")
        for f in os.listdir(PDF_DIR) if f.endswith(".pdf")
    )

    missing = [p for p in papers if p["paper_id"] not in pdf_files
               or not is_valid(PDF_DIR / f"{p['paper_id']}.pdf")]
    print(f"Missing PDFs: {len(missing)}")

    if not missing:
        print("All PDFs present!")
        return

    ok = 0
    fail = 0
    results = {"ok": [], "fail": []}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, p): p["paper_id"] for p in missing}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, success, source = fut.result()
            if success:
                ok += 1
                results["ok"].append(pid)
                print(f"[{i}/{len(missing)}] {pid}: OK ({source})")
            else:
                fail += 1
                results["fail"].append(pid)
                print(f"[{i}/{len(missing)}] {pid}: FAILED ({source})")

    print(f"\nDone: {ok} downloaded, {fail} failed")
    if results["fail"]:
        print(f"Failed IDs: {results['fail']}")


if __name__ == "__main__":
    main()
