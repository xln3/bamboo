#!/usr/bin/env python3
"""Download PDFs from task file (for second server).

Usage:
    python scripts/collect/download_task.py                          # download all
    python scripts/collect/download_task.py --workers 4              # fewer workers
    python scripts/collect/download_task.py --venue ICML             # only ICML
    python scripts/collect/download_task.py --task data/task_neurips_iclr_icml.json
"""
import argparse
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False
    print("Warning: pypdfium2 not installed, using basic PDF validation only")


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


def download_one(task, pdf_dir):
    pid = task["paper_id"]
    url = task["url"]
    method = task["method"]
    pp = pdf_dir / f"{pid}.pdf"

    if is_valid(pp):
        return True

    headers = [
        "-H",
        "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    ]
    if method == "openreview_browser_headers":
        headers += [
            "-H",
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        ]

    try:
        subprocess.run(
            ["curl", "-sL", "-o", str(pp), "--max-time", "180"] + headers + [url],
            capture_output=True,
            timeout=240,
        )
        if is_valid(pp):
            return True
        pp.unlink(missing_ok=True)
        return False
    except Exception:
        pp.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PDFs from task file")
    parser.add_argument(
        "--task",
        default="data/task_neurips_iclr_icml.json",
        help="Path to task JSON file",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--venue", type=str, default=None, help="Filter by venue")
    args = parser.parse_args()

    tasks = json.load(open(args.task))
    if args.venue:
        tasks = [t for t in tasks if t["venue"] == args.venue]

    pdf_dir = Path("data/paper_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Skip already done
    tasks = [t for t in tasks if not is_valid(pdf_dir / f"{t['paper_id']}.pdf")]
    print(f"Downloading {len(tasks)} PDFs with {args.workers} workers...")

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, t, pdf_dir): t["paper_id"] for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            if f.result():
                ok += 1
            else:
                fail += 1
            if (i + 1) % 100 == 0:
                total = sum(1 for p in pdf_dir.glob("*.pdf") if is_valid(p))
                print(f"  [{i+1}/{len(tasks)}] +{ok} ok, {fail} fail | total PDFs: {total}")

    total = sum(1 for p in pdf_dir.glob("*.pdf") if is_valid(p))
    print(f"Done: +{ok} ok, {fail} fail | Total valid PDFs: {total}")


if __name__ == "__main__":
    main()
