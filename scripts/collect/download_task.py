#!/usr/bin/env python3
"""Download PDFs from task file (for second server).

Usage:
    python scripts/collect/download_task.py                          # download all
    python scripts/collect/download_task.py --workers 16             # adjust concurrency
    python scripts/collect/download_task.py --venue ICML             # only ICML
    python scripts/collect/download_task.py --task data/task_neurips_iclr_icml.json
"""
import argparse
import json
import subprocess
import time
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


def download_one(task, pdf_dir, max_retries=3):
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

    for attempt in range(max_retries):
        try:
            subprocess.run(
                ["curl", "-sL", "-o", str(pp), "--max-time", "180",
                 "--retry", "2", "--retry-delay", "3"] + headers + [url],
                capture_output=True,
                timeout=240,
            )
            if is_valid(pp):
                return True
            pp.unlink(missing_ok=True)
        except Exception:
            pp.unlink(missing_ok=True)

        if attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s

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

    # Split by source — arxiv can handle more concurrency than openreview
    arxiv_tasks = [t for t in tasks if t["method"] == "arxiv_direct"
                   and not is_valid(pdf_dir / f"{t['paper_id']}.pdf")]
    or_tasks = [t for t in tasks if t["method"] == "openreview_browser_headers"
                and not is_valid(pdf_dir / f"{t['paper_id']}.pdf")]

    print(f"arxiv: {len(arxiv_tasks)} papers (workers={args.workers})")
    print(f"openreview: {len(or_tasks)} papers (workers={min(args.workers, 4)})")

    # Phase 1: arxiv (full concurrency)
    ok = 0
    fail = 0
    if arxiv_tasks:
        print(f"\n--- Downloading {len(arxiv_tasks)} arxiv PDFs ---")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download_one, t, pdf_dir): t["paper_id"]
                    for t in arxiv_tasks}
            for i, f in enumerate(as_completed(futs)):
                if f.result():
                    ok += 1
                else:
                    fail += 1
                if (i + 1) % 100 == 0:
                    total = sum(1 for p in pdf_dir.glob("*.pdf") if is_valid(p))
                    print(f"  arxiv [{i+1}/{len(arxiv_tasks)}] "
                          f"+{ok} ok, {fail} fail | total PDFs: {total}")
        print(f"  arxiv done: +{ok} ok, {fail} fail")

    # Phase 2: openreview (limited concurrency to avoid 403/rate limit)
    ok2 = 0
    fail2 = 0
    or_workers = min(args.workers, 4)  # max 4 for openreview
    if or_tasks:
        print(f"\n--- Downloading {len(or_tasks)} openreview PDFs (max {or_workers} workers) ---")
        with ThreadPoolExecutor(max_workers=or_workers) as ex:
            futs = {ex.submit(download_one, t, pdf_dir): t["paper_id"]
                    for t in or_tasks}
            for i, f in enumerate(as_completed(futs)):
                if f.result():
                    ok2 += 1
                else:
                    fail2 += 1
                if (i + 1) % 100 == 0:
                    total = sum(1 for p in pdf_dir.glob("*.pdf") if is_valid(p))
                    print(f"  openreview [{i+1}/{len(or_tasks)}] "
                          f"+{ok2} ok, {fail2} fail | total PDFs: {total}")
        print(f"  openreview done: +{ok2} ok, {fail2} fail")

    total = sum(1 for p in pdf_dir.glob("*.pdf") if is_valid(p))
    print(f"\n=== TOTAL: +{ok+ok2} ok, {fail+fail2} fail | Valid PDFs: {total} ===")


if __name__ == "__main__":
    main()
