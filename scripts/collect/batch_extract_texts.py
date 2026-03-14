#!/usr/bin/env python3
"""Batch download PDFs and extract text for claim extraction.

Downloads paper PDFs and extracts text using pdftotext (fast) to a cache
directory. The extracted text files can then be read for claim extraction.

Usage:
    python batch_extract_texts.py [--limit N] [--offset N] [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path

log = logging.getLogger("batch_extract")

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "bamboo_final.json"
TEXT_DIR = Path(__file__).parent.parent.parent / "data" / "paper_texts"


def get_pdf_url(paper: dict) -> str:
    if paper.get("pdf_url", "").startswith("http"):
        return paper["pdf_url"]
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}"
    return ""


def download_and_extract(paper_id: str, pdf_url: str, output_path: str) -> bool:
    """Download PDF and extract text. Returns True on success."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 200:
        return True  # Already cached

    pdf_tmp = f"/tmp/bamboo_dl_{paper_id}.pdf"
    try:
        # Download with curl (use system proxy if set — often faster via CDN)
        result = subprocess.run(
            ["curl", "-sL", "-o", pdf_tmp,
             "--max-time", "120", "--retry", "2", "--retry-delay", "3",
             "-H", "User-Agent: Mozilla/5.0 (compatible; BAMBOO/1.0)",
             pdf_url],
            capture_output=True, text=True,
            timeout=180,
        )
        if result.returncode != 0 or not os.path.exists(pdf_tmp):
            return False
        if os.path.getsize(pdf_tmp) < 1000:
            return False

        # Extract text
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_tmp, output_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return False

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 200:
            try:
                os.remove(output_path)
            except OSError:
                pass
            return False

        return True
    finally:
        try:
            os.remove(pdf_tmp)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    with open(DATA_PATH) as f:
        papers = json.load(f)

    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Select papers that need text extraction
    to_process = []
    for p in papers:
        paper_id = p["paper_id"]
        text_path = TEXT_DIR / f"{paper_id}.txt"
        if text_path.exists() and text_path.stat().st_size > 200:
            continue
        pdf_url = get_pdf_url(p)
        if not pdf_url:
            continue
        to_process.append((paper_id, pdf_url, str(text_path)))

    to_process = to_process[args.offset:]
    if args.limit:
        to_process = to_process[:args.limit]

    already = sum(1 for p in papers
                  if (TEXT_DIR / f"{p['paper_id']}.txt").exists()
                  and (TEXT_DIR / f"{p['paper_id']}.txt").stat().st_size > 200)
    log.info(f"Text cache: {already} already done, processing {len(to_process)}")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for paper_id, pdf_url, text_path in to_process:
            f = executor.submit(download_and_extract, paper_id, pdf_url, text_path)
            futures[f] = paper_id

        for f in as_completed(futures):
            paper_id = futures[f]
            try:
                if f.result():
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                log.warning(f"{paper_id}: {e}")
                failed += 1

            total = success + failed
            if total % 100 == 0:
                log.info(f"  Progress: {success} success, {failed} failed / {len(to_process)}")

    total_cached = sum(1 for p in papers
                       if (TEXT_DIR / f"{p['paper_id']}.txt").exists()
                       and (TEXT_DIR / f"{p['paper_id']}.txt").stat().st_size > 200)
    print(f"\n{'='*60}")
    print(f"Batch text extraction: +{success} new, {failed} failed")
    print(f"Total cached: {total_cached}/{len(papers)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
