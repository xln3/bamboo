#!/usr/bin/env python3
"""Extract missing abstracts from paper PDFs using MinerU.

Downloads PDFs for papers missing abstracts and uses MinerU to extract
the abstract section. Updates bamboo_final.json in place.

Usage:
    python extract_abstracts.py [--limit N]
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
from pathlib import Path

from pdf_extractor import extract_text_mineru, extract_abstract

log = logging.getLogger("extract_abstracts")

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "bamboo_final.json"


def get_pdf_url(paper: dict) -> str:
    """Determine best PDF URL for a paper."""
    if paper.get("pdf_url", "").startswith("http"):
        return paper["pdf_url"]
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}"
    return ""


def download_pdf(url: str, path: str) -> bool:
    """Download PDF via curl. Returns True on success."""
    result = subprocess.run(
        ["curl", "-sL", "-o", path, "--max-time", "60", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    try:
        return os.path.getsize(path) > 1000
    except OSError:
        return False


def process_papers(papers: list[dict], limit: int | None = None) -> int:
    """Extract abstracts for papers missing them.

    Returns number of abstracts successfully extracted.
    """
    to_process = []
    for i, p in enumerate(papers):
        if p.get("abstract"):
            continue
        pdf_url = get_pdf_url(p)
        if pdf_url:
            to_process.append((i, p, pdf_url))

    if limit is not None:
        to_process = to_process[:limit]

    already_have = sum(1 for p in papers if p.get("abstract"))
    log.info(f"Abstracts: {already_have}/{len(papers)} present, "
             f"processing {len(to_process)} papers")

    if not to_process:
        return 0

    success = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (i, paper, pdf_url) in enumerate(to_process):
            title = paper.get("title", "Unknown")
            log.info(f"[{idx + 1}/{len(to_process)}] {title[:70]}")

            pdf_path = os.path.join(tmpdir, "paper.pdf")

            try:
                if not download_pdf(pdf_url, pdf_path):
                    log.warning(f"  PDF download failed")
                    failed += 1
                    continue

                # Use MinerU to extract text
                text = extract_text_mineru(pdf_path)
                if not text:
                    log.warning(f"  Text extraction failed")
                    failed += 1
                    continue

                # Extract abstract from the text
                abstract = extract_abstract(text)
                if abstract and len(abstract) > 50:
                    papers[i]["abstract"] = abstract
                    success += 1
                    log.info(f"  Extracted abstract ({len(abstract)} chars)")
                else:
                    log.warning(f"  Could not find abstract in text")
                    failed += 1

            except Exception as e:
                log.error(f"  Error: {e}")
                failed += 1

            # Clean up PDF for next iteration
            try:
                os.remove(pdf_path)
            except OSError:
                pass

            # Rate limit: be nice to PDF hosts
            time.sleep(1.0)

            # Save progress every 20 papers
            if (idx + 1) % 20 == 0:
                with open(DATA_PATH, "w") as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                log.info(f"  Progress saved. +{success} abstracts, "
                         f"{failed} failed ({idx + 1}/{len(to_process)})")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Extract missing abstracts from paper PDFs using MinerU",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--input", type=str, default=str(DATA_PATH),
                        help="Input JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    input_path = Path(args.input)
    with open(input_path) as f:
        papers = json.load(f)

    log.info(f"Loaded {len(papers)} papers from {input_path}")

    success = process_papers(papers, limit=args.limit)

    # Final save
    with open(input_path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_abstracts = sum(1 for p in papers if p.get("abstract"))
    print(f"\n{'='*60}")
    print(f"Abstract extraction complete: +{success} new abstracts")
    print(f"Total: {total_abstracts}/{len(papers)} "
          f"({total_abstracts / max(len(papers), 1) * 100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
