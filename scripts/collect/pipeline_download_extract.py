#!/usr/bin/env python3
"""Pipeline: Download PDFs → MinerU hybrid extraction → structured markdown.

Phase 1: Parallel PDF download (64 workers, ~300KB/s per connection via proxy)
Phase 2: MinerU hybrid-auto-engine extraction (GPU, batch_size=8 on 48GB)

Usage:
    python pipeline_download_extract.py                    # full pipeline
    python pipeline_download_extract.py --download-only    # just download
    python pipeline_download_extract.py --mineru-only      # just MinerU
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline")

BASE = Path(__file__).parent.parent.parent
DATA = BASE / "data"
PDF_DIR = DATA / "paper_pdfs"
MD_DIR = DATA / "paper_markdowns"
FINAL_JSON = DATA / "bamboo_final.json"

PDF_DIR.mkdir(exist_ok=True)
MD_DIR.mkdir(exist_ok=True)


def get_pdf_url(paper: dict) -> str:
    """Get PDF download URL for a paper."""
    if paper.get("pdf_url"):
        return paper["pdf_url"]
    arxiv_id = paper.get("arxiv_id", "")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"
    paper_url = paper.get("paper_url", "")
    if "openreview.net" in paper_url:
        if "/forum?" in paper_url:
            return paper_url.replace("/forum?", "/pdf?")
        return paper_url
    if "aclanthology.org" in paper_url:
        if not paper_url.endswith(".pdf"):
            return paper_url.rstrip("/") + ".pdf"
        return paper_url
    return paper_url


def is_valid_pdf(path: Path) -> bool:
    """Check if a PDF file is valid (header + pypdfium2 parse test)."""
    if not path.exists() or path.stat().st_size < 10000:
        return False
    with open(path, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        return False
    # Strict validation: actually try to open with pypdfium2
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(str(path))
        n = len(doc)
        doc.close()
        return n > 0
    except Exception:
        return False


def download_one(paper_id: str, pdf_url: str) -> bool:
    """Download a single PDF."""
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if is_valid_pdf(pdf_path):
        return True  # already done

    try:
        result = subprocess.run(
            ["curl", "-sL", "--noproxy", "*", "-o", str(pdf_path),
             "--max-time", "60", "--retry", "2", "--retry-delay", "3",
             "-H", "User-Agent: Mozilla/5.0 (compatible; BAMBOO/1.0)",
             pdf_url],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and is_valid_pdf(pdf_path):
            return True
        pdf_path.unlink(missing_ok=True)
        return False
    except (subprocess.TimeoutExpired, Exception):
        pdf_path.unlink(missing_ok=True)
        return False


def mineru_one(paper_id: str) -> bool:
    """Run MinerU hybrid-auto-engine on one PDF."""
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    md_path = MD_DIR / f"{paper_id}.md"

    if md_path.exists() and md_path.stat().st_size > 500:
        return True  # already done

    if not is_valid_pdf(pdf_path):
        return False

    tmp_out = MD_DIR / f"_tmp_{paper_id}"
    try:
        env = os.environ.copy()
        env["MINERU_MODEL_SOURCE"] = "modelscope"
        env["MINERU_DEVICE_MODE"] = "cuda"

        result = subprocess.run(
            ["mineru", "-p", str(pdf_path), "-o", str(tmp_out),
             "-b", "hybrid-auto-engine"],
            capture_output=True, text=True, timeout=600, env=env,
        )

        # Find output .md file
        md_files = list(tmp_out.rglob("*.md"))
        if md_files:
            shutil.copy2(md_files[0], md_path)
            # Copy images
            for img_dir in tmp_out.rglob("images"):
                dest = MD_DIR / f"{paper_id}_images"
                if not dest.exists():
                    shutil.copytree(img_dir, dest)
                break
            shutil.rmtree(tmp_out, ignore_errors=True)
            return True

        shutil.rmtree(tmp_out, ignore_errors=True)
        return False
    except (subprocess.TimeoutExpired, Exception) as e:
        shutil.rmtree(tmp_out, ignore_errors=True)
        log.warning(f"MinerU failed for {paper_id}: {e}")
        return False


def phase1_download(papers: list[dict], workers: int = 64, limit: int | None = None):
    """Phase 1: parallel PDF download."""
    to_download = []
    skipped = 0
    for p in papers:
        pid = p["paper_id"]
        url = get_pdf_url(p)
        if not url:
            continue
        if is_valid_pdf(PDF_DIR / f"{pid}.pdf"):
            skipped += 1
            continue
        to_download.append((pid, url))

    if limit:
        to_download = to_download[:limit]

    log.info(f"Phase 1: Download {len(to_download)} PDFs ({skipped} already done), {workers} workers")

    if not to_download:
        return

    success = 0
    fail = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, pid, url): pid for pid, url in to_download}
        for i, future in enumerate(as_completed(futures)):
            try:
                if future.result():
                    success += 1
                else:
                    fail += 1
            except Exception:
                fail += 1

            if (i + 1) % 200 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed * 3600
                n_pdfs = len(list(PDF_DIR.glob("*.pdf")))
                log.info(f"  [{i+1}/{len(to_download)}] +{success} ok, {fail} fail | "
                         f"{rate:.0f}/hr | total PDFs: {n_pdfs}")

    elapsed = time.time() - start
    n_pdfs = sum(1 for f in PDF_DIR.glob("*.pdf") if is_valid_pdf(f))
    log.info(f"Phase 1 done: +{success} downloaded, {fail} failed in {elapsed:.0f}s | Total valid PDFs: {n_pdfs}")


def phase2_mineru(workers: int = 1):
    """Phase 2: MinerU extraction on all PDFs without markdowns."""
    to_process = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        if not is_valid_pdf(pdf):
            continue
        pid = pdf.stem
        md_path = MD_DIR / f"{pid}.md"
        if md_path.exists() and md_path.stat().st_size > 500:
            continue
        to_process.append(pid)

    log.info(f"Phase 2: MinerU extraction for {len(to_process)} papers, {workers} worker(s)")

    if not to_process:
        return

    success = 0
    fail = 0
    start = time.time()

    # Sequential processing (GPU is the bottleneck, MinerU handles batching internally)
    for i, pid in enumerate(to_process):
        if mineru_one(pid):
            success += 1
        else:
            fail += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed * 3600
            n_mds = len(list(MD_DIR.glob("*.md")))
            log.info(f"  [{i+1}/{len(to_process)}] +{success} ok, {fail} fail | "
                     f"{rate:.0f}/hr | total MDs: {n_mds}")

    elapsed = time.time() - start
    n_mds = len(list(MD_DIR.glob("*.md")))
    log.info(f"Phase 2 done: +{success} extracted, {fail} failed in {elapsed:.0f}s | Total MDs: {n_mds}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-workers", type=int, default=64)
    parser.add_argument("--mineru-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--mineru-only", action="store_true")
    args = parser.parse_args()

    with open(FINAL_JSON) as f:
        papers = json.load(f)
    log.info(f"Loaded {len(papers)} papers")

    if not args.mineru_only:
        phase1_download(papers, workers=args.download_workers, limit=args.limit)

    if not args.download_only:
        phase2_mineru(workers=args.mineru_workers)


if __name__ == "__main__":
    main()
