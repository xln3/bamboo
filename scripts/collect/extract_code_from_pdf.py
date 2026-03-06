#!/usr/bin/env python3
"""Extract code URLs from paper PDFs (Introduction section only).

Downloads PDF, extracts text from first 2 pages, finds GitHub/GitLab links.
Only checks Introduction to avoid picking up references to other papers' code.

Usage:
    python extract_code_from_pdf.py [venue_id ...]
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger("extract_code_pdf")

GITHUB_RE = re.compile(r"https?://github\.com/[\w\-\.]+/[\w\-\.]+", re.I)
GITLAB_RE = re.compile(r"https?://gitlab\.com/[\w\-\.]+/[\w\-\.]+", re.I)
HF_RE = re.compile(r"https?://huggingface\.co/[\w\-\.]+/[\w\-\.]+", re.I)

CODE_PATTERNS = [GITHUB_RE, GITLAB_RE, HF_RE]

GITHUB_FP = {"github.com/arxiv", "github.com/login", "github.com/features",
             "github.com/about", "github.com/pricing", "github.com/security",
             "github.com/topics", "github.com/orgs", "github.com/settings",
             "github.com/marketplace", "github.com/explore"}


def _download_pdf(url: str, path: str) -> bool:
    """Download PDF via curl."""
    result = subprocess.run(
        ["curl", "-sL", "-o", path, "--max-time", "30", url],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def _extract_intro_text(pdf_path: str, max_pages: int = 2) -> str:
    """Extract text from first N pages using pdftotext."""
    result = subprocess.run(
        ["pdftotext", "-l", str(max_pages), pdf_path, "-"],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout if result.returncode == 0 else ""


def _find_code_url(text: str) -> str:
    """Find first valid code URL in text."""
    for pattern in CODE_PATTERNS:
        for m in pattern.finditer(text):
            url = m.group(0).rstrip("/.)],;:")
            if not any(fp in url.lower() for fp in GITHUB_FP):
                return url
    return ""


def _get_pdf_url(paper: dict) -> str:
    """Determine best PDF URL for a paper."""
    # Direct PDF URL
    if paper.get("pdf_url", "").startswith("http"):
        return paper["pdf_url"]
    # arXiv PDF
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}"
    return ""


def process_venue(venue_id: str):
    """Extract code URLs from PDFs for a venue."""
    path = DATA_DIR / f"{venue_id}.json"
    if not path.exists():
        log.error(f"{venue_id}: JSON not found")
        return

    with open(path) as f:
        papers = json.load(f)

    # Only process papers without code_url that have a PDF source
    to_process = []
    for i, p in enumerate(papers):
        if p.get("code_url"):
            continue
        pdf_url = _get_pdf_url(p)
        if pdf_url:
            to_process.append((i, p, pdf_url))

    already_have = sum(1 for p in papers if p.get("code_url"))
    log.info(f"{venue_id}: {len(papers)} papers, {already_have} have code, "
             f"checking {len(to_process)} PDFs")

    found = 0
    errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (i, paper, pdf_url) in enumerate(to_process):
            pdf_path = f"{tmpdir}/paper.pdf"

            try:
                if not _download_pdf(pdf_url, pdf_path):
                    errors += 1
                    continue

                text = _extract_intro_text(pdf_path, max_pages=2)
                if not text:
                    errors += 1
                    continue

                code_url = _find_code_url(text)
                if code_url:
                    papers[i]["code_url"] = code_url
                    papers[i]["_code_source"] = "pdf_intro"
                    found += 1

            except Exception as e:
                log.debug(f"Error processing {paper.get('title','')[:50]}: {e}")
                errors += 1

            # Rate limit: be nice to PDF hosts
            time.sleep(0.5)

            if (idx + 1) % 100 == 0:
                with open(path, "w") as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                log.info(f"  {idx+1}/{len(to_process)}: +{found} code, {errors} errors")

    # Final save
    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_code = sum(1 for p in papers if p.get("code_url"))
    log.info(f"{venue_id}: +{found} from PDFs ({errors} errors). "
             f"Total: {total_code}/{len(papers)} ({total_code/max(len(papers),1)*100:.1f}%)")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    venues = sys.argv[1:] if len(sys.argv) > 1 else [
        f.stem for f in sorted(DATA_DIR.glob("*.json"))
        if f.stem not in ("all_papers", "papers_with_code", "papers_validated")
    ]

    for v in venues:
        process_venue(v)

    # Summary
    print("\n" + "=" * 60)
    total_all = 0
    total_code = 0
    for f in sorted(DATA_DIR.glob("*.json")):
        if f.stem in ("all_papers", "papers_with_code", "papers_validated"):
            continue
        papers = json.loads(f.read_text())
        n = sum(1 for p in papers if p.get("code_url"))
        print(f"  {f.stem:>12}: {n:>5} code / {len(papers):>5} papers ({n/max(len(papers),1)*100:.1f}%)")
        total_all += len(papers)
        total_code += n
    print(f"  {'TOTAL':>12}: {total_code:>5} code / {total_all:>5} papers ({total_code/max(total_all,1)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
