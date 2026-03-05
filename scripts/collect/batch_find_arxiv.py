#!/usr/bin/env python3
"""Batch find arXiv IDs for papers using arXiv search API.

For papers that already have arXiv IDs (from OpenReview abstracts), skip.
For others, search by exact title match.

Usage:
    python batch_find_arxiv.py [--venue cvpr2025] [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import ssl
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.ssl_ import create_urllib3_context

from config import DATA_DIR

log = logging.getLogger("batch_find_arxiv")

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class _SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

_session = requests.Session()
_session.mount("https://", _SSLAdapter())
_session.verify = False

GITHUB_RE = re.compile(r"https?://github\.com/[\w\-\.]+/[\w\-\.]+", re.I)
ARXIV_API = "http://export.arxiv.org/api/query"

# GitHub false positive patterns
GITHUB_FP = {"github.com/arxiv", "github.com/login", "github.com/features",
             "github.com/about", "github.com/pricing", "github.com/security"}


def find_arxiv_and_code(title: str) -> tuple:
    """Search arXiv for paper, return (arxiv_id, code_url)."""
    arxiv_id = ""
    code_url = ""

    try:
        # arXiv search by exact title
        search_query = 'ti:"' + title.replace('"', '').replace('\\', '') + '"'
        resp = _session.get(
            ARXIV_API,
            params={"search_query": search_query, "max_results": 3},
            timeout=15,
        )
        if resp.status_code != 200:
            return arxiv_id, code_url

        # Parse XML response
        text = resp.text

        # Find entries
        entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
        title_lower = title.lower().strip()
        title_words = set(title_lower.split())

        for entry in entries:
            # Extract title from entry
            entry_title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            if not entry_title_m:
                continue
            entry_title = entry_title_m.group(1).strip().replace("\n", " ")
            entry_words = set(entry_title.lower().split())

            # Check title match (≥70% word overlap)
            if len(title_words & entry_words) < len(title_words) * 0.6:
                continue

            # Extract arXiv ID
            id_m = re.search(r"<id>http://arxiv\.org/abs/(\d{4}\.\d{4,5})", entry)
            if id_m:
                arxiv_id = id_m.group(1)

            # Check for GitHub link in entry (abstract, comments, etc.)
            for m in GITHUB_RE.finditer(entry):
                url = m.group(0).rstrip("/.")
                if not any(fp in url.lower() for fp in GITHUB_FP):
                    code_url = url
                    break

            # If found arXiv ID, also check the abstract page for GitHub links
            if arxiv_id and not code_url:
                time.sleep(0.3)
                try:
                    page_resp = _session.get(
                        f"https://arxiv.org/abs/{arxiv_id}", timeout=15
                    )
                    if page_resp.status_code == 200:
                        for m in GITHUB_RE.finditer(page_resp.text):
                            url = m.group(0).rstrip("/.")
                            if not any(fp in url.lower() for fp in GITHUB_FP):
                                code_url = url
                                break
                except Exception:
                    pass

            break  # Use first matching entry

    except Exception as e:
        log.debug(f"arXiv search failed for '{title[:50]}': {e}")

    return arxiv_id, code_url


def process_venue(venue_id: str, limit: int = 0):
    """Find arXiv IDs and code URLs for papers in a venue file."""
    path = DATA_DIR / f"{venue_id}.json"
    if not path.exists():
        log.error(f"File not found: {path}")
        return

    with open(path) as f:
        papers = json.load(f)

    # Only search papers without both arxiv_id and code_url
    to_search = [p for p in papers if not p.get("arxiv_id") or not p.get("code_url")]
    already_complete = len(papers) - len(to_search)

    if limit > 0:
        to_search = to_search[:limit]

    log.info(
        f"{venue_id}: {len(papers)} papers, {already_complete} already complete, "
        f"searching {len(to_search)}"
    )

    found_arxiv = 0
    found_code = 0

    for i, paper in enumerate(tqdm(to_search, desc=f"{venue_id}")):
        # Skip if we already have both
        if paper.get("arxiv_id") and paper.get("code_url"):
            continue

        # Fast path: if we already have arXiv ID, just check the page for code
        if paper.get("arxiv_id") and not paper.get("code_url"):
            try:
                page_resp = _session.get(
                    f"https://arxiv.org/abs/{paper['arxiv_id']}", timeout=15
                )
                if page_resp.status_code == 200:
                    for m in GITHUB_RE.finditer(page_resp.text):
                        url = m.group(0).rstrip("/.")
                        if not any(fp in url.lower() for fp in GITHUB_FP):
                            paper["code_url"] = url
                            paper["_code_source"] = "arxiv_page"
                            found_code += 1
                            break
            except Exception:
                pass
            time.sleep(1)
        else:
            # Full search: find arXiv ID + code URL by title
            arxiv_id, code_url = find_arxiv_and_code(paper["title"])

            if arxiv_id and not paper.get("arxiv_id"):
                paper["arxiv_id"] = arxiv_id
                found_arxiv += 1

            if code_url and not paper.get("code_url"):
                paper["code_url"] = code_url
                paper["_code_source"] = "arxiv"
                found_code += 1

            # arXiv rate limit: 1 request per 3 seconds recommended
            time.sleep(3)

        # Save progress every 50 papers
        if (i + 1) % 50 == 0:
            with open(path, "w") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            log.info(f"  Progress: {i+1}/{len(to_search)}, +{found_arxiv} arXiv, +{found_code} code")

    # Final save
    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_arxiv = sum(1 for p in papers if p.get("arxiv_id"))
    total_code = sum(1 for p in papers if p.get("code_url"))
    log.info(
        f"{venue_id}: done. +{found_arxiv} arXiv IDs, +{found_code} code URLs. "
        f"Total: {total_arxiv} arXiv, {total_code} code / {len(papers)} papers"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", type=str, help="Specific venue")
    parser.add_argument("--limit", type=int, default=0, help="Max papers per venue")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    venue_files = sorted(DATA_DIR.glob("*.json"))
    if args.venue:
        venue_files = [DATA_DIR / f"{args.venue}.json"]

    for vf in venue_files:
        vid = vf.stem
        if vid in ("all_papers", "papers_with_code", "papers_validated"):
            continue
        process_venue(vid, limit=args.limit)

    # Print summary
    print("\n" + "=" * 60)
    total_all = 0
    total_arxiv = 0
    total_code = 0
    for vf in sorted(DATA_DIR.glob("*.json")):
        vid = vf.stem
        if vid in ("all_papers", "papers_with_code", "papers_validated"):
            continue
        with open(vf) as f:
            papers = json.load(f)
        n_arxiv = sum(1 for p in papers if p.get("arxiv_id"))
        n_code = sum(1 for p in papers if p.get("code_url"))
        print(f"  {vid:>12}: {n_arxiv:>4} arXiv, {n_code:>4} code / {len(papers):>5} papers")
        total_all += len(papers)
        total_arxiv += n_arxiv
        total_code += n_code
    print(f"  {'TOTAL':>12}: {total_arxiv:>4} arXiv, {total_code:>4} code / {total_all:>5} papers")
    print("=" * 60)


if __name__ == "__main__":
    main()
