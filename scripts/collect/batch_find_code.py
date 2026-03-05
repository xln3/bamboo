#!/usr/bin/env python3
"""Batch find code URLs for papers without them.

Runs PapersWithCode API search for all papers missing code URLs.
Progress is saved after each venue file, so it's safe to interrupt and resume.

Usage:
    python batch_find_code.py [--venue iclr2025] [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import ssl

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.ssl_ import create_urllib3_context

from config import DATA_DIR

log = logging.getLogger("batch_find_code")

GITHUB_RE = re.compile(r"https?://github\.com/[\w\-\.]+/[\w\-\.]+", re.I)
RATE_LIMIT_S = 1.0


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

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def find_code_dblp(title: str) -> str:
    """Search DBLP for paper, then check arXiv page for GitHub links."""
    try:
        time.sleep(RATE_LIMIT_S)
        resp = _session.get(
            "https://dblp.org/search/publ/api",
            params={"q": title, "format": "json", "h": 3},
            timeout=15,
        )
        if resp.status_code != 200:
            return ""

        hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return ""

        title_lower = title.lower().strip()
        title_words = set(title_lower.split())

        for hit in hits:
            info = hit.get("info", {})
            h_title = (info.get("title") or "").lower().strip().rstrip(".")
            h_words = set(h_title.split())
            if len(title_words & h_words) < len(title_words) * 0.7:
                continue

            # Check electronic edition URLs for arXiv
            ee = info.get("ee", "")
            urls = ee if isinstance(ee, list) else [ee]
            for url in urls:
                if "arxiv.org" in url:
                    # Found arXiv link, check the page for GitHub links
                    arxiv_id = re.search(r"(\d{4}\.\d{4,5})", url)
                    if arxiv_id:
                        return _check_arxiv_for_code(arxiv_id.group(1))
            break

    except Exception as e:
        log.debug(f"DBLP search failed for '{title[:50]}': {e}")
    return ""


def _check_arxiv_for_code(arxiv_id: str) -> str:
    """Check arXiv abstract page for GitHub links."""
    try:
        time.sleep(0.5)
        resp = _session.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=15)
        if resp.status_code == 200:
            matches = GITHUB_RE.findall(resp.text)
            # Filter false positives
            for m in matches:
                if not any(fp in m.lower() for fp in [
                    "github.com/arxiv", "github.com/login", "github.com/features",
                    "github.com/about", "github.com/pricing",
                ]):
                    return m.rstrip("/.")
    except Exception as e:
        log.debug(f"arXiv check failed for {arxiv_id}: {e}")
    return ""


def find_code_hf(title: str) -> str:
    """Search HuggingFace daily papers for code links."""
    try:
        time.sleep(RATE_LIMIT_S)
        resp = _session.get(
            "https://huggingface.co/api/daily_papers",
            params={"search": title},
            timeout=15,
        )
        if resp.status_code != 200:
            return ""

        results = resp.json()
        if not results:
            return ""

        title_lower = title.lower().strip()
        title_words = set(title_lower.split())

        for r in results[:3]:
            paper = r.get("paper", {})
            r_title = (paper.get("title") or "").lower().strip()
            r_words = set(r_title.split())
            if len(title_words & r_words) >= len(title_words) * 0.7:
                # Check for GitHub in summary
                summary = paper.get("summary", "")
                m = GITHUB_RE.search(summary)
                if m:
                    return m.group(0).rstrip("/.")
                break

    except Exception as e:
        log.debug(f"HuggingFace search failed: {e}")
    return ""


def find_code_for_paper(title: str) -> str:
    """Try multiple sources to find code URL for a paper."""
    # Try DBLP -> arXiv -> GitHub first (most reliable chain)
    url = find_code_dblp(title)
    if url:
        return url

    # Try HuggingFace daily papers
    url = find_code_hf(title)
    if url:
        return url

    return ""


def process_venue(venue_id: str, limit: int = 0):
    """Find code URLs for papers in a venue file."""
    path = DATA_DIR / f"{venue_id}.json"
    if not path.exists():
        log.error(f"File not found: {path}")
        return

    with open(path) as f:
        papers = json.load(f)

    total = len(papers)
    already_has = sum(1 for p in papers if p.get("code_url"))
    to_search = [p for p in papers if not p.get("code_url")]

    if limit > 0:
        to_search = to_search[:limit]

    log.info(f"{venue_id}: {total} papers, {already_has} already have code, searching {len(to_search)}")

    found = 0
    for i, paper in enumerate(tqdm(to_search, desc=f"{venue_id}")):
        url = find_code_for_paper(paper["title"])
        if url:
            paper["code_url"] = url.rstrip("/.")
            paper["_code_source"] = "dblp+arxiv"
            found += 1

        # Save progress every 100 papers
        if (i + 1) % 100 == 0:
            with open(path, "w") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            log.info(f"  Progress: {i+1}/{len(to_search)}, found {found} new code URLs")

    # Final save
    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_with_code = sum(1 for p in papers if p.get("code_url"))
    log.info(f"{venue_id}: done. {found} new code URLs found. Total with code: {total_with_code}/{total}")


def main():
    parser = argparse.ArgumentParser(description="Batch find code URLs via PapersWithCode")
    parser.add_argument("--venue", type=str, help="Specific venue to process")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to search per venue (0=all)")
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
        venue_id = vf.stem
        if venue_id in ("all_papers", "papers_with_code", "papers_validated"):
            continue
        process_venue(venue_id, limit=args.limit)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary after batch find-code")
    print("=" * 60)
    total_all = 0
    total_code = 0
    for vf in sorted(DATA_DIR.glob("*.json")):
        vid = vf.stem
        if vid in ("all_papers", "papers_with_code", "papers_validated"):
            continue
        with open(vf) as f:
            papers = json.load(f)
        with_code = sum(1 for p in papers if p.get("code_url"))
        print(f"  {vid:>12}: {with_code:>4}/{len(papers):>5} with code ({with_code/max(len(papers),1)*100:.1f}%)")
        total_all += len(papers)
        total_code += with_code
    print(f"  {'TOTAL':>12}: {total_code:>4}/{total_all:>5} with code ({total_code/max(total_all,1)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
