#!/usr/bin/env python3
"""Find code URLs for ICRA papers that already have arXiv IDs.

Uses curl to fetch arXiv abstract pages and extract GitHub links.
For papers without arXiv IDs, does arXiv title search.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from config import DATA_DIR

GITHUB_RE = re.compile(r"https?://github\.com/[\w\-\.]+/[\w\-\.]+", re.I)
GITHUB_FP = {"github.com/arxiv", "github.com/login", "github.com/features",
             "github.com/about", "github.com/pricing", "github.com/security"}


def _curl(url: str) -> str:
    """Fetch URL via curl."""
    result = subprocess.run(
        ["curl", "-s", "-L", "--max-time", "15", url],
        capture_output=True, text=True,
    )
    return result.stdout if result.returncode == 0 else ""


def _extract_github(text: str) -> str:
    """Extract first valid GitHub URL from text."""
    for m in GITHUB_RE.finditer(text):
        url = m.group(0).rstrip("/.")
        if not any(fp in url.lower() for fp in GITHUB_FP):
            return url
    return ""


def main():
    path = DATA_DIR / "icra2025.json"
    with open(path) as f:
        papers = json.load(f)

    # Phase 1: Check arXiv pages for papers with arXiv IDs
    with_arxiv = [(i, p) for i, p in enumerate(papers) if p.get("arxiv_id") and not p.get("code_url")]
    print(f"Phase 1: Checking {len(with_arxiv)} arXiv pages for GitHub links...")

    found = 0
    for idx, (i, paper) in enumerate(with_arxiv):
        html = _curl(f"https://arxiv.org/abs/{paper['arxiv_id']}")
        code_url = _extract_github(html)
        if code_url:
            papers[i]["code_url"] = code_url
            papers[i]["_code_source"] = "arxiv_page"
            found += 1

        if (idx + 1) % 50 == 0:
            with open(path, "w") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            print(f"  {idx+1}/{len(with_arxiv)}, found {found} code URLs")

        time.sleep(1)  # be nice to arXiv

    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    print(f"Phase 1 done: found {found} code URLs from {len(with_arxiv)} arXiv pages")

    # Phase 2: arXiv title search for papers without arXiv ID
    without_arxiv = [(i, p) for i, p in enumerate(papers) if not p.get("arxiv_id") and not p.get("code_url")]
    print(f"\nPhase 2: Searching arXiv for {len(without_arxiv)} papers by title...")

    found2_arxiv = 0
    found2_code = 0
    for idx, (i, paper) in enumerate(without_arxiv):
        title = paper["title"]
        search_query = 'ti:"' + title.replace('"', '').replace('\\', '') + '"'
        url = f"http://export.arxiv.org/api/query?search_query={search_query}&max_results=3"
        # URL encode
        import urllib.parse
        url = f"http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": search_query,
            "max_results": 3,
        })

        text = _curl(url)
        if not text:
            time.sleep(3)
            continue

        entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
        title_words = set(title.lower().split())

        for entry in entries:
            entry_title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            if not entry_title_m:
                continue
            entry_title = entry_title_m.group(1).strip().replace("\n", " ")
            entry_words = set(entry_title.lower().split())
            if len(title_words & entry_words) < len(title_words) * 0.6:
                continue

            id_m = re.search(r"<id>http://arxiv\.org/abs/(\d{4}\.\d{4,5})", entry)
            if id_m:
                papers[i]["arxiv_id"] = id_m.group(1)
                found2_arxiv += 1

            code_url = _extract_github(entry)
            if code_url:
                papers[i]["code_url"] = code_url
                papers[i]["_code_source"] = "arxiv"
                found2_code += 1

            if not code_url and papers[i].get("arxiv_id"):
                time.sleep(0.5)
                html = _curl(f"https://arxiv.org/abs/{papers[i]['arxiv_id']}")
                code_url = _extract_github(html)
                if code_url:
                    papers[i]["code_url"] = code_url
                    papers[i]["_code_source"] = "arxiv_page"
                    found2_code += 1

            break

        if (idx + 1) % 50 == 0:
            with open(path, "w") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            print(f"  {idx+1}/{len(without_arxiv)}, +{found2_arxiv} arXiv, +{found2_code} code")

        time.sleep(3)  # arXiv rate limit

    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_code = sum(1 for p in papers if p.get("code_url"))
    total_arxiv = sum(1 for p in papers if p.get("arxiv_id"))
    print(f"\nFinal: {total_arxiv} arXiv, {total_code} code / {len(papers)} papers")


if __name__ == "__main__":
    main()
