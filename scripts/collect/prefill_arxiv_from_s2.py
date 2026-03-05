#!/usr/bin/env python3
"""Pre-fill arXiv IDs using Semantic Scholar bulk search.

S2 can return 1000 papers per request with arXiv IDs, much faster
than querying arXiv one-by-one. We match by title fuzzy match.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

from config import DATA_DIR

# Map our venue IDs to S2 venue names
S2_VENUES = {
    "iclr2025": ("ICLR", 2025),
    "icml2025": ("ICML", 2025),
    "neurips2025": ("NeurIPS", 2025),
    "cvpr2025": ("CVPR", 2025),
    "iccv2025": ("ICCV", 2025),
    "acl2025": ("ACL", 2025),
    "emnlp2025": ("EMNLP", 2025),
    "aaai2025": ("AAAI", 2025),
    "icra2025": ("ICRA", 2025),
}

S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def _curl_json(url: str) -> dict:
    result = subprocess.run(
        ["curl", "-s", "--max-time", "30", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def fetch_s2_venue(s2_venue: str, year: int) -> list:
    """Fetch all papers from S2 for a venue+year."""
    all_papers = []
    token = None
    page = 0

    while True:
        url = f"{S2_BULK}?query=&venue={urllib.parse.quote(s2_venue)}&year={year}&fields=title,externalIds&limit=1000"
        if token:
            url += f"&token={token}"

        data = _curl_json(url)
        if not data:
            break

        batch = data.get("data", [])
        all_papers.extend(batch)
        print(f"    Page {page}: {len(batch)} papers (total: {len(all_papers)})", flush=True)

        token = data.get("token")
        if not token or not batch:
            break
        page += 1
        time.sleep(1)

    return all_papers


def normalize_title(t: str) -> str:
    """Normalize title for matching."""
    return " ".join(t.lower().split())


def main():
    venues = sys.argv[1:] if len(sys.argv) > 1 else list(S2_VENUES.keys())

    for venue_id in venues:
        if venue_id not in S2_VENUES:
            continue

        path = DATA_DIR / f"{venue_id}.json"
        if not path.exists():
            continue

        s2_venue, year = S2_VENUES[venue_id]
        print(f"\n{venue_id} ({s2_venue} {year}):", flush=True)

        with open(path) as f:
            papers = json.load(f)

        need_arxiv = sum(1 for p in papers if not p.get("arxiv_id"))
        if need_arxiv == 0:
            print(f"  All {len(papers)} papers already have arXiv IDs, skipping")
            continue

        print(f"  {need_arxiv}/{len(papers)} need arXiv IDs, fetching from S2...", flush=True)

        s2_papers = fetch_s2_venue(s2_venue, year)
        print(f"  Got {len(s2_papers)} papers from S2", flush=True)

        # Build lookup by normalized title
        s2_lookup = {}
        for sp in s2_papers:
            ext = sp.get("externalIds") or {}
            arxiv = ext.get("ArXiv", "")
            if arxiv:
                key = normalize_title(sp.get("title", ""))
                s2_lookup[key] = arxiv

        # Match
        filled = 0
        for p in papers:
            if p.get("arxiv_id"):
                continue
            key = normalize_title(p.get("title", ""))
            if key in s2_lookup:
                p["arxiv_id"] = s2_lookup[key]
                filled += 1

        if filled:
            with open(path, "w") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)

        total_arxiv = sum(1 for p in papers if p.get("arxiv_id"))
        print(f"  Filled {filled} arXiv IDs. Total: {total_arxiv}/{len(papers)}", flush=True)


if __name__ == "__main__":
    main()
