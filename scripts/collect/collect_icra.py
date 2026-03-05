#!/usr/bin/env python3
"""Collect ICRA 2025 papers via Semantic Scholar bulk search API.

Uses subprocess+curl to bypass Python 3.8 SSL issues.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from config import DATA_DIR

S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,venue,year,externalIds,openAccessPdf,url"


def _curl_json(url: str) -> dict:
    """Fetch JSON via curl subprocess."""
    result = subprocess.run(
        ["curl", "-s", "--max-time", "30", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    return json.loads(result.stdout)


def collect_icra():
    out_path = DATA_DIR / "icra2025.json"
    papers = []
    token = None
    page = 0

    while True:
        url = f"{S2_BULK}?query=&venue=ICRA&year=2025&fields={FIELDS}&limit=1000"
        if token:
            url += f"&token={token}"

        print(f"  Page {page}: fetching...", end=" ", flush=True)
        data = _curl_json(url)

        batch = data.get("data", [])
        print(f"got {len(batch)} papers")

        for p in batch:
            entry = {
                "title": p.get("title", ""),
                "venue": "ICRA",
                "year": 2025,
            }
            # Extract arXiv ID
            ext = p.get("externalIds") or {}
            if ext.get("ArXiv"):
                entry["arxiv_id"] = ext["ArXiv"]
            if ext.get("DOI"):
                entry["doi"] = ext["DOI"]

            # Paper URL
            if p.get("openAccessPdf", {}) and p["openAccessPdf"].get("url"):
                entry["paper_url"] = p["openAccessPdf"]["url"]
            elif entry.get("arxiv_id"):
                entry["paper_url"] = f"https://arxiv.org/abs/{entry['arxiv_id']}"

            papers.append(entry)

        token = data.get("token")
        if not token or not batch:
            break

        page += 1
        time.sleep(1)  # rate limit

    # Deduplicate by title
    seen = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    print(f"\nTotal: {len(unique)} unique ICRA 2025 papers")

    # Save
    with open(out_path, "w") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

    return unique


if __name__ == "__main__":
    collect_icra()
