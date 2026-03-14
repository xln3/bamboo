#!/usr/bin/env python3
"""Fetch missing abstracts from arXiv API.

For papers that have arxiv_id but no abstract, queries the arXiv API
to get the abstract text. Much faster than downloading PDFs + MinerU.

Usage:
    python fetch_arxiv_abstracts.py [--limit N] [--batch-size 50]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

log = logging.getLogger("fetch_arxiv_abstracts")

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "bamboo_final.json"

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def fetch_arxiv_batch(arxiv_ids: list[str]) -> dict[str, str]:
    """Fetch abstracts for a batch of arXiv IDs.

    Args:
        arxiv_ids: List of arXiv IDs (e.g., ["2502.05107", "2501.12345"]).

    Returns:
        Dict mapping arxiv_id → abstract text.
    """
    if not arxiv_ids:
        return {}

    # arXiv API supports up to ~200 IDs per request
    id_list = ",".join(arxiv_ids)
    url = f"{ARXIV_API}?id_list={id_list}&max_results={len(arxiv_ids)}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read().decode("utf-8")
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            log.warning(f"arXiv API error (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                return {}
        except Exception as e:
            log.warning(f"arXiv fetch error: {e}")
            return {}

    # Parse XML response
    results = {}
    try:
        root = ET.fromstring(xml_data)
        for entry in root.findall(f"{ATOM_NS}entry"):
            # Extract arXiv ID from the entry ID URL
            id_elem = entry.find(f"{ATOM_NS}id")
            if id_elem is None:
                continue
            entry_id = id_elem.text.strip()
            # Extract just the ID from URL like "http://arxiv.org/abs/2502.05107v1"
            m = re.search(r"(\d{4}\.\d{4,5})", entry_id)
            if not m:
                continue
            arxiv_id = m.group(1)

            # Get abstract (called "summary" in Atom)
            summary_elem = entry.find(f"{ATOM_NS}summary")
            if summary_elem is not None and summary_elem.text:
                abstract = summary_elem.text.strip()
                # Clean up whitespace
                abstract = re.sub(r"\s+", " ", abstract)
                if len(abstract) > 50:
                    results[arxiv_id] = abstract

    except ET.ParseError as e:
        log.warning(f"XML parse error: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch missing abstracts from arXiv API",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of IDs per arXiv API request")
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

    # Find papers missing abstracts that have arxiv_id
    missing = []
    for i, p in enumerate(papers):
        if p.get("abstract"):
            continue
        if p.get("arxiv_id"):
            missing.append((i, p))

    if args.limit:
        missing = missing[:args.limit]

    already_have = sum(1 for p in papers if p.get("abstract"))
    log.info(f"Abstracts: {already_have}/{len(papers)} present, "
             f"fetching {len(missing)} from arXiv API")

    if not missing:
        log.info("Nothing to fetch")
        return

    # Process in batches
    success = 0
    failed = 0
    batch_size = args.batch_size

    for batch_start in range(0, len(missing), batch_size):
        batch = missing[batch_start:batch_start + batch_size]
        arxiv_ids = [p["arxiv_id"] for _, p in batch]

        log.info(f"  Fetching batch {batch_start // batch_size + 1} "
                 f"({len(batch)} IDs)...")

        results = fetch_arxiv_batch(arxiv_ids)

        for i, paper in batch:
            arxiv_id = paper["arxiv_id"]
            if arxiv_id in results:
                papers[i]["abstract"] = results[arxiv_id]
                success += 1
            else:
                failed += 1

        # Rate limit: arXiv requests 3 second delay
        time.sleep(3.0)

        # Save progress every 5 batches
        if (batch_start // batch_size + 1) % 5 == 0:
            with open(input_path, "w") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            log.info(f"  Progress saved: +{success} abstracts, "
                     f"{failed} not found")

    # Final save
    with open(input_path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_abstracts = sum(1 for p in papers if p.get("abstract"))
    print(f"\n{'='*60}")
    print(f"arXiv abstract fetch complete:")
    print(f"  Fetched: {success}")
    print(f"  Not found: {failed}")
    print(f"  Total abstracts: {total_abstracts}/{len(papers)} "
          f"({100 * total_abstracts / max(len(papers), 1):.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
