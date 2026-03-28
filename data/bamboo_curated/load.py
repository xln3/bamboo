"""Load all partitioned bamboo_curated chunk files into a single list.

Usage:
    from data.bamboo_curated.load import load_curated
    papers = load_curated()

Also works as a CLI:
    python -m data.bamboo_curated.load          # prints count
    python -m data.bamboo_curated.load --dump   # dumps merged JSON to stdout
"""
from __future__ import annotations

import json
from pathlib import Path

CURATED_DIR = Path(__file__).parent


def load_curated(directory: Path | str | None = None) -> list[dict]:
    """Load all chunk files from the curated directory, sorted by paper_id."""
    d = Path(directory) if directory else CURATED_DIR
    papers = []
    for f in sorted(d.glob("bamboo-*.json")):
        with open(f) as fh:
            papers.extend(json.load(fh))
    papers.sort(key=lambda p: p["paper_id"])
    return papers


def load_curated_or_file(path: str | Path) -> list[dict]:
    """Load papers from either a single JSON file or a directory of chunks."""
    p = Path(path)
    if p.is_dir():
        return load_curated(p)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    # Maybe it's the old bamboo_curated.json path but now a directory
    if p.suffix == ".json" and p.with_suffix("").is_dir():
        return load_curated(p.with_suffix(""))
    raise FileNotFoundError(f"Not found: {p}")


if __name__ == "__main__":
    import sys
    papers = load_curated()
    if "--dump" in sys.argv:
        json.dump(papers, sys.stdout, indent=2, ensure_ascii=False)
    else:
        print(f"Loaded {len(papers)} papers from {CURATED_DIR}")
        if papers:
            print(f"  First: {papers[0]['paper_id']}")
            print(f"  Last:  {papers[-1]['paper_id']}")
