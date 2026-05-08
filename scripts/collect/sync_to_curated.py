#!/usr/bin/env python3
"""Fan out required_assets from bamboo_final.json into bamboo_curated/* chunks.

bamboo_final.json is the single source of truth for inline required_assets
(populated by sync_to_final.py from data/paper_assets/*). The 100-paper
curated chunks are partitioned views of final.json; this script copies each
paper's `required_assets` from final into the matching chunk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
FINAL_JSON = REPO_ROOT / "data" / "bamboo_final.json"
CURATED_DIR = REPO_ROOT / "data" / "bamboo_curated"


def main() -> int:
    if not FINAL_JSON.exists():
        print(f"missing {FINAL_JSON}", file=sys.stderr)
        return 2

    with open(FINAL_JSON, encoding="utf-8") as f:
        finals = json.load(f)
    by_id = {p["paper_id"]: p for p in finals}

    n_chunks_changed = 0
    n_papers_changed = 0
    n_papers_total = 0
    for chunk_path in sorted(CURATED_DIR.glob("bamboo-*_to_bamboo-*.json")):
        with open(chunk_path, encoding="utf-8") as f:
            papers = json.load(f)
        chunk_changed = False
        for p in papers:
            n_papers_total += 1
            ref = by_id.get(p.get("paper_id"))
            if ref is None:
                continue
            new_ra = ref.get("required_assets")
            if p.get("required_assets") != new_ra:
                p["required_assets"] = new_ra
                n_papers_changed += 1
                chunk_changed = True
        if chunk_changed:
            tmp = chunk_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            tmp.replace(chunk_path)
            n_chunks_changed += 1

    print(f"updated {n_chunks_changed} chunks; "
          f"changed required_assets on {n_papers_changed}/{n_papers_total} papers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
