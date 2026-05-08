#!/usr/bin/env python3
"""Sync data/paper_assets/*.json into bamboo_final.json[i].required_assets.

Useful when extract_assets.py was killed before its 10-paper flush, or
after a sharded parallel run that used --no-final-write.

Reads every paper_assets/*.json, builds the same inline_summary the main
extractor would write, and replaces the matching entry in bamboo_final.json.
Also clears _assets_error for any paper that now has assets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
FINAL_JSON = REPO_ROOT / "data" / "bamboo_final.json"
PAPER_ASSETS = REPO_ROOT / "data" / "paper_assets"


def inline_summary(assets: dict) -> dict:
    return {
        "datasets": assets["datasets"],
        "model_weights": assets["model_weights"],
        "other_assets": assets["other_assets"],
        "total_download_size_bytes": assets["total_download_size_bytes"],
        "total_download_size_human": assets["total_download_size_human"],
        "extraction_method": assets["extraction_method"],
        "extraction_model": assets["extraction_model"],
        "extracted_at": assets["extracted_at"],
        "fully_automatable": assets.get("fully_automatable"),
        "manual_blockers": assets.get("manual_blockers"),
    }


def main() -> int:
    if not FINAL_JSON.exists():
        print(f"missing {FINAL_JSON}", file=sys.stderr)
        return 2

    with open(FINAL_JSON, encoding="utf-8") as f:
        papers = json.load(f)
    index = {p.get("paper_id"): i for i, p in enumerate(papers)}

    n_synced = 0
    n_cleared = 0
    n_missing = 0
    for path in sorted(PAPER_ASSETS.glob("bamboo-*.json")):
        with open(path, encoding="utf-8") as f:
            assets = json.load(f)
        pid = assets.get("paper_id")
        idx = index.get(pid)
        if idx is None:
            n_missing += 1
            continue

        # Strip the 'paper_id' wrapper so inline_summary sees the same shape.
        stripped = {k: v for k, v in assets.items() if k != "paper_id"}
        try:
            new_inline = inline_summary(stripped)
        except KeyError as e:
            print(f"[skip] {path.name}: missing key {e}", file=sys.stderr)
            continue

        # Compare current inline (if any) — only count as synced if changed.
        current = papers[idx].get("required_assets")
        if current != new_inline:
            papers[idx]["required_assets"] = new_inline
            n_synced += 1
        if papers[idx].pop("_assets_error", None) is not None:
            n_cleared += 1

    tmp = FINAL_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    tmp.replace(FINAL_JSON)

    print(f"synced {n_synced} required_assets entries; "
          f"cleared {n_cleared} _assets_error flags; "
          f"{n_missing} paper_ids absent from bamboo_final.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
