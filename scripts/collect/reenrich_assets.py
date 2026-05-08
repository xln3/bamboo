#!/usr/bin/env python3
"""Re-run enrich() over already-extracted paper_assets/*.json files.

Use after editing asset_kb.json or tightening HF lookup so previously
unresolved sizes can pick up the new KB entries / HF search hits without
spending another LLM call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect import access_patterns, asset_kb  # noqa: E402
from scripts.collect.extract_assets import apply_url_classification  # noqa: E402

ASSETS_DIR = REPO_ROOT / "data" / "paper_assets"
FINAL_JSON = REPO_ROOT / "data" / "bamboo_final.json"


def _paper_title(paper_id: str, papers_index: dict[str, dict] | None) -> str:
    if papers_index is None:
        return ""
    p = papers_index.get(paper_id)
    if isinstance(p, dict):
        return p.get("title") or ""
    return ""


def reenrich_file(path: Path, *, dry_run: bool = False,
                  papers_index: dict[str, dict] | None = None) -> dict:
    """Re-enrich a single paper_assets/*.json file. Returns a delta summary."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    paper_id = data.get("paper_id") or path.stem
    paper_title = _paper_title(paper_id, papers_index)

    delta = {"datasets": [], "model_weights": [], "other_assets": []}

    for container in ("datasets", "model_weights"):
        for i, a in enumerate(data.get(container, []) or []):
            before = (a.get("size_bytes"), a.get("size_source"), a.get("access_pattern"))
            new = asset_kb.enrich(a, container=container)
            new = apply_url_classification(new, paper_id=paper_id, paper_title=paper_title, container=container)
            after = (new.get("size_bytes"), new.get("size_source"), new.get("access_pattern"))
            data[container][i] = new
            if before != after:
                delta[container].append({
                    "name": new.get("name"),
                    "before": {"size_bytes": before[0], "size_source": before[1], "access_pattern": before[2]},
                    "after":  {"size_bytes": after[0],  "size_source": after[1],  "access_pattern": after[2]},
                })

    # 'other_assets' don't get KB / HF lookup; just normalize size_human + classify.
    for i, a in enumerate(data.get("other_assets", []) or []):
        sb = a.get("size_bytes")
        if not isinstance(sb, int) or sb <= 0:
            a["size_bytes"] = None
            a.setdefault("size_source", "unknown")
        else:
            a.setdefault("size_source", "estimated")
        a["size_human"] = asset_kb.human_size(a.get("size_bytes"))
        a = apply_url_classification(a, paper_id=paper_id, paper_title=paper_title, container="other_assets")
        data["other_assets"][i] = a

    total = sum(
        x["size_bytes"] for x in (
            (data.get("datasets") or []) +
            (data.get("model_weights") or []) +
            (data.get("other_assets") or [])
        ) if isinstance(x.get("size_bytes"), int) and x["size_bytes"] > 0
    )
    data["total_download_size_bytes"] = total or None
    data["total_download_size_human"] = asset_kb.human_size(total or None)

    all_assets = (data.get("datasets") or []) + (data.get("model_weights") or []) + (data.get("other_assets") or [])
    data["manual_blockers"] = sum(1 for a in all_assets if a.get("manual_action_required"))
    data["fully_automatable"] = bool(all_assets) and all(
        a.get("automation_status") in ("automatable", "automatable_with_auth") for a in all_assets
    )

    if not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return delta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paper", action="append",
                   help="paper_id to re-enrich (repeatable). Default: all files in paper_assets/.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print deltas without writing.")
    args = p.parse_args()

    if args.paper:
        paths = [ASSETS_DIR / f"{pid}.json" for pid in args.paper]
    else:
        paths = sorted(ASSETS_DIR.glob("bamboo-*.json"))

    papers_index: dict[str, dict] | None = None
    if FINAL_JSON.exists():
        try:
            with open(FINAL_JSON, encoding="utf-8") as f:
                papers = json.load(f)
            papers_index = {p.get("paper_id"): p for p in papers if isinstance(p, dict)}
        except Exception:
            papers_index = None

    n_changed = 0
    for path in paths:
        if not path.exists():
            print(f"[skip] {path.name} (missing)")
            continue
        delta = reenrich_file(path, dry_run=args.dry_run, papers_index=papers_index)
        changes = sum(len(v) for v in delta.values())
        if changes:
            n_changed += 1
            print(f"[{path.stem}] {changes} change(s):")
            for container, items in delta.items():
                for d in items:
                    print(f"  {container[:5]:5} {d['name']!r:50} "
                          f"{d['before']['size_source']!r:10} -> "
                          f"{d['after']['size_source']!r:10} "
                          f"({d['before']['size_bytes']} -> {d['after']['size_bytes']})")
        else:
            print(f"[{path.stem}] no changes")
        time.sleep(0.05)

    print(f"\n{n_changed}/{len(paths)} files updated.")


if __name__ == "__main__":
    main()
