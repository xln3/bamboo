#!/usr/bin/env python3
"""Bidirectional sync between JSON (machine) and TSV (human-editable).

Usage:
    python sync_curated.py export          # JSON → TSV (all venues)
    python sync_curated.py export cvpr2025 # JSON → TSV (one venue)
    python sync_curated.py import          # TSV → JSON (merge human edits)
    python sync_curated.py import cvpr2025 # TSV → JSON (one venue)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from config import DATA_DIR

CURATED_DIR = DATA_DIR.parent / "curated"
CURATED_DIR.mkdir(parents=True, exist_ok=True)

# Columns visible in TSV (order matters for readability)
TSV_COLUMNS = [
    "title",
    "venue",
    "code_url",
    "arxiv_id",
    "paper_url",
    "code_commit",
    "status",       # auto | verified | excluded | needs_review
    "notes",        # free-form human notes
    "repo_valid",   # True/False/empty
    "stars",
    "domain",
    "venue_track",
]

# Fields that humans can edit (changes merged back to JSON)
HUMAN_EDITABLE = {"code_url", "arxiv_id", "status", "notes", "domain"}


def export_venue(venue_id: str):
    """Export a venue JSON to TSV."""
    json_path = DATA_DIR / f"{venue_id}.json"
    if not json_path.exists():
        print(f"  {venue_id}: JSON not found, skipping")
        return

    tsv_path = CURATED_DIR / f"{venue_id}.tsv"

    with open(json_path) as f:
        papers = json.load(f)

    # If TSV already exists, load human edits to preserve them
    existing_edits = {}
    if tsv_path.exists():
        existing_edits = _load_tsv_edits(tsv_path)

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t",
                                extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for p in papers:
            row = {
                "title": p.get("title", ""),
                "venue": p.get("venue", ""),
                "code_url": p.get("code_url", ""),
                "arxiv_id": p.get("arxiv_id", ""),
                "paper_url": p.get("paper_url", ""),
                "code_commit": p.get("code_commit", ""),
                "status": p.get("_status", "auto"),
                "notes": p.get("_notes", ""),
                "repo_valid": str(p.get("_repo_valid", "")) if p.get("_validated") else "",
                "stars": str(p.get("_stars", "")),
                "domain": p.get("domain", ""),
                "venue_track": p.get("venue_track", "main"),
            }

            # Preserve human edits from existing TSV
            key = row["title"]
            if key in existing_edits:
                for field in HUMAN_EDITABLE:
                    if field in existing_edits[key]:
                        row[field] = existing_edits[key][field]

            writer.writerow(row)

    n_code = sum(1 for p in papers if p.get("code_url"))
    print(f"  {venue_id}: exported {len(papers)} papers ({n_code} with code) → {tsv_path.name}")


def import_venue(venue_id: str):
    """Import human edits from TSV back to JSON."""
    json_path = DATA_DIR / f"{venue_id}.json"
    tsv_path = CURATED_DIR / f"{venue_id}.tsv"

    if not tsv_path.exists():
        print(f"  {venue_id}: TSV not found, skipping")
        return
    if not json_path.exists():
        print(f"  {venue_id}: JSON not found, skipping")
        return

    with open(json_path) as f:
        papers = json.load(f)

    edits = _load_tsv_edits(tsv_path)

    changes = 0
    for p in papers:
        key = p.get("title", "")
        if key not in edits:
            continue

        edit = edits[key]

        # Merge editable fields
        if edit.get("code_url") and edit["code_url"] != p.get("code_url", ""):
            p["code_url"] = edit["code_url"]
            p["_code_source"] = "human"
            changes += 1

        if edit.get("arxiv_id") and edit["arxiv_id"] != p.get("arxiv_id", ""):
            p["arxiv_id"] = edit["arxiv_id"]
            changes += 1

        if edit.get("status") and edit["status"] != "auto":
            p["_status"] = edit["status"]
            changes += 1

        if edit.get("notes"):
            p["_notes"] = edit["notes"]
            changes += 1

        if edit.get("domain"):
            p["domain"] = edit["domain"]
            changes += 1

    if changes:
        with open(json_path, "w") as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"  {venue_id}: imported {changes} changes from TSV")


def _load_tsv_edits(tsv_path: Path) -> dict:
    """Load TSV and return {title: {field: value}} for editable fields."""
    edits = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = row.get("title", "")
            if not key:
                continue
            edits[key] = {field: row.get(field, "") for field in HUMAN_EDITABLE}
    return edits


def get_venue_ids() -> list:
    """Get all venue IDs from JSON files."""
    return sorted(
        f.stem for f in DATA_DIR.glob("*.json")
        if f.stem not in ("all_papers", "papers_with_code", "papers_validated")
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python sync_curated.py [export|import] [venue_id]")
        sys.exit(1)

    action = sys.argv[1]
    venues = [sys.argv[2]] if len(sys.argv) > 2 else get_venue_ids()

    if action == "export":
        print("Exporting JSON → TSV:")
        for v in venues:
            export_venue(v)
    elif action == "import":
        print("Importing TSV → JSON:")
        for v in venues:
            import_venue(v)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
