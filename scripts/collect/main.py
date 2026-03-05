#!/usr/bin/env python3
"""BAMBOO paper collection pipeline.

Usage:
    # Collect from all venues
    python main.py collect --all

    # Collect from specific venue
    python main.py collect --venue iclr2025

    # Find code URLs for collected papers
    python main.py find-code

    # Validate code repositories
    python main.py validate

    # Full pipeline
    python main.py pipeline

    # Show statistics
    python main.py stats
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

from tqdm import tqdm

from config import VENUES, DATA_DIR
from venues import collect_venue
from code_finder import find_code_urls, validate_code_url

log = logging.getLogger("bamboo")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_venue_file(venue_id: str) -> Path:
    return DATA_DIR / f"{venue_id}.json"


def get_merged_file() -> Path:
    return DATA_DIR / "all_papers.json"


def load_papers(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_papers(papers: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(papers)} papers to {path}")


# ---------- Commands ----------

def cmd_collect(args):
    """Collect papers from venues."""
    if args.all:
        venues_to_collect = VENUES
    elif args.venue:
        venues_to_collect = [v for v in VENUES if v.venue_id == args.venue]
        if not venues_to_collect:
            log.error(f"Unknown venue: {args.venue}. Available: {[v.venue_id for v in VENUES]}")
            sys.exit(1)
    else:
        log.error("Specify --all or --venue <venue_id>")
        sys.exit(1)

    for cfg in venues_to_collect:
        venue_file = get_venue_file(cfg.venue_id)

        # Resume support: skip if already collected (use --force to re-collect)
        if venue_file.exists() and not args.force:
            existing = load_papers(venue_file)
            log.info(f"Skipping {cfg.venue_id}: already have {len(existing)} papers (use --force to re-collect)")
            continue

        log.info(f"=== Collecting {cfg.name} {cfg.year} ===")
        papers = collect_venue(cfg)

        if papers:
            # Add collection metadata
            for p in papers:
                p["_collected_at"] = datetime.now(timezone.utc).isoformat()
                p["_venue_id"] = cfg.venue_id

            save_papers(papers, venue_file)
        else:
            log.warning(f"No papers collected for {cfg.venue_id}")


def cmd_find_code(args):
    """Find code URLs for collected papers."""
    all_papers = _load_all_venue_papers()
    if not all_papers:
        log.error("No papers found. Run 'collect' first.")
        sys.exit(1)

    total = len(all_papers)
    found = 0
    already_has = 0

    for paper in tqdm(all_papers, desc="Finding code URLs"):
        if paper.get("code_url") and not args.force:
            already_has += 1
            continue

        code_urls = find_code_urls(paper)
        if code_urls:
            paper["code_url"] = code_urls[0]  # Primary URL
            paper["code_urls_all"] = code_urls  # All found URLs
            found += 1

    log.info(f"Code URLs: {already_has} already had, {found} newly found, {total - already_has - found} still missing")

    # Save back to per-venue files
    _save_back_to_venue_files(all_papers)

    # Also save merged file
    papers_with_code = [p for p in all_papers if p.get("code_url")]
    save_papers(papers_with_code, DATA_DIR / "papers_with_code.json")
    log.info(f"Papers with code: {len(papers_with_code)} / {total}")


def cmd_validate(args):
    """Validate code repositories."""
    papers_file = DATA_DIR / "papers_with_code.json"
    if not papers_file.exists():
        log.error("No papers_with_code.json found. Run 'find-code' first.")
        sys.exit(1)

    papers = load_papers(papers_file)
    valid = 0
    invalid = 0

    for paper in tqdm(papers, desc="Validating repos"):
        if paper.get("_repo_validated") and not args.force:
            if paper.get("_repo_valid"):
                valid += 1
            else:
                invalid += 1
            continue

        url = paper.get("code_url", "")
        if not url:
            continue

        result = validate_code_url(url)
        paper["_repo_validated"] = True
        paper["_repo_valid"] = result.get("valid", False)
        paper["_repo_info"] = result

        if result.get("valid"):
            valid += 1
        else:
            invalid += 1
            log.debug(f"Invalid repo: {url} - {result.get('reason', 'unknown')}")

    save_papers(papers, papers_file)

    # Save validated-only papers
    valid_papers = [p for p in papers if p.get("_repo_valid")]
    save_papers(valid_papers, DATA_DIR / "papers_validated.json")
    log.info(f"Valid repos: {valid}, Invalid: {invalid}")


def cmd_pipeline(args):
    """Run the full collection pipeline."""
    log.info("=== Phase 1: Collect papers from all venues ===")
    args.all = True
    args.force = args.force
    cmd_collect(args)

    log.info("\n=== Phase 2: Find code URLs ===")
    cmd_find_code(args)

    log.info("\n=== Phase 3: Validate repositories ===")
    cmd_validate(args)

    log.info("\n=== Phase 4: Generate statistics ===")
    cmd_stats(args)


def cmd_stats(args):
    """Show collection statistics."""
    print("\n" + "=" * 60)
    print("BAMBOO Paper Collection Statistics")
    print("=" * 60)

    total_all = 0
    total_with_code = 0

    for cfg in VENUES:
        venue_file = get_venue_file(cfg.venue_id)
        if venue_file.exists():
            papers = load_papers(venue_file)
            with_code = sum(1 for p in papers if p.get("code_url"))
            print(f"  {cfg.name:>8} {cfg.year}: {len(papers):>5} papers, {with_code:>5} with code ({with_code/max(len(papers),1)*100:.1f}%)")
            total_all += len(papers)
            total_with_code += with_code
        else:
            print(f"  {cfg.name:>8} {cfg.year}: not collected yet")

    print(f"  {'TOTAL':>13}: {total_all:>5} papers, {total_with_code:>5} with code ({total_with_code/max(total_all,1)*100:.1f}%)")

    # Check for validated papers
    validated_file = DATA_DIR / "papers_validated.json"
    if validated_file.exists():
        validated = load_papers(validated_file)
        print(f"\n  Validated repos: {len(validated)}")

    print("=" * 60)


# ---------- Helpers ----------

def _load_all_venue_papers() -> list[dict]:
    """Load papers from all venue files."""
    all_papers = []
    for cfg in VENUES:
        venue_file = get_venue_file(cfg.venue_id)
        if venue_file.exists():
            papers = load_papers(venue_file)
            all_papers.extend(papers)
    return all_papers


def _save_back_to_venue_files(all_papers: list[dict]):
    """Save papers back to their per-venue files."""
    by_venue: dict[str, list[dict]] = {}
    for p in all_papers:
        vid = p.get("_venue_id", "unknown")
        by_venue.setdefault(vid, []).append(p)

    for venue_id, papers in by_venue.items():
        save_papers(papers, get_venue_file(venue_id))


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="BAMBOO paper collection pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command")

    # collect
    p_collect = sub.add_parser("collect", help="Collect papers from venues")
    p_collect.add_argument("--all", action="store_true", help="Collect from all venues")
    p_collect.add_argument("--venue", type=str, help="Collect from specific venue (e.g., iclr2025)")
    p_collect.add_argument("--force", action="store_true", help="Force re-collection even if data exists")

    # find-code
    p_code = sub.add_parser("find-code", help="Find code URLs for collected papers")
    p_code.add_argument("--force", action="store_true", help="Re-search even if code URL exists")

    # validate
    p_val = sub.add_parser("validate", help="Validate code repositories")
    p_val.add_argument("--force", action="store_true", help="Re-validate even if already validated")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Run full collection pipeline")
    p_pipe.add_argument("--force", action="store_true", help="Force re-run all stages")

    # stats
    sub.add_parser("stats", help="Show collection statistics")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "collect": cmd_collect,
        "find-code": cmd_find_code,
        "validate": cmd_validate,
        "pipeline": cmd_pipeline,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
