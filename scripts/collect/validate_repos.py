#!/usr/bin/env python3
"""Validate code URLs: check GitHub repos exist, are non-empty, pin commit hash.

Uses GitHub API (with optional token for higher rate limits).
Saves validation results back to venue JSON files.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger("validate_repos")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_API = "https://api.github.com"


def _curl_json(url: str, headers: list[str] | None = None) -> dict | None:
    """Fetch JSON via curl."""
    cmd = ["curl", "-s", "--max-time", "15"]
    if headers:
        for h in headers:
            cmd.extend(["-H", h])
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def validate_repo(url: str) -> dict:
    """Validate a code repo URL using git ls-remote. Returns validation info."""
    result = {
        "valid": False,
        "commit": "",
        "error": "",
    }

    # Normalize URL
    git_url = url.rstrip("/")
    if not git_url.endswith(".git"):
        git_url += ".git"

    try:
        proc = subprocess.run(
            ["git", "ls-remote", "--heads", "--refs", git_url],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.lower()
            if "not found" in stderr or "repository not found" in stderr:
                result["error"] = "not_found"
            elif "could not resolve" in stderr:
                result["error"] = "dns_error"
            else:
                result["error"] = f"git_error: {proc.stderr[:100]}"
            return result

        lines = proc.stdout.strip().split("\n")
        if not lines or not lines[0].strip():
            result["error"] = "empty_repo"
            return result

        # Find HEAD or main/master branch commit
        commit = ""
        for line in lines:
            parts = line.split("\t")
            if len(parts) == 2:
                sha, ref = parts
                if ref in ("refs/heads/main", "refs/heads/master"):
                    commit = sha[:40]
                    break
        # Fallback: use first branch
        if not commit:
            commit = lines[0].split("\t")[0][:40]

        result["valid"] = True
        result["commit"] = commit

    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)[:100]

    return result


def process_venue(venue_id: str):
    """Validate code URLs for a venue."""
    path = DATA_DIR / f"{venue_id}.json"
    if not path.exists():
        return

    with open(path) as f:
        papers = json.load(f)

    to_validate = [
        (i, p) for i, p in enumerate(papers)
        if p.get("code_url") and not p.get("_validated")
    ]

    if not to_validate:
        log.info(f"{venue_id}: nothing to validate")
        return

    log.info(f"{venue_id}: validating {len(to_validate)} repos")

    valid_count = 0
    invalid_count = 0

    for idx, (i, paper) in enumerate(to_validate):
        url = paper["code_url"]

        info = validate_repo(url)

        paper["_validated"] = True
        paper["_repo_valid"] = info["valid"]

        if info["valid"]:
            paper["code_commit"] = info["commit"]
            valid_count += 1
        else:
            paper["_validation_error"] = info["error"]
            invalid_count += 1

        time.sleep(0.5)

        if (idx + 1) % 50 == 0:
            with open(path, "w") as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            log.info(f"  {idx+1}/{len(to_validate)}: {valid_count} valid, {invalid_count} invalid")

    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    log.info(f"{venue_id}: {valid_count} valid, {invalid_count} invalid")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    venues = sys.argv[1:] if len(sys.argv) > 1 else None
    if not venues:
        venues = [f.stem for f in sorted(DATA_DIR.glob("*.json"))
                  if f.stem not in ("all_papers", "papers_with_code", "papers_validated")]

    for v in venues:
        process_venue(v)

    # Summary
    print("\n" + "=" * 60)
    total_valid = 0
    total_invalid = 0
    total_code = 0
    for f in sorted(DATA_DIR.glob("*.json")):
        if f.stem in ("all_papers", "papers_with_code", "papers_validated"):
            continue
        papers = json.loads(f.read_text())
        n_code = sum(1 for p in papers if p.get("code_url"))
        n_valid = sum(1 for p in papers if p.get("_repo_valid") is True)
        n_invalid = sum(1 for p in papers if p.get("_repo_valid") is False)
        if n_code:
            print(f"  {f.stem:>12}: {n_valid:>4} valid, {n_invalid:>4} invalid / {n_code:>4} code")
        total_valid += n_valid
        total_invalid += n_invalid
        total_code += n_code
    print(f"  {'TOTAL':>12}: {total_valid:>4} valid, {total_invalid:>4} invalid / {total_code:>4} code")
    print("=" * 60)


if __name__ == "__main__":
    main()
