#!/usr/bin/env python3
"""Sync data directories with HuggingFace dataset repo (xln3/bamboo-papers).

Syncs: paper_pdfs, paper_markdowns, paper_claims_v2.

Usage:
    python hf_sync.py status                          # show sync status
    python hf_sync.py pull                             # download all missing files
    python hf_sync.py pull --only paper_claims_v2      # only download claims
    python hf_sync.py push                             # upload all local files
    python hf_sync.py push --only paper_claims_v2      # only upload claims
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_tree

REPO_ID = "xln3/bamboo-papers"
BASE = Path(__file__).parent.parent.parent

# All syncable data directories: (remote_prefix, local_dir, suffix, min_size)
SYNC_DIRS = [
    ("paper_pdfs",      BASE / "data" / "paper_pdfs",      ".pdf",  10000),
    ("paper_markdowns", BASE / "data" / "paper_markdowns",  ".md",   500),
    ("paper_claims_v2", BASE / "data" / "paper_claims_v2",  ".json", 10),
]

for _, d, _, _ in SYNC_DIRS:
    d.mkdir(exist_ok=True)


def get_remote_files(api: HfApi, prefix: str) -> set[str]:
    """List files on HF under a prefix."""
    files = set()
    try:
        for item in api.list_repo_tree(REPO_ID, path_prefix=prefix, repo_type="dataset"):
            if hasattr(item, "rfilename"):
                name = item.rfilename.split("/")[-1]
                files.add(name)
    except Exception as e:
        print(f"Warning: could not list {prefix}: {e}")
    return files


def get_local_files(directory: Path, suffix: str) -> set[str]:
    """List local files."""
    return set(f.name for f in directory.glob(f"*{suffix}") if f.stat().st_size > 100)


def cmd_status(args):
    api = HfApi()

    print(f"{'Directory':>20} {'Remote':>8} {'Local':>8} {'Both':>8} {'Remote-only':>12} {'Local-only':>12}")
    print("-" * 72)

    total_pull = 0
    total_push = 0
    for prefix, local_dir, suffix, min_size in SYNC_DIRS:
        remote = get_remote_files(api, prefix)
        local = get_local_files(local_dir, suffix)
        both = remote & local
        remote_only = remote - local
        local_only = local - remote
        total_pull += len(remote_only)
        total_push += len(local_only)
        print(f"{prefix:>20} {len(remote):>8} {len(local):>8} {len(both):>8} {len(remote_only):>12} {len(local_only):>12}")

    print()
    if total_pull:
        print(f"Run 'python hf_sync.py pull' to download {total_pull} files")
    if total_push:
        print(f"Run 'python hf_sync.py push' to upload {total_push} files")


def cmd_pull(args):
    api = HfApi()
    only = set(args.only.split(",")) if args.only else None

    for prefix, local_dir, suffix, _ in SYNC_DIRS:
        if only and prefix not in only:
            continue
        remote = get_remote_files(api, prefix)
        local = get_local_files(local_dir, suffix)
        to_download = remote - local
        if not to_download:
            print(f"{prefix}: up to date")
            continue
        print(f"Pulling {len(to_download)} {prefix}...")
        for i, name in enumerate(sorted(to_download)):
            hf_hub_download(REPO_ID, f"{prefix}/{name}", repo_type="dataset",
                            local_dir=str(BASE / "data"), local_dir_use_symlinks=False)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(to_download)}]", flush=True)
        print(f"Done: {len(to_download)} {prefix} pulled")


def cmd_push(args):
    """Push using upload_large_folder for resilient, resumable uploads."""
    import shutil
    import tempfile

    api = HfApi()
    only = set(args.only.split(",")) if args.only else None

    staging = Path(tempfile.mkdtemp(prefix="hf_push_"))
    try:
        total = 0
        for prefix, local_dir, suffix, min_size in SYNC_DIRS:
            if only and prefix not in only:
                continue
            dst = staging / prefix
            dst.mkdir()
            count = 0
            for f in local_dir.glob(f"*{suffix}"):
                if f.stat().st_size > min_size and not f.name.startswith("_tmp_"):
                    (dst / f.name).symlink_to(f.resolve())
                    count += 1
            print(f"Staging {count} {prefix}...")
            total += count

        if total == 0:
            print("Nothing to push")
            return

        print(f"Uploading {total} files (resumable)...")
        api.upload_large_folder(
            folder_path=str(staging),
            repo_id=REPO_ID,
            repo_type="dataset",
            ignore_patterns=[".*", "_tmp_*", "*_images*", ".cache*"],
            num_workers=8,
        )
        print("Done: push complete")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Sync with HuggingFace dataset")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status")

    pull_p = sub.add_parser("pull")
    pull_p.add_argument("--only", help="Comma-separated prefixes to sync (e.g., paper_claims_v2,paper_markdowns)")

    push_p = sub.add_parser("push")
    push_p.add_argument("--only", help="Comma-separated prefixes to sync (e.g., paper_claims_v2)")

    args = parser.parse_args()
    if args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "pull":
        cmd_pull(args)
    elif args.cmd == "push":
        cmd_push(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
