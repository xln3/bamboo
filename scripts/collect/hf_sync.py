#!/usr/bin/env python3
"""Sync PDFs and MDs with HuggingFace dataset repo.

Supports multi-server coordination:
- Pull: download PDFs/MDs that exist on HF but not locally
- Push: upload local PDFs/MDs that don't exist on HF yet
- Status: show what's done, what's local-only, what's remote-only

Usage:
    python hf_sync.py status              # show sync status
    python hf_sync.py pull                # download missing PDFs/MDs from HF
    python hf_sync.py pull --mds-only     # only download MDs
    python hf_sync.py push                # upload new local PDFs/MDs to HF
    python hf_sync.py push --mds-only     # only upload MDs
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
PDF_DIR = BASE / "data" / "paper_pdfs"
MD_DIR = BASE / "data" / "paper_markdowns"

PDF_DIR.mkdir(exist_ok=True)
MD_DIR.mkdir(exist_ok=True)


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

    remote_pdfs = get_remote_files(api, "paper_pdfs")
    remote_mds = get_remote_files(api, "paper_markdowns")
    local_pdfs = get_local_files(PDF_DIR, ".pdf")
    local_mds = get_local_files(MD_DIR, ".md")

    both_pdfs = remote_pdfs & local_pdfs
    remote_only_pdfs = remote_pdfs - local_pdfs
    local_only_pdfs = local_pdfs - remote_pdfs

    both_mds = remote_mds & local_mds
    remote_only_mds = remote_mds - local_mds
    local_only_mds = local_mds - remote_mds

    print(f"{'':>20} {'Remote':>8} {'Local':>8} {'Both':>8} {'Remote-only':>12} {'Local-only':>12}")
    print("-" * 70)
    print(f"{'PDFs':>20} {len(remote_pdfs):>8} {len(local_pdfs):>8} {len(both_pdfs):>8} {len(remote_only_pdfs):>12} {len(local_only_pdfs):>12}")
    print(f"{'MDs':>20} {len(remote_mds):>8} {len(local_mds):>8} {len(both_mds):>8} {len(remote_only_mds):>12} {len(local_only_mds):>12}")
    print()
    if remote_only_pdfs:
        print(f"Run 'python hf_sync.py pull' to download {len(remote_only_pdfs)} PDFs + {len(remote_only_mds)} MDs")
    if local_only_pdfs or local_only_mds:
        print(f"Run 'python hf_sync.py push' to upload {len(local_only_pdfs)} PDFs + {len(local_only_mds)} MDs")


def cmd_pull(args):
    api = HfApi()

    if not args.mds_only:
        remote_pdfs = get_remote_files(api, "paper_pdfs")
        local_pdfs = get_local_files(PDF_DIR, ".pdf")
        to_download = remote_pdfs - local_pdfs
        print(f"Pulling {len(to_download)} PDFs...")
        for i, name in enumerate(sorted(to_download)):
            hf_hub_download(REPO_ID, f"paper_pdfs/{name}", repo_type="dataset",
                            local_dir=str(BASE / "data"), local_dir_use_symlinks=False)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(to_download)}]", flush=True)
        print(f"Done: {len(to_download)} PDFs pulled")

    remote_mds = get_remote_files(api, "paper_markdowns")
    local_mds = get_local_files(MD_DIR, ".md")
    to_download = remote_mds - local_mds
    print(f"Pulling {len(to_download)} MDs...")
    for i, name in enumerate(sorted(to_download)):
        hf_hub_download(REPO_ID, f"paper_markdowns/{name}", repo_type="dataset",
                        local_dir=str(BASE / "data"), local_dir_use_symlinks=False)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(to_download)}]", flush=True)
    print(f"Done: {len(to_download)} MDs pulled")


def cmd_push(args):
    """Push using upload_large_folder for resilient, resumable uploads."""
    import shutil
    import tempfile

    api = HfApi()

    # Create a staging dir with the right repo structure (symlinks)
    staging = Path(tempfile.mkdtemp(prefix="hf_push_"))
    try:
        if not args.mds_only:
            dst = staging / "paper_pdfs"
            dst.mkdir()
            count = 0
            for f in PDF_DIR.glob("*.pdf"):
                if f.stat().st_size > 10000:
                    (dst / f.name).symlink_to(f.resolve())
                    count += 1
            print(f"Staging {count} PDFs...")

        if not args.pdfs_only:
            dst = staging / "paper_markdowns"
            dst.mkdir()
            count = 0
            for f in MD_DIR.glob("*.md"):
                if f.stat().st_size > 500 and not f.name.startswith("_tmp_"):
                    (dst / f.name).symlink_to(f.resolve())
                    count += 1
            print(f"Staging {count} MDs...")

        print("Uploading (resumable)...")
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
    pull_p.add_argument("--mds-only", action="store_true")

    push_p = sub.add_parser("push")
    push_p.add_argument("--mds-only", action="store_true")
    push_p.add_argument("--pdfs-only", action="store_true")

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
