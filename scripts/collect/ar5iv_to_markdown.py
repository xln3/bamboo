#!/usr/bin/env python3
"""Convert arXiv papers to markdown via ar5iv HTML.

Preserves tables, images (with full URLs), and formulas (LaTeX from alttext).

Usage:
    python ar5iv_to_markdown.py                    # convert all missing papers with arXiv IDs
    python ar5iv_to_markdown.py --ids bamboo-00217 bamboo-00317
    python ar5iv_to_markdown.py --workers 4        # parallel downloads
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

BASE = Path(__file__).parent.parent.parent
DATA = BASE / "data"
MD_DIR = DATA / "paper_markdowns"
MD_DIR.mkdir(exist_ok=True)

AR5IV_BASE = "https://ar5iv.labs.arxiv.org"


def load_papers() -> dict[str, dict]:
    with open(DATA / "bamboo_final.json") as f:
        return {p["paper_id"]: p for p in json.load(f)}


def get_missing_with_arxiv(papers: dict) -> list[tuple[str, str]]:
    """Return (paper_id, arxiv_id) for papers with arXiv ID but no markdown."""
    result = []
    for pid, p in sorted(papers.items()):
        arxiv_id = p.get("arxiv_id", "")
        if not arxiv_id:
            continue
        md_path = MD_DIR / f"{pid}.md"
        if md_path.exists() and md_path.stat().st_size > 500:
            continue
        result.append((pid, arxiv_id))
    return result


def fetch_html(arxiv_id: str, timeout: int = 60) -> str:
    """Download ar5iv HTML for an arXiv paper."""
    url = f"{AR5IV_BASE}/html/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (bamboo-collector)"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read().decode("utf-8", errors="replace")


def convert_html_to_markdown(html: str, arxiv_id: str) -> str:
    """Convert ar5iv HTML to clean markdown with tables, images, and formulas."""
    # Fix relative image URLs to absolute
    html = re.sub(
        r'src="(/html/[^"]+)"',
        lambda m: f'src="{AR5IV_BASE}{m.group(1)}"',
        html,
    )

    soup = BeautifulSoup(html, "html.parser")

    # Step 1: Convert <math alttext="..."> to $LaTeX$
    for math_tag in soup.find_all("math"):
        alt = math_tag.get("alttext", "")
        if alt:
            # Display math if inside equation div
            parent_classes = " ".join(math_tag.parent.get("class", [])) if math_tag.parent else ""
            is_display = "ltx_equation" in parent_classes or "ltx_eqn" in parent_classes
            if is_display:
                math_tag.replace_with(f" $${alt}$$ ")
            else:
                math_tag.replace_with(f"${alt}$")
        else:
            math_tag.replace_with(math_tag.get_text())

    # Step 2: Enhance figure captions — keep figure number + caption text
    for fig in soup.find_all("figure"):
        caption = fig.find("figcaption")
        if caption:
            # Extract caption text and prepend as paragraph after the image
            cap_text = caption.get_text(strip=True)
            # Don't remove caption — markdownify will handle it

    # Step 3: Remove non-content elements
    for tag in soup.find_all(["nav", "script", "style", "header", "footer"]):
        tag.decompose()
    # Remove bibliography section (too noisy, not needed for claims)
    for bib in soup.find_all("section", class_="ltx_bibliography"):
        bib.decompose()

    # Step 4: Extract main content
    main = soup.find("article") or soup.find("div", class_="ltx_page_content") or soup

    # Step 5: Convert to markdown
    result = md(str(main), heading_style="ATX", strip=["span"])

    # Step 6: Clean up
    result = re.sub(r"\n{3,}", "\n\n", result)  # collapse excessive blank lines
    result = re.sub(r"\[([^\]]*)\]\{[^}]*\}", r"\1", result)  # remove {.ltx_*} attributes
    result = re.sub(r"^\s*\n", "\n", result, flags=re.MULTILINE)  # strip leading whitespace on blank lines

    return result


def process_one(pid: str, arxiv_id: str) -> tuple[str, bool, str]:
    """Process a single paper. Returns (pid, success, message)."""
    md_path = MD_DIR / f"{pid}.md"
    if md_path.exists() and md_path.stat().st_size > 500:
        return pid, True, "already exists"

    try:
        html = fetch_html(arxiv_id)
        markdown = convert_html_to_markdown(html, arxiv_id)

        if len(markdown) < 1000:
            return pid, False, f"output too short ({len(markdown)} chars)"

        md_path.write_text(markdown, encoding="utf-8")
        return pid, True, f"{len(markdown)} chars"

    except Exception as e:
        return pid, False, str(e)[:200]


def main():
    parser = argparse.ArgumentParser(description="Convert arXiv papers to markdown via ar5iv")
    parser.add_argument("--ids", nargs="+", help="Specific paper IDs to convert")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    papers = load_papers()

    if args.ids:
        todo = []
        for pid in args.ids:
            p = papers.get(pid)
            if not p:
                print(f"WARNING: {pid} not found in dataset")
                continue
            arxiv_id = p.get("arxiv_id", "")
            if not arxiv_id:
                print(f"WARNING: {pid} has no arXiv ID")
                continue
            todo.append((pid, arxiv_id))
    else:
        todo = get_missing_with_arxiv(papers)

    if not todo:
        print("Nothing to convert — all papers with arXiv IDs already have markdowns")
        return

    print(f"Converting {len(todo)} papers via ar5iv...", flush=True)

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for pid, arxiv_id in todo:
            f = pool.submit(process_one, pid, arxiv_id)
            futures[f] = (pid, arxiv_id)

        for f in as_completed(futures):
            pid, success, msg = f.result()
            if success:
                ok += 1
                print(f"  OK  {pid}: {msg}", flush=True)
            else:
                fail += 1
                print(f"  FAIL {pid}: {msg}", flush=True)

    print(f"\nDone: {ok} converted, {fail} failed out of {len(todo)}", flush=True)


if __name__ == "__main__":
    main()
