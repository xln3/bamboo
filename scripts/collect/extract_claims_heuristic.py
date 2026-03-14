#!/usr/bin/env python3
"""Extract ground truth claims from paper PDFs using MinerU + heuristics.

A non-LLM approach: uses MinerU for high-quality PDF→markdown extraction,
then applies regex/heuristic rules to extract claims from result tables
and inline numbers. Useful as a baseline or when no LLM API is available.

Usage:
    python extract_claims_heuristic.py [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from pdf_extractor import extract_text_mineru, extract_text_fast

log = logging.getLogger("extract_claims_heuristic")

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "bamboo_final.json"

# Common ML metric names and their properties
KNOWN_METRICS = {
    # Higher-is-better metrics
    "accuracy": {"hib": True, "tol": 0.05, "unit": "%"},
    "acc": {"hib": True, "tol": 0.05, "unit": "%"},
    "top-1": {"hib": True, "tol": 0.05, "unit": "%"},
    "top-5": {"hib": True, "tol": 0.05, "unit": "%"},
    "f1": {"hib": True, "tol": 0.05, "unit": "%"},
    "f1-score": {"hib": True, "tol": 0.05, "unit": "%"},
    "bleu": {"hib": True, "tol": 0.05, "unit": "raw"},
    "bleu-4": {"hib": True, "tol": 0.05, "unit": "raw"},
    "rouge": {"hib": True, "tol": 0.05, "unit": "raw"},
    "rouge-l": {"hib": True, "tol": 0.05, "unit": "raw"},
    "rouge-1": {"hib": True, "tol": 0.05, "unit": "raw"},
    "meteor": {"hib": True, "tol": 0.05, "unit": "raw"},
    "map": {"hib": True, "tol": 0.05, "unit": "%"},
    "ap": {"hib": True, "tol": 0.05, "unit": "%"},
    "ap50": {"hib": True, "tol": 0.05, "unit": "%"},
    "ap75": {"hib": True, "tol": 0.05, "unit": "%"},
    "auc": {"hib": True, "tol": 0.05, "unit": "ratio_0_1"},
    "auroc": {"hib": True, "tol": 0.05, "unit": "ratio_0_1"},
    "recall": {"hib": True, "tol": 0.05, "unit": "%"},
    "precision": {"hib": True, "tol": 0.05, "unit": "%"},
    "psnr": {"hib": True, "tol": 0.05, "unit": "dB"},
    "ssim": {"hib": True, "tol": 0.05, "unit": "ratio_0_1"},
    "iou": {"hib": True, "tol": 0.05, "unit": "%"},
    "miou": {"hib": True, "tol": 0.05, "unit": "%"},
    "dice": {"hib": True, "tol": 0.05, "unit": "%"},
    "ndcg": {"hib": True, "tol": 0.05, "unit": "raw"},
    "hit": {"hib": True, "tol": 0.05, "unit": "%"},
    "mrr": {"hib": True, "tol": 0.05, "unit": "raw"},
    "cider": {"hib": True, "tol": 0.05, "unit": "raw"},
    "spice": {"hib": True, "tol": 0.05, "unit": "raw"},
    "em": {"hib": True, "tol": 0.05, "unit": "%"},
    # Lower-is-better metrics
    "fid": {"hib": False, "tol": 0.10, "unit": "raw"},
    "loss": {"hib": False, "tol": 0.05, "unit": "raw"},
    "perplexity": {"hib": False, "tol": 0.05, "unit": "raw"},
    "ppl": {"hib": False, "tol": 0.05, "unit": "raw"},
    "wer": {"hib": False, "tol": 0.05, "unit": "%"},
    "cer": {"hib": False, "tol": 0.05, "unit": "%"},
    "mae": {"hib": False, "tol": 0.05, "unit": "raw"},
    "mse": {"hib": False, "tol": 0.05, "unit": "raw"},
    "rmse": {"hib": False, "tol": 0.05, "unit": "raw"},
    "error": {"hib": False, "tol": 0.05, "unit": "%"},
    "eer": {"hib": False, "tol": 0.05, "unit": "%"},
    "ter": {"hib": False, "tol": 0.05, "unit": "%"},
    "nll": {"hib": False, "tol": 0.05, "unit": "raw"},
    "bpd": {"hib": False, "tol": 0.05, "unit": "raw"},
    "lpips": {"hib": False, "tol": 0.10, "unit": "raw"},
}

# Pattern to match markdown tables
TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n)+)",
    re.MULTILINE,
)

# Pattern to match numbers (including negative, decimal, scientific notation)
NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")

# Pattern to match "our method achieves XX.X% accuracy"
INLINE_CLAIM_RE = re.compile(
    r"(?:our|proposed|we)\s+(?:method|model|approach|system|framework)?\s*"
    r"(?:achieves?|obtains?|reaches?|scores?|gets?|yields?|demonstrates?)\s+"
    r"(?:(?:a|an|the)\s+)?"
    r"(?:(?:new\s+)?(?:state[- ]of[- ]the[- ]art|SOTA|best)\s+)?"
    r"(\d+\.?\d*)\s*%?\s*"
    r"(?:in\s+|on\s+|for\s+)?"
    r"(\w[\w\s-]{2,30})",
    re.IGNORECASE,
)


def parse_markdown_table(table_text: str) -> list[dict]:
    """Parse a markdown table into rows of dicts."""
    lines = table_text.strip().split("\n")
    if len(lines) < 3:
        return []

    # Parse header
    header = [cell.strip() for cell in lines[0].split("|")[1:-1]]

    # Skip separator line (line 1)

    # Parse data rows
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) == len(header):
            row = dict(zip(header, cells))
            rows.append(row)

    return rows


def find_metric_in_header(header: str) -> dict | None:
    """Check if a table header matches a known metric."""
    h_lower = header.lower().strip()
    # Remove common prefixes/suffixes
    h_clean = re.sub(r"[↑↓▲▼\(\)]", "", h_lower).strip()

    for metric_name, props in KNOWN_METRICS.items():
        if metric_name in h_clean or h_clean == metric_name:
            return {"name": metric_name, **props}

    return None


def extract_claims_from_table(table_text: str, table_idx: int,
                               paper_title: str) -> list[dict]:
    """Extract claims from a single markdown table."""
    rows = parse_markdown_table(table_text)
    if not rows:
        return []

    # Get headers
    lines = table_text.strip().split("\n")
    headers = [cell.strip() for cell in lines[0].split("|")[1:-1]]

    # Find which columns contain metrics
    metric_cols = {}
    for i, h in enumerate(headers):
        metric_info = find_metric_in_header(h)
        if metric_info:
            metric_cols[i] = (h, metric_info)

    if not metric_cols:
        return []

    claims = []
    for row in rows:
        row_label = list(row.values())[0] if row else ""
        # Check if this row might be the proposed method
        # (heuristic: bold text, or row with "ours", or last row)
        is_ours = any(kw in row_label.lower() for kw in
                      ["ours", "our", "proposed", "**"])

        for col_idx, (col_header, metric_info) in metric_cols.items():
            if col_idx >= len(headers):
                continue
            cell_value = row.get(headers[col_idx], "")

            # Extract number from cell (handle bold, ±, etc.)
            cell_clean = re.sub(r"\*\*|\*|\\textbf{|}|±.*", "", cell_value)
            nums = NUMBER_RE.findall(cell_clean)
            if not nums:
                continue

            try:
                value = float(nums[0])
            except ValueError:
                continue

            claims.append({
                "claim_id": "",  # Will be assigned later
                "description": f"Table {table_idx + 1}: {row_label} - {col_header}",
                "metric_name": metric_info["name"],
                "metric_value": value,
                "metric_unit": metric_info["unit"],
                "source_location": f"Table {table_idx + 1}",
                "higher_is_better": metric_info["hib"],
                "tolerance": metric_info["tol"],
                "category": "main" if is_ours else "baseline",
            })

    return claims


def extract_claims_from_text(text: str) -> list[dict]:
    """Extract inline claims from text."""
    claims = []
    for m in INLINE_CLAIM_RE.finditer(text):
        value_str = m.group(1)
        context = m.group(2).strip()
        try:
            value = float(value_str)
        except ValueError:
            continue

        claims.append({
            "claim_id": "",
            "description": f"Inline: {m.group(0)[:100]}",
            "metric_name": "accuracy",  # Default for inline claims
            "metric_value": value,
            "metric_unit": "%",
            "source_location": "Text (inline)",
            "higher_is_better": True,
            "tolerance": 0.05,
            "category": "main",
        })

    return claims


def extract_claims_from_numbers_near_metrics(text: str) -> list[dict]:
    """Extract claims by finding known metric names near numbers in text.

    Works with pdftotext output where tables are space-aligned.
    """
    claims = []
    lines = text.split("\n")

    # Track which metric context we're in
    current_table = None
    table_idx = 0

    for line_no, line in enumerate(lines):
        line_lower = line.lower().strip()

        # Detect table captions
        m = re.match(r"(?:table|tab\.?)\s*(\d+)", line_lower)
        if m:
            current_table = int(m.group(1))
            table_idx = current_table

        # Look for lines with metric names and numbers
        for metric_name, props in KNOWN_METRICS.items():
            if metric_name in line_lower:
                # Found a metric name in this line — extract all numbers nearby
                nums = NUMBER_RE.findall(line)
                for num_str in nums:
                    try:
                        val = float(num_str)
                        # Skip numbers that are likely years, indices, etc.
                        if 1900 < val < 2100:
                            continue
                        if val == 0:
                            continue
                        claims.append({
                            "claim_id": "",
                            "description": f"Table {table_idx}: {metric_name} = {val}",
                            "metric_name": metric_name,
                            "metric_value": val,
                            "metric_unit": props["unit"],
                            "source_location": f"Table {table_idx}" if current_table else f"Line {line_no}",
                            "higher_is_better": props["hib"],
                            "tolerance": props["tol"],
                            "category": "main",
                        })
                        break  # One number per metric per line
                    except ValueError:
                        continue

    return claims


def extract_claims_from_paper(text: str, title: str) -> list[dict]:
    """Extract all claims from paper text."""
    all_claims = []

    # Find and parse markdown tables (if text has them)
    for i, m in enumerate(TABLE_RE.finditer(text)):
        table_claims = extract_claims_from_table(m.group(1), i, title)
        all_claims.extend(table_claims)

    # If no markdown tables found, try line-by-line metric detection
    # (works with pdftotext output)
    if not all_claims:
        metric_claims = extract_claims_from_numbers_near_metrics(text)
        all_claims.extend(metric_claims)

    # Extract inline claims
    inline_claims = extract_claims_from_text(text)
    all_claims.extend(inline_claims)

    # Deduplicate: same metric + same value = same claim
    seen = set()
    unique_claims = []
    for c in all_claims:
        key = (c["metric_name"], c["metric_value"])
        if key not in seen:
            seen.add(key)
            unique_claims.append(c)

    # Assign claim IDs
    for i, c in enumerate(unique_claims):
        c["claim_id"] = f"c{i + 1}"

    return unique_claims


def get_pdf_url(paper: dict) -> str:
    """Determine best PDF URL for a paper."""
    if paper.get("pdf_url", "").startswith("http"):
        return paper["pdf_url"]
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}"
    return ""


def download_pdf(url: str, path: str) -> bool:
    """Download PDF via curl."""
    result = subprocess.run(
        ["curl", "-sL", "-o", path, "--max-time", "60", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    try:
        return os.path.getsize(path) > 1000
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract claims from paper PDFs using MinerU + heuristics",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--input", type=str, default=str(DATA_PATH),
                        help="Input JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    input_path = Path(args.input)
    with open(input_path) as f:
        papers = json.load(f)

    # Find papers without claims that have a PDF source
    to_process = []
    for i, p in enumerate(papers):
        if p.get("ground_truth_claims"):
            continue
        pdf_url = get_pdf_url(p)
        if pdf_url:
            to_process.append((i, p, pdf_url))

    if args.limit:
        to_process = to_process[:args.limit]

    log.info(f"Processing {len(to_process)} papers for claim extraction")

    success = 0
    failed = 0
    total_claims = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (i, paper, pdf_url) in enumerate(to_process):
            title = paper.get("title", "Unknown")
            log.info(f"[{idx + 1}/{len(to_process)}] {title[:70]}")

            pdf_path = os.path.join(tmpdir, "paper.pdf")

            try:
                if not download_pdf(pdf_url, pdf_path):
                    log.warning(f"  PDF download failed")
                    failed += 1
                    continue

                text = extract_text_fast(pdf_path)
                if not text or len(text) < 500:
                    log.warning(f"  Text extraction failed")
                    failed += 1
                    continue

                claims = extract_claims_from_paper(text, title)

                if claims:
                    papers[i]["ground_truth_claims"] = claims
                    papers[i]["_claims_method"] = "heuristic"
                    success += 1
                    total_claims += len(claims)
                    log.info(f"  Extracted {len(claims)} claims")
                else:
                    log.info(f"  No claims found via heuristics")
                    failed += 1

            except Exception as e:
                log.error(f"  Error: {e}")
                failed += 1

            try:
                os.remove(pdf_path)
            except OSError:
                pass

            # Rate limit
            time.sleep(1.0)

            # Save progress
            if (idx + 1) % 10 == 0:
                with open(input_path, "w") as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                log.info(f"  Progress: {success} success, {failed} failed, "
                         f"{total_claims} claims total")

    # Final save
    with open(input_path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    n_with_claims = sum(1 for p in papers if p.get("ground_truth_claims"))
    print(f"\n{'='*60}")
    print(f"Heuristic claim extraction complete:")
    print(f"  New: {success} papers, {total_claims} claims")
    print(f"  Failed: {failed}")
    print(f"  Total with claims: {n_with_claims}/{len(papers)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
