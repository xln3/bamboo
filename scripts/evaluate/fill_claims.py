#!/usr/bin/env python3
"""Fill ground_truth_claims in bamboo_curated.json using Claude Opus 4.6.

Calls `claude` CLI (--model opus) for EVERY paper — even those with existing
claims — to ensure quality.  Saves each paper's claims to an individual file
under data/paper_claims_v2/ for resumability, then merges into bamboo_curated.json.

Usage:
    # Full run — all 3983 papers, 8 parallel workers
    python3 -m scripts.evaluate.fill_claims --workers 8

    # Resume (skips papers with existing v2 claim files)
    python3 -m scripts.evaluate.fill_claims --workers 8

    # Limit for testing
    python3 -m scripts.evaluate.fill_claims --limit 5 --workers 2

    # Only merge existing v2 claim files into bamboo_curated.json
    python3 -m scripts.evaluate.fill_claims --merge-only
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger("fill_claims")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
CURATED_DIR = DATA / "bamboo_curated"
CLAIMS_V2 = DATA / "paper_claims_v2"
MD_DIR = DATA / "paper_markdowns"

LOWER_IS_BETTER_METRICS = {
    "fid", "loss", "perplexity", "ppl", "wer", "cer", "mae", "mse", "rmse",
    "error", "error_rate", "eer", "ter", "nll", "bpc", "bits_per_char",
    "latency", "inference_time", "training_time", "time", "runtime",
}

VALID_CATEGORIES = {"main", "ablation", "analysis", "baseline"}

SYSTEM_PROMPT = """\
You are a meticulous ML research auditor. Extract ALL reproducible quantitative claims from this machine-learning paper.

A "claim" is a single experimental result that:
1. Reports a concrete numeric value (accuracy, F1, BLEU, FID, mIoU, latency, etc.).
2. Is produced by the authors' own method (NOT a cited baseline).
3. Can be verified by re-running the authors' released code.

For each claim, return EXACTLY these fields:
- claim_id: Sequential identifier (c1, c2, c3, ...)
- description: Human-readable summary, e.g., "Table 1: CIFAR-10 test accuracy of ProposedMethod (ResNet-50 backbone)"
- metric_name: The metric exactly as the paper uses it (e.g., "mIoU", "Top-1 Accuracy", "BLEU-4", "FID")
- metric_value: The numeric value as a float (47.73, not "47.73%"). Keep original precision.
- metric_unit: "%" for percentages, "ratio_0_1" for 0-1 ratios, "raw" for unitless, "dB", "ms", etc.
- source_location: Where in the paper (e.g., "Table 1", "Table 3, row 2", "Figure 4")
- higher_is_better: true if higher is better (accuracy), false if lower is better (FID, loss)
- tolerance: Relative tolerance for reproduction matching:
  - 0.01 for deterministic metrics
  - 0.05 for standard metrics (accuracy, F1, BLEU, mAP)
  - 0.10 for high-variance metrics (FID, IS)
  - 0.20 for timing/throughput metrics
- dataset: The evaluation dataset (e.g., "ImageNet-1K", "nuScenes", "WMT14 En-De")
- category: "main" for primary results, "ablation" for ablation studies, "analysis" for extra analysis. Do NOT extract "baseline" rows.

RULES:
1. One claim = one number. "92.3 / 87.1" (precision/recall) → two claims.
2. If reported as "95.3 ± 0.2", use metric_value: 95.3.
3. Do NOT extract: hyperparameters, dataset statistics, training cost, qualitative results.
4. Extract EXACT values from the paper. Do not fabricate.
5. Bold/underlined values are especially important — capture them.

COMPLETENESS IS CRITICAL — do NOT omit any result:
6. Extract EVERY row and column for the authors' method from EVERY results table (Table 1, 2, 3, ...).
7. Extract ALL ablation study results (every variant/component removal row).
8. Extract results from Figures if exact numeric values are given (bar charts with labels, user studies with percentages, inference time comparisons).
9. If the method has multiple variants/scales (e.g., Small/Base/Large, different backbones), extract ALL of them.
10. Include analysis results: inference speed, FLOPs, parameter counts if reported as experimental results.
11. Go through the paper section by section — do not stop after the main results table.

Return ONLY a JSON array of claim objects. No markdown fences, no commentary.
IMPORTANT: Output compact JSON (no indentation, no pretty-printing) to avoid output truncation."""


def build_prompt(paper: dict, md_text: str) -> str:
    """Build the full prompt for claude CLI."""
    return f"""{SYSTEM_PROMPT}

---

Paper: {paper['title']}
Venue: {paper['venue']} {paper['year']}
Code: {paper['code_url']}

--- BEGIN PAPER MARKDOWN ---
{md_text}
--- END PAPER MARKDOWN ---

Extract all reproducible quantitative claims. Return ONLY a compact JSON array (no indentation)."""


def validate_claim(claim: dict, idx: int) -> dict | None:
    """Validate and normalize a single claim. Returns cleaned claim or None."""
    if not isinstance(claim, dict):
        return None

    for field in ("description", "metric_name", "metric_value", "source_location"):
        if field not in claim:
            return None

    val = claim.get("metric_value")
    if not isinstance(val, (int, float)):
        try:
            val = float(val)
        except (ValueError, TypeError):
            return None

    clean: dict = {
        "claim_id": f"c{idx}",
        "description": str(claim.get("description", "")),
        "metric_name": str(claim.get("metric_name", "")),
        "metric_value": val,
        "source_location": str(claim.get("source_location", "")),
    }

    clean["metric_unit"] = str(claim.get("metric_unit", "%"))

    hib = claim.get("higher_is_better")
    if isinstance(hib, bool):
        clean["higher_is_better"] = hib
    else:
        metric_lower = clean["metric_name"].lower().replace(" ", "_").replace("-", "_")
        clean["higher_is_better"] = metric_lower not in LOWER_IS_BETTER_METRICS

    tol = claim.get("tolerance", 0.05)
    if isinstance(tol, (int, float)) and 0.001 <= tol <= 0.5:
        clean["tolerance"] = tol
    else:
        clean["tolerance"] = 0.05

    ds = claim.get("dataset")
    if ds and isinstance(ds, str):
        clean["dataset"] = ds

    cat = claim.get("category", "main")
    clean["category"] = cat if cat in VALID_CATEGORIES else "main"

    return clean


def parse_claims_json(text: str) -> list[dict] | None:
    """Extract JSON array from LLM output text, handling truncated output."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)

    # Strip any leading commentary before the array
    bracket_pos = text.find("[")
    if bracket_pos > 0:
        text = text[bracket_pos:]

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find all JSON arrays in text and concatenate them
    # (model sometimes outputs multiple arrays separated by ---)
    all_claims = []
    for match in re.finditer(r"\[.*?\]", text, re.DOTALL):
        try:
            chunk = json.loads(match.group(0))
            if isinstance(chunk, list):
                all_claims.extend(chunk)
        except json.JSONDecodeError:
            continue
    if all_claims:
        return all_claims

    # Handle truncated output: try to recover partial JSON array
    # Case 1: truncated at end (has [ but no ])
    if "[" in text:
        arr_start = text.index("[")
        last_brace = text.rfind("}")
        if last_brace > arr_start:
            candidate = text[arr_start:last_brace + 1].rstrip().rstrip(",") + "]"
            try:
                result = json.loads(candidate)
                if isinstance(result, list) and len(result) > 0:
                    log.info(f"Recovered {len(result)} claims from end-truncated output")
                    return result
            except json.JSONDecodeError:
                last_complete = candidate.rfind("},")
                if last_complete > 0:
                    candidate = candidate[:last_complete + 1] + "]"
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, list) and len(result) > 0:
                            log.info(f"Recovered {len(result)} claims from end-truncated output")
                            return result
                    except json.JSONDecodeError:
                        pass

    # Case 2: truncated at start (no [ but has claim_id objects)
    if '"claim_id"' in text:
        # Find first complete claim object
        first_obj = text.find('{"claim_id"')
        if first_obj < 0:
            first_obj = text.find("{\"claim_id\"")
        if first_obj >= 0:
            fragment = text[first_obj:]
            last_brace = fragment.rfind("}")
            if last_brace > 0:
                candidate = "[" + fragment[:last_brace + 1].rstrip().rstrip(",") + "]"
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list) and len(result) > 0:
                        log.info(f"Recovered {len(result)} claims from start-truncated output")
                        return result
                except json.JSONDecodeError:
                    last_complete = candidate.rfind("},")
                    if last_complete > 0:
                        candidate = candidate[:last_complete + 1] + "]"
                        try:
                            result = json.loads(candidate)
                            if isinstance(result, list) and len(result) > 0:
                                log.info(f"Recovered {len(result)} claims from start-truncated output")
                                return result
                        except json.JSONDecodeError:
                            pass

    return None


def extract_one(paper: dict) -> tuple[str, list[dict] | None]:
    """Extract claims for one paper via claude CLI. Returns (paper_id, claims)."""
    pid = paper["paper_id"]
    out_file = CLAIMS_V2 / f"{pid}.json"

    # Resume: skip if already extracted (including valid 0-claim papers)
    if out_file.exists():
        try:
            claims = json.loads(out_file.read_text())
            if isinstance(claims, list):
                return pid, claims
        except (json.JSONDecodeError, OSError):
            pass  # re-extract on corrupt file

    md_path = DATA / paper["md_file"]
    if not md_path.exists():
        log.warning(f"{pid}: markdown not found")
        return pid, None

    md_text = md_path.read_text()
    if len(md_text) < 500:
        log.warning(f"{pid}: markdown too short ({len(md_text)} chars)")
        return pid, None

    # Truncate very long papers to fit context
    max_chars = 300_000  # Opus has 200k tokens, ~750k chars; leave room for prompt
    if len(md_text) > max_chars:
        md_text = md_text[:max_chars]

    prompt = build_prompt(paper, md_text)

    # Write prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(prompt)
        prompt_file = tmp.name

    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", "sonnet",
                "--output-format", "text",
                "--max-turns", "1",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=900,  # 15 min per paper
        )
    except subprocess.TimeoutExpired:
        log.error(f"{pid}: claude CLI timed out")
        return pid, None
    except Exception as e:
        log.error(f"{pid}: claude CLI error: {e}")
        return pid, None
    finally:
        os.unlink(prompt_file)

    if result.returncode != 0:
        log.error(f"{pid}: claude exit code {result.returncode}: {result.stderr[:200]}")
        return pid, None

    raw = parse_claims_json(result.stdout)
    if raw is None:
        log.error(f"{pid}: failed to parse JSON from output ({len(result.stdout)} chars)")
        # Save raw output for debugging
        (CLAIMS_V2 / f"{pid}.raw.txt").write_text(result.stdout)
        return pid, None

    # Validate and clean
    claims = []
    for i, r in enumerate(raw):
        c = validate_claim(r, i + 1)
        if c is not None:
            claims.append(c)

    for i, c in enumerate(claims):
        c["claim_id"] = f"c{i + 1}"

    # Save individual claim file
    out_file.write_text(json.dumps(claims, indent=2, ensure_ascii=False))

    return pid, claims


def merge_into_curated(papers: list[dict]) -> int:
    """Merge all v2 claim files into papers list. Returns count updated."""
    updated = 0
    for paper in papers:
        pid = paper["paper_id"]
        claim_file = CLAIMS_V2 / f"{pid}.json"
        if claim_file.exists():
            try:
                claims = json.loads(claim_file.read_text())
                if isinstance(claims, list) and len(claims) > 0:
                    paper["ground_truth_claims"] = claims
                    updated += 1
            except (json.JSONDecodeError, OSError):
                pass
    return updated


def load_curated() -> list[dict]:
    """Load all papers from the partitioned curated directory."""
    papers = []
    for f in sorted(CURATED_DIR.glob("bamboo-*.json")):
        with open(f) as fh:
            papers.extend(json.load(fh))
    papers.sort(key=lambda p: p["paper_id"])
    return papers


def save(papers: list[dict]):
    """Save papers back to partitioned chunk files (100 per file)."""
    papers.sort(key=lambda p: p["paper_id"])
    chunk_size = 100
    # Clear old chunks
    for old in CURATED_DIR.glob("bamboo-*.json"):
        old.unlink()
    for i in range(0, len(papers), chunk_size):
        chunk = papers[i:i + chunk_size]
        first_id = chunk[0]["paper_id"]
        last_id = chunk[-1]["paper_id"]
        out = CURATED_DIR / f"{first_id}_to_{last_id}.json"
        with open(out, "w") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Fill ground_truth_claims via Claude Opus 4.6")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to extract (for testing)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save progress every N papers (default: 50)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing v2 claim files, no extraction")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    CLAIMS_V2.mkdir(exist_ok=True)

    # Load curated dataset
    papers = load_curated()
    papers_by_id = {p["paper_id"]: p for p in papers}

    if args.merge_only:
        n = merge_into_curated(papers)
        save(papers)
        total = sum(1 for p in papers if p.get("ground_truth_claims"))
        total_claims = sum(
            len(p["ground_truth_claims"]) for p in papers
            if p.get("ground_truth_claims"))
        log.info(f"Merged {n} papers. Total: {total}/{len(papers)}, {total_claims} claims")
        return

    # Determine what needs extraction (skip papers with existing v2 claim files)
    to_extract = []
    already_done = 0
    for p in papers:
        pid = p["paper_id"]
        v2_file = CLAIMS_V2 / f"{pid}.json"
        if v2_file.exists():
            try:
                c = json.loads(v2_file.read_text())
                if isinstance(c, list):  # accept 0-claim papers as done
                    already_done += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        to_extract.append(p)

    if args.limit is not None:
        to_extract = to_extract[:args.limit]

    log.info(f"Total: {len(papers)} papers. Already done (v2): {already_done}. "
             f"To extract: {len(to_extract)}")

    if not to_extract:
        log.info("All papers already extracted. Running merge...")
        n = merge_into_curated(papers)
        save(papers)
        log.info(f"Merged {n}. Done.")
        return

    success = 0
    failed = 0
    counter = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_one, p): p["paper_id"]
                   for p in to_extract}

        for future in as_completed(futures):
            pid = futures[future]
            counter += 1
            try:
                _, claims = future.result()
                if claims is not None:
                    if len(claims) > 0:
                        papers_by_id[pid]["ground_truth_claims"] = claims
                    success += 1
                    log.info(f"[{counter}/{len(to_extract)}] {pid}: "
                             f"{len(claims)} claims ✓")
                else:
                    failed += 1
                    log.warning(f"[{counter}/{len(to_extract)}] {pid}: FAILED")
            except Exception as e:
                failed += 1
                log.error(f"[{counter}/{len(to_extract)}] {pid}: {e}")

            if counter % args.save_every == 0:
                merge_into_curated(papers)
                save(papers)
                total = sum(1 for p in papers if p.get("ground_truth_claims"))
                log.info(f"--- Checkpoint: {success} ok, {failed} fail, "
                         f"{total}/{len(papers)} total ---")

    # Final merge and save
    merge_into_curated(papers)
    save(papers)

    total = sum(1 for p in papers if p.get("ground_truth_claims"))
    total_claims = sum(
        len(p["ground_truth_claims"]) for p in papers
        if p.get("ground_truth_claims"))
    log.info(f"Done: +{success} extracted, {failed} failed. "
             f"Total: {total}/{len(papers)} papers, {total_claims} claims")


if __name__ == "__main__":
    main()
