#!/usr/bin/env python3
"""Fill ground_truth_claims using GLM-5.1 via Anthropic-compatible API.

Uses asyncio + httpx for efficient concurrent requests with proper rate-limit
backoff. Reads paper markdowns, extracts claims via GLM-5.1, saves per-paper
JSON files under data/paper_claims_v2/.

Usage:
    # Extract claims for all papers missing v2 files
    python3 -m scripts.evaluate.fill_claims_glm --workers 16

    # Resume (skips papers with existing v2 claim files)
    python3 -m scripts.evaluate.fill_claims_glm --workers 16

    # Only merge existing v2 claim files into bamboo_curated.json
    python3 -m scripts.evaluate.fill_claims_glm --merge-only

    # Verify existing claims quality
    python3 -m scripts.evaluate.fill_claims_glm --verify-only --limit 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

import httpx

log = logging.getLogger("fill_claims_glm")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "data"
CURATED = DATA / "bamboo_curated.json"
CLAIMS_V2 = DATA / "paper_claims_v2"
MD_DIR = DATA / "paper_markdowns"

# GLM API config (Anthropic-compatible endpoint) — overridable via CLI args
API_BASE = "https://open.bigmodel.cn/api/anthropic"
API_KEY = "fdf6ff04dba847208473f0c848498049.zfOt5uBBAZZW0ypT"
MODEL = "glm-5.1"

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
    if "[" in text:
        arr_start = text.index("[")
        last_brace = text.rfind("}")
        if last_brace > arr_start:
            candidate = text[arr_start:last_brace + 1].rstrip().rstrip(",") + "]"
            try:
                result = json.loads(candidate)
                if isinstance(result, list) and len(result) > 0:
                    log.info(f"Recovered {len(result)} claims from truncated output")
                    return result
            except json.JSONDecodeError:
                last_complete = candidate.rfind("},")
                if last_complete > 0:
                    candidate = candidate[:last_complete + 1] + "]"
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, list) and len(result) > 0:
                            log.info(f"Recovered {len(result)} claims from truncated output")
                            return result
                    except json.JSONDecodeError:
                        pass

    # Handle start-truncated output
    if '"claim_id"' in text:
        first_obj = text.find('{"claim_id"')
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
                    pass

    return None


def build_user_prompt(paper: dict, md_text: str) -> str:
    """Build the user message for the API call."""
    return f"""Paper: {paper['title']}
Venue: {paper.get('venue', 'Unknown')} {paper.get('year', '')}
Code: {paper.get('code_url', 'N/A')}

--- BEGIN PAPER MARKDOWN ---
{md_text}
--- END PAPER MARKDOWN ---

Extract all reproducible quantitative claims. Return ONLY a compact JSON array (no indentation)."""


async def call_glm(
    client: httpx.AsyncClient,
    paper: dict,
    md_text: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> tuple[str, list[dict] | None]:
    """Call GLM-5.1 API for one paper with retry + backoff."""
    pid = paper["paper_id"]

    user_prompt = build_user_prompt(paper, md_text)

    body = {
        "model": MODEL,
        "max_tokens": 16384,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_prompt}],
    }

    for attempt in range(max_retries):
        async with semaphore:
            # Stagger requests to avoid rate limit bursts
            await asyncio.sleep(3.0)
            try:
                resp = await client.post(
                    f"{API_BASE}/v1/messages",
                    json=body,
                    headers={
                        "x-api-key": API_KEY,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    timeout=300.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    # Extract text from Anthropic-format response
                    text = ""
                    for block in data.get("content", []):
                        if block.get("type") == "text":
                            text += block["text"]
                    if not text:
                        log.warning(f"{pid}: empty response")
                        return pid, None

                    raw = parse_claims_json(text)
                    if raw is None:
                        log.error(f"{pid}: failed to parse JSON ({len(text)} chars)")
                        (CLAIMS_V2 / f"{pid}.raw.txt").write_text(text)
                        return pid, None

                    # Validate and clean
                    claims = []
                    for i, r in enumerate(raw):
                        c = validate_claim(r, i + 1)
                        if c is not None:
                            claims.append(c)
                    for i, c in enumerate(claims):
                        c["claim_id"] = f"c{i + 1}"

                    return pid, claims

                elif resp.status_code in (429, 529):
                    import random
                    wait = min(2 ** attempt * 10 + random.uniform(5, 15), 180)
                    log.warning(f"{pid}: rate limited ({resp.status_code}), retry in {wait:.0f}s")
                    await asyncio.sleep(wait)
                    continue

                elif resp.status_code >= 500:
                    wait = min(2 ** attempt * 3, 60)
                    log.warning(f"{pid}: server error {resp.status_code}, retry in {wait}s")
                    await asyncio.sleep(wait)
                    continue

                else:
                    log.error(f"{pid}: API error {resp.status_code}: {resp.text[:300]}")
                    return pid, None

            except httpx.TimeoutException:
                wait = min(2 ** attempt * 5, 120)
                log.warning(f"{pid}: timeout, retry in {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
                continue

            except Exception as e:
                log.error(f"{pid}: unexpected error: {e}")
                return pid, None

    log.error(f"{pid}: all {max_retries} retries exhausted")
    return pid, None


async def extract_one(
    client: httpx.AsyncClient,
    paper: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, list[dict] | None]:
    """Extract claims for one paper. Returns (paper_id, claims)."""
    pid = paper["paper_id"]
    out_file = CLAIMS_V2 / f"{pid}.json"

    # Resume: skip if already extracted
    if out_file.exists():
        try:
            claims = json.loads(out_file.read_text())
            if isinstance(claims, list):
                return pid, claims
        except (json.JSONDecodeError, OSError):
            pass

    md_path = MD_DIR / f"{pid}.md"
    if not md_path.exists():
        # Try from paper metadata
        md_file = paper.get("md_file", "")
        if md_file:
            md_path = DATA / md_file
    if not md_path.exists():
        log.warning(f"{pid}: markdown not found")
        return pid, None

    md_text = md_path.read_text()
    if len(md_text) < 500:
        log.warning(f"{pid}: markdown too short ({len(md_text)} chars)")
        return pid, None

    # Truncate very long papers
    max_chars = 200_000  # GLM-5.1 context may be smaller than Claude
    if len(md_text) > max_chars:
        md_text = md_text[:max_chars]

    pid, claims = await call_glm(client, paper, md_text, semaphore)

    if claims is not None:
        out_file.write_text(json.dumps(claims, indent=2, ensure_ascii=False))

    return pid, claims


async def verify_claims(papers: list[dict], limit: int = 50):
    """Verify quality of existing v2 claim files by spot-checking."""
    import random

    v2_files = sorted(CLAIMS_V2.glob("*.json"))
    if not v2_files:
        log.info("No v2 files to verify")
        return

    sample = random.sample(v2_files, min(limit, len(v2_files)))
    papers_by_id = {p["paper_id"]: p for p in papers}

    total_claims = 0
    issues = []
    stats = {"total": 0, "valid": 0, "empty": 0, "outlier_values": 0,
             "missing_dataset": 0, "duplicate_claims": 0}

    for f in sample:
        pid = f.stem
        try:
            claims = json.loads(f.read_text())
        except json.JSONDecodeError:
            issues.append(f"{pid}: corrupt JSON")
            continue

        stats["total"] += 1
        if not claims:
            stats["empty"] += 1
            continue

        stats["valid"] += 1
        total_claims += len(claims)

        # Check for issues
        seen_vals = set()
        for c in claims:
            val = c.get("metric_value", 0)
            # Extreme outlier check
            if abs(val) > 1_000_000:
                stats["outlier_values"] += 1
                issues.append(f"{pid}/{c['claim_id']}: extreme value {val}")

            if not c.get("dataset"):
                stats["missing_dataset"] += 1

            # Duplicate check (same metric+value+dataset)
            key = (c.get("metric_name"), val, c.get("dataset"))
            if key in seen_vals:
                stats["duplicate_claims"] += 1
            seen_vals.add(key)

    log.info(f"Verified {stats['total']} files ({total_claims} claims):")
    log.info(f"  Valid: {stats['valid']}, Empty: {stats['empty']}")
    log.info(f"  Outlier values: {stats['outlier_values']}")
    log.info(f"  Missing dataset: {stats['missing_dataset']}")
    log.info(f"  Duplicate claims: {stats['duplicate_claims']}")
    if issues:
        log.info(f"  Issues ({len(issues)}):")
        for issue in issues[:20]:
            log.info(f"    - {issue}")


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


def save(papers: list[dict]):
    """Atomically save curated JSON."""
    tmp = CURATED.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)
    tmp.replace(CURATED)


async def run_extraction(papers: list[dict], workers: int, save_every: int,
                          limit: int | None, partition: str | None = None):
    """Run async extraction on all papers needing claims."""
    CLAIMS_V2.mkdir(exist_ok=True)
    papers_by_id = {p["paper_id"]: p for p in papers}

    # Determine what needs extraction
    to_extract = []
    already_done = 0
    for p in papers:
        pid = p["paper_id"]
        v2_file = CLAIMS_V2 / f"{pid}.json"
        if v2_file.exists():
            try:
                c = json.loads(v2_file.read_text())
                if isinstance(c, list):
                    already_done += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        # Must have markdown
        md_path = MD_DIR / f"{pid}.md"
        if not md_path.exists():
            md_file = p.get("md_file", "")
            if md_file:
                md_path = DATA / md_file
        if md_path.exists():
            to_extract.append(p)

    if partition:
        k, n = map(int, partition.split("/"))
        to_extract = [p for i, p in enumerate(to_extract) if i % n == k]
        log.info(f"Partition {k}/{n}: taking {len(to_extract)} papers")

    if limit is not None:
        to_extract = to_extract[:limit]

    log.info(f"Total: {len(papers)} papers. Already done (v2): {already_done}. "
             f"To extract: {len(to_extract)} (model={MODEL})")

    if not to_extract:
        log.info("All papers already extracted. Running merge...")
        n = merge_into_curated(papers)
        save(papers)
        log.info(f"Merged {n}. Done.")
        return

    semaphore = asyncio.Semaphore(workers)
    success = 0
    failed = 0
    counter = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        # Process in batches for checkpoint saves
        batch_size = save_every
        for batch_start in range(0, len(to_extract), batch_size):
            batch = to_extract[batch_start:batch_start + batch_size]

            tasks = [
                extract_one(client, p, semaphore)
                for p in batch
            ]

            for coro in asyncio.as_completed(tasks):
                pid, claims = await coro
                counter += 1

                if claims is not None:
                    if len(claims) > 0:
                        papers_by_id[pid]["ground_truth_claims"] = claims
                    success += 1
                    log.info(f"[{counter}/{len(to_extract)}] {pid}: "
                             f"{len(claims)} claims ✓")
                else:
                    failed += 1
                    log.warning(f"[{counter}/{len(to_extract)}] {pid}: FAILED")

            # Checkpoint
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


def main():
    parser = argparse.ArgumentParser(
        description="Fill ground_truth_claims via GLM-5.1")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to extract (for testing)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Concurrent API requests (default: 16)")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save progress every N papers (default: 50)")
    parser.add_argument("--merge-only", action="store_true",
                        help="Only merge existing v2 claim files, no extraction")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify quality of existing v2 claim files")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name (e.g. glm-5, glm-5.1)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Override API key")
    parser.add_argument("--partition", type=str, default=None,
                        help="Worker partition 'K/N' — this worker takes every Nth paper starting at K (0-indexed)")
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    CLAIMS_V2.mkdir(exist_ok=True)

    with open(CURATED) as f:
        papers = json.load(f)

    if args.verify_only:
        asyncio.run(verify_claims(papers, limit=args.limit or 50))
        return

    if args.merge_only:
        n = merge_into_curated(papers)
        save(papers)
        total = sum(1 for p in papers if p.get("ground_truth_claims"))
        total_claims = sum(
            len(p["ground_truth_claims"]) for p in papers
            if p.get("ground_truth_claims"))
        log.info(f"Merged {n} papers. Total: {total}/{len(papers)}, {total_claims} claims")
        return

    asyncio.run(run_extraction(papers, args.workers, args.save_every, args.limit, args.partition))


if __name__ == "__main__":
    main()
