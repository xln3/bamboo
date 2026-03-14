#!/usr/bin/env python3
"""Extract ground truth claims from paper PDFs using an LLM.

Phase 3 of the BAMBOO dataset construction pipeline.
For each paper, downloads the PDF, extracts text using MinerU (with
pdftotext fallback), sends it to an LLM to identify all quantitative
claims, and saves structured claim data back into bamboo_final.json.

Supports both per-venue JSONs (legacy) and bamboo_final.json (current).

Usage:
    python extract_claims.py [--limit 10] [--model gpt-4o-mini] [--api-base URL]
    python extract_claims.py --venue iclr2025  # legacy per-venue mode

Environment:
    OPENAI_API_KEY  - required for LLM API auth
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
import urllib.request
import urllib.error
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger("extract_claims")

CLAIM_SCHEMA_FIELDS = [
    "claim_id", "description", "metric_name", "metric_value", "metric_unit",
    "source_location", "higher_is_better", "tolerance", "claim_difficulty",
    "experiment_hint", "dataset", "category",
]

VALID_CATEGORIES = {"main", "ablation", "analysis", "baseline"}
VALID_DIFFICULTIES = {"quick", "medium", "long", "prohibitive"}
VALID_UNITS = {"%", "ratio_0_1", "raw", "dB", "ms", "s", "FLOPs", "params", "tokens/s"}

# Metrics where lower is better
LOWER_IS_BETTER_METRICS = {
    "fid", "loss", "perplexity", "ppl", "wer", "cer", "mae", "mse", "rmse",
    "error", "error_rate", "eer", "ter", "nll", "bpc", "bits_per_char",
    "latency", "inference_time", "training_time", "time", "runtime",
}

SYSTEM_PROMPT = """You are a precise scientific data extractor. Your job is to extract ALL quantitative experimental claims from a machine learning paper.

A "claim" is any numerical result that the paper reports from an experiment. This includes:
- Values in result tables (each cell with a number is a potential claim)
- Numerical results mentioned in figures (if exact values are given)
- Inline numerical results in the text (e.g., "our method achieves 95.3% accuracy")

For each claim, extract EXACTLY these fields:
- claim_id: Sequential identifier (c1, c2, c3, ...)
- description: Human-readable description (e.g., "Table 1: CIFAR-10 test accuracy of ProposedMethod")
- metric_name: The metric (accuracy, F1, BLEU, FID, mAP, PSNR, SSIM, perplexity, etc.)
- metric_value: The EXACT numerical value from the paper (as a number, not string)
- metric_unit: The unit or scale. Use "%" for percentages, "ratio_0_1" for 0-1 ratios, "raw" for unitless scores, "dB" for decibels, etc.
- source_location: Where in the paper (e.g., "Table 1", "Table 2, row 3", "Figure 4", "Section 4.2")
- higher_is_better: true if higher values are better (e.g., accuracy, F1), false if lower is better (e.g., FID, loss, perplexity, error rate)
- tolerance: Relative tolerance for reproduction matching, based on metric type:
  - 0.01 for deterministic metrics (exact match, parameter count, dataset size)
  - 0.05 for standard metrics (accuracy, F1, BLEU, mAP, PSNR)
  - 0.10 for high-variance metrics (FID, IS, generative metrics)
  - 0.20 for timing/throughput metrics (inference speed, training time, FLOPs/s)
- claim_difficulty: Estimated compute to reproduce:
  - "quick": < 10 min (small dataset, few epochs, inference only)
  - "medium": 10 min - 2 hours (standard training on small/medium dataset)
  - "long": 2 - 24 hours (large dataset, many epochs, multi-GPU)
  - "prohibitive": > 24 hours or requires resources most labs don't have
- experiment_hint: Optional hint about which experiment/script to run (if inferable from the paper)
- dataset: The dataset name used for this result (e.g., "CIFAR-10", "ImageNet", "WMT14 En-De")
- category: One of:
  - "main": Primary experimental results (the paper's main contribution tables)
  - "ablation": Ablation study results
  - "analysis": Additional analysis, scaling experiments, visualization metrics
  - "baseline": Results re-reported from other papers (not original to this paper)

IMPORTANT RULES:
1. Extract the EXACT numerical values from the paper. Do not round or modify them.
2. Only extract claims for the PROPOSED METHOD in this paper, not for competing baselines (mark those as category "baseline").
3. For tables, extract results for the proposed method's rows. If the paper has multiple variants, extract all of them.
4. If a value is reported as "95.3 ± 0.2", use 95.3 as the metric_value.
5. Do NOT extract values that are hyperparameters, dataset statistics, or model architecture details — only experimental RESULTS.
6. For percentage values shown as "95.3", use metric_value: 95.3 and metric_unit: "%".
7. For values shown as "0.953", determine from context whether it's a ratio (metric_unit: "ratio_0_1") or a raw score.
8. Be precise about source_location — specify the exact table, row, and column when possible.
9. Set higher_is_better correctly for each specific metric.

Respond with ONLY a JSON array of claim objects. No markdown, no explanation, no code fences. Just the raw JSON array."""


USER_PROMPT_TEMPLATE = """Extract ALL quantitative experimental claims from this paper.

Paper title: {title}

--- BEGIN PAPER TEXT ---
{text}
--- END PAPER TEXT ---

Return a JSON array of claims. Each claim must have all required fields: claim_id, description, metric_name, metric_value, source_location. Optional fields: metric_unit (default "%"), higher_is_better (default true), tolerance (default 0.05), claim_difficulty, experiment_hint, dataset, category (default "main").

Remember: extract EXACT numerical values. Only experimental results, not hyperparameters."""


def get_pdf_url(paper: dict) -> str:
    """Determine best PDF URL for a paper."""
    if paper.get("pdf_url", "").startswith("http"):
        return paper["pdf_url"]
    if paper.get("arxiv_id"):
        return f"https://arxiv.org/pdf/{paper['arxiv_id']}"
    return ""


def download_pdf(url: str, path: str) -> bool:
    """Download PDF via curl. Returns True on success."""
    result = subprocess.run(
        ["curl", "-sL", "-o", path, "--max-time", "60", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    # Verify it's actually a PDF (at least a few KB)
    try:
        size = os.path.getsize(path)
        if size < 1000:
            return False
    except OSError:
        return False
    return True


def extract_text(pdf_path: str) -> str:
    """Extract text from PDF using MinerU (with pdftotext fallback)."""
    try:
        from pdf_extractor import extract_text_mineru
        text = extract_text_mineru(pdf_path)
        if text and len(text) > 100:
            return text
    except ImportError:
        log.warning("pdf_extractor module not available, using pdftotext")
    except Exception as e:
        log.warning(f"MinerU extraction failed: {e}")

    # Fallback to pdftotext
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", pdf_path, "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return result.stdout
    except subprocess.TimeoutExpired:
        log.warning("pdftotext timed out")
    except Exception as e:
        log.warning(f"pdftotext failed: {e}")
    return ""


def call_llm(text: str, title: str, model: str, api_base: str, api_key: str) -> list[dict] | None:
    """Send extracted text to LLM and parse the JSON response.

    Returns a list of claim dicts on success, None on failure.
    """
    # Truncate text to avoid token limits (~100k chars is roughly 25k tokens)
    max_chars = 100_000
    if len(text) > max_chars:
        text = text[:max_chars]
        log.info(f"  Truncated text to {max_chars} chars")

    user_content = USER_PROMPT_TEMPLATE.format(title=title, text=text)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 8192,
    }

    url = f"{api_base.rstrip('/')}/chat/completions"
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        log.error(f"  LLM API HTTP {e.code}: {e.reason}")
        try:
            err_body = e.read().decode("utf-8")[:500]
            log.debug(f"  Response: {err_body}")
        except Exception:
            pass
        return None
    except urllib.error.URLError as e:
        log.error(f"  LLM API connection error: {e.reason}")
        return None
    except Exception as e:
        log.error(f"  LLM API error: {e}")
        return None

    # Extract the response text
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        log.error(f"  Unexpected LLM response structure")
        return None

    # Parse JSON from the response (handle markdown code fences)
    content = content.strip()
    if content.startswith("```"):
        # Strip markdown code fences
        lines = content.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        content = "\n".join(lines)

    try:
        claims = json.loads(content)
    except json.JSONDecodeError as e:
        log.error(f"  Failed to parse LLM JSON: {e}")
        log.debug(f"  Raw content (first 500 chars): {content[:500]}")
        # Try to find a JSON array in the response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                claims = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(claims, list):
        log.error(f"  LLM returned non-list: {type(claims)}")
        return None

    return claims


def validate_claim(claim: dict, idx: int) -> dict | None:
    """Validate and normalize a single claim dict. Returns cleaned claim or None."""
    if not isinstance(claim, dict):
        return None

    # Required fields
    for field in ("description", "metric_name", "metric_value", "source_location"):
        if field not in claim:
            log.debug(f"  Claim {idx}: missing required field '{field}'")
            return None

    # Validate metric_value is a number
    val = claim.get("metric_value")
    if not isinstance(val, (int, float)):
        try:
            val = float(val)
        except (ValueError, TypeError):
            log.debug(f"  Claim {idx}: metric_value not numeric: {val}")
            return None

    # Build clean claim
    clean = {
        "claim_id": f"c{idx}",
        "description": str(claim.get("description", "")),
        "metric_name": str(claim.get("metric_name", "")),
        "metric_value": val,
        "source_location": str(claim.get("source_location", "")),
    }

    # Optional fields with defaults
    unit = str(claim.get("metric_unit", "%"))
    clean["metric_unit"] = unit

    hib = claim.get("higher_is_better")
    if isinstance(hib, bool):
        clean["higher_is_better"] = hib
    elif isinstance(hib, str) and hib.lower() in ("true", "false"):
        clean["higher_is_better"] = hib.lower() == "true"
    else:
        # Infer from metric name when not provided or not parseable
        metric_lower = clean["metric_name"].lower().replace(" ", "_").replace("-", "_")
        clean["higher_is_better"] = metric_lower not in LOWER_IS_BETTER_METRICS

    tol = claim.get("tolerance", 0.05)
    if isinstance(tol, (int, float)) and 0.001 <= tol <= 0.5:
        clean["tolerance"] = tol
    else:
        clean["tolerance"] = 0.05

    difficulty = claim.get("claim_difficulty")
    if difficulty in VALID_DIFFICULTIES:
        clean["claim_difficulty"] = difficulty

    hint = claim.get("experiment_hint")
    if hint and isinstance(hint, str):
        clean["experiment_hint"] = hint

    dataset = claim.get("dataset")
    if dataset and isinstance(dataset, str):
        clean["dataset"] = dataset

    category = claim.get("category", "main")
    clean["category"] = category if category in VALID_CATEGORIES else "main"

    return clean


def process_paper(paper: dict, tmpdir: str, model: str, api_base: str, api_key: str) -> list[dict] | None:
    """Extract claims from a single paper. Returns list of claims or None on failure."""
    title = paper.get("title", "Unknown")

    pdf_url = get_pdf_url(paper)
    if not pdf_url:
        log.warning(f"  No PDF URL for: {title[:60]}")
        return None

    pdf_path = os.path.join(tmpdir, "paper.pdf")

    # Download
    if not download_pdf(pdf_url, pdf_path):
        log.warning(f"  PDF download failed: {pdf_url}")
        return None

    # Extract text
    text = extract_text(pdf_path)
    if not text or len(text) < 500:
        log.warning(f"  PDF text extraction failed or too short ({len(text)} chars)")
        return None

    log.info(f"  Extracted {len(text)} chars from PDF")

    # Call LLM
    raw_claims = call_llm(text, title, model, api_base, api_key)
    if raw_claims is None:
        log.warning(f"  LLM extraction failed")
        return None

    # Validate and number claims
    claims = []
    for i, raw in enumerate(raw_claims):
        clean = validate_claim(raw, i + 1)
        if clean is not None:
            claims.append(clean)

    # Re-number sequentially after filtering
    for i, c in enumerate(claims):
        c["claim_id"] = f"c{i + 1}"

    log.info(f"  Extracted {len(claims)} valid claims (from {len(raw_claims)} raw)")
    return claims


def process_venue(venue_id: str, model: str, api_base: str, api_key: str,
                  limit: int | None = None):
    """Process all valid papers in a venue."""
    path = DATA_DIR / f"{venue_id}.json"
    if not path.exists():
        log.error(f"{venue_id}: JSON not found at {path}")
        return

    with open(path) as f:
        papers = json.load(f)

    # Select papers: _repo_valid and not yet extracted
    to_process = []
    for i, p in enumerate(papers):
        if not p.get("_repo_valid"):
            continue
        if p.get("ground_truth_claims"):
            continue
        if p.get("_claims_error"):
            # Skip papers that previously failed — can be retried by clearing the flag
            continue
        to_process.append((i, p))

    if limit is not None:
        to_process = to_process[:limit]

    already_done = sum(1 for p in papers if p.get("ground_truth_claims"))
    valid_total = sum(1 for p in papers if p.get("_repo_valid"))
    log.info(f"{venue_id}: {valid_total} valid papers, {already_done} already extracted, "
             f"processing {len(to_process)}")

    if not to_process:
        return

    success = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (i, paper) in enumerate(to_process):
            title = paper.get("title", "Unknown")
            log.info(f"[{idx + 1}/{len(to_process)}] {title[:70]}")

            try:
                claims = process_paper(paper, tmpdir, model, api_base, api_key)

                if claims is not None and len(claims) > 0:
                    papers[i]["ground_truth_claims"] = claims
                    papers[i]["_claims_extracted_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
                    )
                    papers[i]["_claims_model"] = model
                    success += 1
                elif claims is not None and len(claims) == 0:
                    log.warning(f"  No claims extracted (paper may be theoretical)")
                    papers[i]["_claims_error"] = "no_claims_found"
                    failed += 1
                else:
                    papers[i]["_claims_error"] = "extraction_failed"
                    failed += 1

            except Exception as e:
                log.error(f"  Unexpected error: {e}")
                papers[i]["_claims_error"] = f"exception: {str(e)[:100]}"
                failed += 1

            # Rate limit: respect API and PDF hosts
            time.sleep(1.0)

            # Save progress every 10 papers
            if (idx + 1) % 10 == 0:
                with open(path, "w") as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                log.info(f"  Progress saved. {success} success, {failed} failed "
                         f"({idx + 1}/{len(to_process)})")

    # Final save
    with open(path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_with_claims = sum(1 for p in papers if p.get("ground_truth_claims"))
    log.info(f"{venue_id}: +{success} extracted, {failed} failed. "
             f"Total with claims: {total_with_claims}/{valid_total}")


FINAL_JSON = Path(__file__).parent.parent.parent / "data" / "bamboo_final.json"


def process_final_json(model: str, api_base: str, api_key: str,
                       limit: int | None = None, retry_failed: bool = False):
    """Process bamboo_final.json directly (current mode)."""
    if not FINAL_JSON.exists():
        log.error(f"bamboo_final.json not found at {FINAL_JSON}")
        return

    with open(FINAL_JSON) as f:
        papers = json.load(f)

    # Select papers to process
    to_process = []
    for i, p in enumerate(papers):
        if p.get("ground_truth_claims"):
            continue
        if p.get("_claims_error") and not retry_failed:
            continue
        to_process.append((i, p))

    if limit is not None:
        to_process = to_process[:limit]

    already_done = sum(1 for p in papers if p.get("ground_truth_claims"))
    log.info(f"Total: {len(papers)} papers, {already_done} already extracted, "
             f"processing {len(to_process)}")

    if not to_process:
        return

    # Clear error flags for retry
    if retry_failed:
        for i, p in to_process:
            if "_claims_error" in papers[i]:
                del papers[i]["_claims_error"]

    success = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (i, paper) in enumerate(to_process):
            title = paper.get("title", "Unknown")
            log.info(f"[{idx + 1}/{len(to_process)}] {title[:70]}")

            try:
                claims = process_paper(paper, tmpdir, model, api_base, api_key)

                if claims is not None and len(claims) > 0:
                    papers[i]["ground_truth_claims"] = claims
                    papers[i]["_claims_extracted_at"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
                    )
                    papers[i]["_claims_model"] = model
                    success += 1
                elif claims is not None and len(claims) == 0:
                    log.warning(f"  No claims extracted")
                    papers[i]["_claims_error"] = "no_claims_found"
                    failed += 1
                else:
                    papers[i]["_claims_error"] = "extraction_failed"
                    failed += 1

            except Exception as e:
                log.error(f"  Unexpected error: {e}")
                papers[i]["_claims_error"] = f"exception: {str(e)[:100]}"
                failed += 1

            time.sleep(1.0)

            # Save progress every 10 papers
            if (idx + 1) % 10 == 0:
                with open(FINAL_JSON, "w") as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                log.info(f"  Progress saved. {success} success, {failed} failed "
                         f"({idx + 1}/{len(to_process)})")

    # Final save
    with open(FINAL_JSON, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    total_with_claims = sum(1 for p in papers if p.get("ground_truth_claims"))
    total_claim_count = sum(
        len(p["ground_truth_claims"]) for p in papers
        if p.get("ground_truth_claims")
    )
    log.info(f"Done: +{success} extracted, {failed} failed. "
             f"Total: {total_with_claims}/{len(papers)} "
             f"({total_claim_count} claims)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ground truth claims from paper PDFs using an LLM",
    )
    parser.add_argument("--venue", type=str, default=None,
                        help="Process only this venue (legacy per-venue mode)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        help="LLM model name (default: gpt-4o-mini)")
    parser.add_argument("--api-base", type=str, default="https://api.openai.com/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry papers with _claims_error")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        log.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    if args.venue:
        # Legacy per-venue mode
        process_venue(args.venue, args.model, args.api_base, api_key, args.limit)
    else:
        # Current mode: process bamboo_final.json directly
        process_final_json(args.model, args.api_base, api_key,
                           limit=args.limit, retry_failed=args.retry_failed)

    # Summary
    if FINAL_JSON.exists():
        papers = json.loads(FINAL_JSON.read_text())
        n_with_claims = sum(1 for p in papers if p.get("ground_truth_claims"))
        n_total_claims = sum(
            len(p["ground_truth_claims"]) for p in papers
            if p.get("ground_truth_claims")
        )
        print(f"\n{'='*60}")
        print(f"  Papers with claims: {n_with_claims}/{len(papers)}")
        print(f"  Total claims: {n_total_claims}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
