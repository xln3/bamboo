#!/usr/bin/env python3
"""Compute difficulty scores for BAMBOO papers.

Analyzes each paper's code repository to compute a composite difficulty score
based on 6 dimensions. Assigns difficulty tiers (1-4).

Uses GitHub API for repo metadata and heuristics for scoring.
Works directly on bamboo_final.json.

Usage:
    python compute_difficulty.py [--limit N] [--token GITHUB_TOKEN]

Environment:
    GITHUB_TOKEN  - optional, for higher API rate limits
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("compute_difficulty")

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "bamboo_final.json"
GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _github_headers(token: str) -> list[str]:
    """Build curl headers for GitHub API."""
    headers = ["-H", "Accept: application/vnd.github.v3+json"]
    if token:
        headers.extend(["-H", f"Authorization: Bearer {token}"])
    return headers


def _curl_json(url: str, token: str = "") -> dict | None:
    """Fetch JSON from URL via curl."""
    cmd = ["curl", "-s", "--max-time", "15"]
    cmd.extend(_github_headers(token))
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict) and data.get("message", "").startswith("API rate limit"):
            log.warning("GitHub API rate limit hit")
            return None
        return data
    except json.JSONDecodeError:
        return None


def parse_github_url(code_url: str) -> tuple[str, str] | None:
    """Parse owner/repo from a GitHub URL."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", code_url)
    if m:
        return m.group(1), m.group(2)
    return None


def get_repo_info(owner: str, repo: str, token: str = "") -> dict | None:
    """Get repo metadata from GitHub API."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    return _curl_json(url, token)


def get_repo_languages(owner: str, repo: str, token: str = "") -> dict | None:
    """Get language breakdown from GitHub API."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/languages"
    return _curl_json(url, token)


def get_repo_tree_sample(owner: str, repo: str, sha: str,
                         token: str = "") -> dict | None:
    """Get repo file tree (non-recursive, root level) from GitHub API."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{sha}"
    return _curl_json(url, token)


# ---------------------------------------------------------------------------
# Scoring heuristics (each returns 0.0-1.0)
# ---------------------------------------------------------------------------

# Known frameworks and their complexity weights
FRAMEWORK_COMPLEXITY = {
    # Simple pip-install frameworks
    "scikit-learn": 0.1, "sklearn": 0.1,
    "xgboost": 0.15, "lightgbm": 0.15,
    # Standard DL frameworks
    "pytorch": 0.3, "torch": 0.3,
    "tensorflow": 0.35, "keras": 0.25,
    "jax": 0.4, "flax": 0.4,
    # Complex / custom
    "detectron2": 0.5, "mmdetection": 0.5, "mmcv": 0.5,
    "fairseq": 0.5, "megatron": 0.7,
    "deepspeed": 0.6, "colossalai": 0.6,
    "triton": 0.6, "cuda": 0.7,
    "apex": 0.5, "flash_attn": 0.5, "flash-attn": 0.5,
    "transformers": 0.3, "diffusers": 0.35,
    "paddlepaddle": 0.35, "mxnet": 0.35,
}

# Keywords suggesting heavy GPU/hardware requirements
GPU_KEYWORDS = [
    "multi-gpu", "multi_gpu", "multigpu", "distributed",
    "8xA100", "4xA100", "8xH100", "4xH100", "8xV100",
    "data_parallel", "model_parallel", "pipeline_parallel",
    "fsdp", "deepspeed", "megatron", "zero3", "zero-3",
    "a100", "h100", "v100",
]

# Keywords suggesting large datasets
LARGE_DATA_KEYWORDS = [
    "imagenet", "laion", "c4", "pile", "redpajama", "openwebtext",
    "common_crawl", "commoncrawl", "webvid", "webdataset",
    "kinetics", "howto100m", "yfcc", "cc3m", "cc12m",
]


def score_framework_complexity(languages: dict, title: str,
                                abstract: str) -> float:
    """Score framework complexity (0-1) from languages and title/abstract."""
    text = f"{title} {abstract}".lower()

    # Check for known frameworks
    max_score = 0.3  # default (assumes PyTorch-level)
    for fw, score in FRAMEWORK_COMPLEXITY.items():
        if fw in text:
            max_score = max(max_score, score)

    # C/C++ or CUDA presence bumps complexity
    if languages:
        total_bytes = sum(languages.values()) or 1
        cpp_bytes = languages.get("C++", 0) + languages.get("C", 0)
        cuda_bytes = languages.get("Cuda", 0) + languages.get("CUDA", 0)
        if cuda_bytes > 0:
            max_score = max(max_score, 0.7)
        elif cpp_bytes / total_bytes > 0.2:
            max_score = max(max_score, 0.5)

    return min(max_score, 1.0)


def score_dependency_count(repo_info: dict, languages: dict) -> float:
    """Score dependency complexity (0-1) from repo size and language count."""
    size_kb = repo_info.get("size", 0) if repo_info else 0
    n_langs = len(languages) if languages else 1

    # More languages and larger repos tend to have more dependencies
    size_score = min(size_kb / 100000, 1.0)  # 100MB = max
    lang_score = min(n_langs / 10, 1.0)

    return min(0.5 * size_score + 0.5 * lang_score, 1.0)


def score_dataset_requirements(title: str, abstract: str) -> float:
    """Score dataset requirements (0-1) from title/abstract keywords."""
    text = f"{title} {abstract}".lower()

    for kw in LARGE_DATA_KEYWORDS:
        if kw in text:
            return 0.8

    # Check for dataset-heavy indicators
    if any(w in text for w in ["pretrain", "pre-train", "large-scale"]):
        return 0.6

    if any(w in text for w in ["fine-tun", "finetun"]):
        return 0.4

    return 0.3  # default: standard dataset


def score_hardware_requirements(title: str, abstract: str,
                                 repo_info: dict) -> float:
    """Score hardware requirements (0-1) from text analysis."""
    text = f"{title} {abstract}".lower()

    for kw in GPU_KEYWORDS:
        if kw in text:
            return 0.8

    # Large model indicators
    if any(w in text for w in ["billion", "7b", "13b", "70b", "175b"]):
        return 0.9

    if any(w in text for w in ["llm", "language model", "large model"]):
        return 0.6

    # Vision models are typically medium
    if any(w in text for w in ["image", "video", "3d"]):
        return 0.4

    return 0.3  # default: single GPU


def score_code_quality(repo_info: dict) -> float:
    """Score code quality (0=good, 1=poor) from repo metadata.

    Inverted: higher score = harder to reproduce (worse quality).
    """
    if not repo_info:
        return 0.7  # unknown = assume medium-poor

    has_readme = True  # GitHub repos almost always have one
    stars = repo_info.get("stargazers_count", 0)

    # Well-starred repos tend to have better documentation
    if stars > 100:
        return 0.2
    elif stars > 20:
        return 0.3
    elif stars > 5:
        return 0.5
    else:
        return 0.6


def score_reproduction_time(title: str, abstract: str, domain: str) -> float:
    """Score estimated reproduction time (0-1) from paper content."""
    text = f"{title} {abstract}".lower()

    # Training-heavy keywords
    if any(w in text for w in ["pretrain", "pre-train", "large-scale training"]):
        return 0.8

    if any(w in text for w in ["training time", "24 hour", "48 hour",
                                "days of training", "gpu hours"]):
        return 0.7

    # Domain-based defaults
    domain_defaults = {
        "nlp": 0.5,
        "vision": 0.4,
        "generative": 0.6,
        "multimodal": 0.5,
        "reinforcement-learning": 0.5,
        "robotics": 0.4,
        "audio": 0.4,
        "graph": 0.3,
        "tabular": 0.2,
        "systems": 0.4,
        "theory": 0.2,
        "other": 0.4,
    }

    return domain_defaults.get(domain, 0.4)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

DIFFICULTY_WEIGHTS = {
    "framework_complexity": 0.20,
    "dependency_count": 0.15,
    "dataset_requirements": 0.20,
    "hardware_requirements": 0.20,
    "code_quality": 0.10,
    "reproduction_time": 0.15,
}


def compute_difficulty(paper: dict, repo_info: dict | None,
                       languages: dict | None) -> dict:
    """Compute difficulty score for a single paper.

    Returns dict with tier, composite_score, and per-dimension scores.
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    domain = paper.get("domain", "other")

    scores = {
        "framework_complexity": score_framework_complexity(
            languages, title, abstract),
        "dependency_count": score_dependency_count(repo_info, languages),
        "dataset_requirements": score_dataset_requirements(title, abstract),
        "hardware_requirements": score_hardware_requirements(
            title, abstract, repo_info),
        "code_quality": score_code_quality(repo_info),
        "reproduction_time": score_reproduction_time(title, abstract, domain),
    }

    # Weighted composite (0-1 scale)
    composite = sum(
        scores[dim] * weight
        for dim, weight in DIFFICULTY_WEIGHTS.items()
    )

    # Map 0-1 composite to 1-5 tier scale
    tier_score = 1.0 + composite * 4.0  # maps [0,1] → [1,5]

    if tier_score <= 2.0:
        tier = 1
    elif tier_score <= 3.0:
        tier = 2
    elif tier_score <= 4.0:
        tier = 3
    else:
        tier = 4

    return {
        "tier": tier,
        "composite_score": round(tier_score, 2),
        "dimensions": {
            dim: round(scores[dim], 3)
            for dim in DIFFICULTY_WEIGHTS
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_papers(papers: list[dict], token: str = "",
                   limit: int | None = None) -> int:
    """Compute difficulty for papers missing it.

    Returns number of papers scored.
    """
    to_process = [
        (i, p) for i, p in enumerate(papers)
        if p.get("difficulty") is None
    ]

    if limit is not None:
        to_process = to_process[:limit]

    log.info(f"Computing difficulty for {len(to_process)} papers "
             f"({sum(1 for p in papers if p.get('difficulty'))} already done)")

    scored = 0
    api_calls = 0

    for idx, (i, paper) in enumerate(to_process):
        code_url = paper.get("code_url", "")
        parsed = parse_github_url(code_url)

        repo_info = None
        languages = None

        if parsed:
            owner, repo = parsed
            repo_info = get_repo_info(owner, repo, token)
            if repo_info:
                languages = get_repo_languages(owner, repo, token)
                api_calls += 2
            else:
                api_calls += 1

            # Rate limit: GitHub allows 5000/hour with token, 60/hour without
            if api_calls % 50 == 0:
                time.sleep(1.0)
        else:
            # Non-GitHub repos: score with text-only heuristics
            pass

        difficulty = compute_difficulty(paper, repo_info, languages)
        papers[i]["difficulty"] = difficulty
        scored += 1

        if (idx + 1) % 100 == 0:
            done_pct = (idx + 1) / len(to_process) * 100
            log.info(f"  [{idx + 1}/{len(to_process)}] ({done_pct:.0f}%) "
                     f"api_calls={api_calls}")

    return scored


def main():
    parser = argparse.ArgumentParser(
        description="Compute difficulty scores for BAMBOO papers",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max papers to process")
    parser.add_argument("--token", type=str,
                        default=os.environ.get("GITHUB_TOKEN", ""),
                        help="GitHub API token")
    parser.add_argument("--input", type=str, default=str(DATA_PATH),
                        help="Input JSON path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: overwrite input)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    with open(input_path) as f:
        papers = json.load(f)

    log.info(f"Loaded {len(papers)} papers from {input_path}")

    scored = process_papers(papers, token=args.token, limit=args.limit)

    # Save
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    # Summary
    tiers = {1: 0, 2: 0, 3: 0, 4: 0}
    for p in papers:
        d = p.get("difficulty")
        if d:
            tiers[d["tier"]] = tiers.get(d["tier"], 0) + 1

    total_scored = sum(1 for p in papers if p.get("difficulty"))
    print(f"\n{'='*60}")
    print(f"Difficulty scoring complete: {scored} new, {total_scored} total")
    print(f"\nTier distribution:")
    for tier in sorted(tiers):
        print(f"  Tier {tier}: {tiers[tier]:>5d} papers")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
