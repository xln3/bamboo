#!/usr/bin/env python3
"""Finalize BAMBOO benchmark dataset from raw venue JSON files.

Filters validated papers, assigns paper IDs, classifies domain, and writes
the final benchmark JSON.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import DATA_DIR

# All 9 venues in canonical order
VENUE_IDS = [
    "aaai2025", "acl2025", "cvpr2025", "emnlp2025",
    "iccv2025", "iclr2025", "icml2025", "icra2025", "neurips2025",
]

OUTPUT_PATH = DATA_DIR.parent / "bamboo_final.json"


# ---------------------------------------------------------------------------
# Domain classification (keyword heuristics)
# ---------------------------------------------------------------------------

# Each domain maps to a list of patterns. Patterns are checked against
# the lowercased title + abstract. Order matters: first match wins, so more
# specific domains come before broader ones.
DOMAIN_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("robotics", [
        r"\brobot\w*\b", r"\bmanipulat\w*\b", r"\bgrasping\b",
        r"\blocomotion\b", r"\bnavigat\w*\b", r"\bslam\b",
        r"\bodometry\b", r"\bautonomous\s+driv\w*\b", r"\bego[\s-]?motion\b",
        r"\bmotion\s+planning\b", r"\bpath\s+planning\b",
        r"\brobo\w*\s+learn\w*\b", r"\bsim[\s-]?to[\s-]?real\b",
        r"\bdrone\b", r"\buav\b", r"\bquadrotor\b", r"\bhumanoid\b",
        r"\bkinematic\w*\b", r"\bdexterous\b", r"\bend[\s-]?effector\b",
        r"\brobot\w*\s+arm\b", r"\bcontrol\s+polic\w*\b",
    ]),
    ("audio", [
        r"\baudio\b", r"\bspeech\b", r"\bspeaker\b", r"\bvoice\b",
        r"\bmusic\b", r"\bsound\b", r"\bacoustic\b", r"\basr\b",
        r"\btts\b", r"\btext[\s-]?to[\s-]?speech\b",
        r"\bspeech[\s-]?recognition\b", r"\bspeech[\s-]?synthesis\b",
        r"\bspeech[\s-]?enhancement\b", r"\bsource\s+separation\b",
        r"\bmel[\s-]?spectro\w*\b", r"\bwaveform\b",
    ]),
    ("multimodal", [
        r"\bmultimodal\b", r"\bmulti[\s-]?modal\b", r"\bcross[\s-]?modal\b",
        r"\bvision[\s-]?language\b", r"\bvisual[\s-]?question\b",
        r"\bvqa\b", r"\bimage[\s-]?caption\w*\b", r"\bvideo[\s-]?caption\w*\b",
        r"\btext[\s-]?to[\s-]?image\b", r"\bimage[\s-]?to[\s-]?text\b",
        r"\btext[\s-]?to[\s-]?video\b", r"\bvideo[\s-]?to[\s-]?text\b",
        r"\bclip\b", r"\bvisual[\s-]?ground\w*\b",
        r"\bvision[\s-]?and[\s-]?language\b", r"\bembodied\b",
        r"\bmulti[\s-]?view\b", r"\bvideo[\s-]?language\b",
        r"\brefer\w*\s+express\w*\b",
    ]),
    ("generative", [
        r"\bdiffusion\b", r"\bdenois\w*\b", r"\bscore[\s-]?match\w*\b",
        r"\bgan\b", r"\bgenerative\s+adversarial\b",
        r"\bvae\b", r"\bvariational\s+auto[\s-]?encoder\b",
        r"\bflow[\s-]?match\w*\b", r"\bnormalizing\s+flow\b",
        r"\bimage\s+generat\w*\b", r"\bvideo\s+generat\w*\b",
        r"\b3d\s+generat\w*\b", r"\btext[\s-]?to[\s-]?3d\b",
        r"\bimage[\s-]?to[\s-]?3d\b", r"\bnerf\b",
        r"\bneural\s+radiance\b", r"\bgaussian\s+splat\w*\b",
        r"\b3d\s+gaussian\b", r"\bstable\s+diffusion\b",
        r"\blatent\s+diffusion\b", r"\bautoregress\w*\s+generat\w*\b",
        r"\bimage\s+synthesis\b", r"\bstyle\s+transfer\b",
        r"\bsuper[\s-]?resolut\w*\b", r"\binpaint\w*\b",
        r"\bimage\s+edit\w*\b", r"\bimage\s+restor\w*\b",
    ]),
    ("reinforcement-learning", [
        r"\breinforcement\s+learn\w*\b", r"\brl\b", r"\bpolic\w+\s+gradient\b",
        r"\bactor[\s-]?critic\b", r"\bq[\s-]?learn\w*\b", r"\bdqn\b",
        r"\bppo\b", r"\bsac\b", r"\btd3\b", r"\bddpg\b",
        r"\breward\s+model\w*\b", r"\breward\s+shap\w*\b",
        r"\binverse\s+rl\b", r"\bimitation\s+learn\w*\b",
        r"\boffline\s+rl\b", r"\bmodel[\s-]?based\s+rl\b",
        r"\bmulti[\s-]?agent\s+rl\b", r"\bmarl\b",
        r"\brlhf\b", r"\bbandits?\b", r"\bmarkov\s+decision\b",
        r"\bmdp\b", r"\bworld\s+model\b", r"\benvironment\s+model\b",
    ]),
    ("graph", [
        r"\bgraph\s+neural\b", r"\bgnn\b", r"\bgraph\s+network\b",
        r"\bgraph\s+convol\w*\b", r"\bgcn\b", r"\bgat\b",
        r"\bgraph\s+transform\w*\b", r"\bgraph\s+learn\w*\b",
        r"\bmessage\s+passing\b", r"\bnode\s+classif\w*\b",
        r"\blink\s+predict\w*\b", r"\bgraph\s+classif\w*\b",
        r"\bknowledge\s+graph\b", r"\bheterogen\w*\s+graph\b",
        r"\bhypergraph\b", r"\bspectral\s+graph\b",
        r"\bgraph\s+generat\w*\b", r"\bmolecul\w*\s+graph\b",
    ]),
    ("nlp", [
        r"\blanguage\s+model\w*\b", r"\bllm\b", r"\bllms\b",
        r"\btransformer\w*\b", r"\bbert\b", r"\bgpt\b",
        r"\battention\s+mechan\w*\b", r"\bself[\s-]?attention\b",
        r"\btext\s+classif\w*\b", r"\bsentiment\b",
        r"\bnamed\s+entity\b", r"\bner\b", r"\bpos\s+tag\w*\b",
        r"\bmachine\s+translat\w*\b", r"\bnmt\b",
        r"\bsummariz\w*\b", r"\bquestion\s+answer\w*\b",
        r"\binformation\s+extract\w*\b", r"\brelation\s+extract\w*\b",
        r"\bdialogue\b", r"\bchatbot\b", r"\bconversat\w*\b",
        r"\btoken\w*\b", r"\bprompt\w*\b", r"\binstruct\w*\s+tun\w*\b",
        r"\bfine[\s-]?tun\w*\b", r"\bpre[\s-]?train\w*\b",
        r"\btext\s+generat\w*\b", r"\bnlp\b", r"\bnatural\s+language\b",
        r"\bword\s+embed\w*\b", r"\bsentence\s+embed\w*\b",
        r"\bsemantic\s+similar\w*\b", r"\btextual\b",
        r"\bcode\s+generat\w*\b", r"\breason\w*\b",
        r"\bchain[\s-]?of[\s-]?thought\b", r"\bin[\s-]?context\s+learn\w*\b",
        r"\bretrieval[\s-]?augment\w*\b", r"\brag\b",
        r"\balignment\b", r"\brlhf\b",
    ]),
    ("vision", [
        r"\bimage\b", r"\bvisual\b", r"\bvision\b", r"\bconvol\w*\b",
        r"\bcnn\b", r"\bresnet\b", r"\bvit\b",
        r"\bobject\s+detect\w*\b", r"\bsegmentat\w*\b",
        r"\bsemantic\s+segment\w*\b", r"\binstance\s+segment\w*\b",
        r"\bpanoptic\b", r"\bpose\s+estimat\w*\b",
        r"\baction\s+recognit\w*\b", r"\bvideo\s+understand\w*\b",
        r"\boptical\s+flow\b", r"\bdepth\s+estimat\w*\b",
        r"\bstereo\b", r"\b3d\s+reconstruct\w*\b",
        r"\bpoint\s+cloud\b", r"\bface\b", r"\bfacial\b",
        r"\bocr\b", r"\bscene\s+understand\w*\b",
        r"\bimage\s+retriev\w*\b", r"\bvisual\s+recognit\w*\b",
        r"\bbackbone\b", r"\bfeature\s+extract\w*\b",
        r"\bdata\s+augment\w*\b", r"\bself[\s-]?supervis\w*\b",
        r"\bcontrastive\s+learn\w*\b", r"\bvideo\b",
        r"\btrack\w*\b", r"\bmedical\s+imag\w*\b",
        r"\bremote\s+sens\w*\b", r"\bsatellite\b",
    ]),
    ("tabular", [
        r"\btabular\b", r"\btable\b", r"\bstructured\s+data\b",
        r"\bfeature\s+select\w*\b", r"\bgradient\s+boost\w*\b",
        r"\bxgboost\b", r"\blightgbm\b", r"\bcatboost\b",
        r"\brandom\s+forest\b", r"\bdecision\s+tree\b",
        r"\btime[\s-]?series\b", r"\bforecast\w*\b",
        r"\banomaly\s+detect\w*\b", r"\bfraud\s+detect\w*\b",
        r"\brecommendat\w*\b", r"\bcollaborative\s+filter\w*\b",
    ]),
    ("systems", [
        r"\bdistribut\w+\s+train\w*\b", r"\bfederat\w+\s+learn\w*\b",
        r"\bmodel\s+compress\w*\b", r"\bprun\w*\b", r"\bquantiz\w*\b",
        r"\bknowledge\s+distill\w*\b", r"\bneural\s+architect\w*\s+search\b",
        r"\bnas\b", r"\bautoml\b", r"\bhyperparameter\b",
        r"\befficient\s+infer\w*\b", r"\bmodel\s+parallel\w*\b",
        r"\bdata\s+parallel\w*\b", r"\bpipeline\s+parallel\w*\b",
        r"\bmixed[\s-]?precision\b", r"\bspars\w*\b",
        r"\bcompil\w*\b", r"\boperator\s+fusion\b",
        r"\bhardware[\s-]?aware\b", r"\bedge\s+deploy\w*\b",
        r"\bon[\s-]?device\b", r"\bmobile\b",
        r"\bcontinual\s+learn\w*\b", r"\blifelong\s+learn\w*\b",
    ]),
    ("theory", [
        r"\btheor\w*\b", r"\bconvergence\b", r"\bregret\s+bound\b",
        r"\bsample\s+complex\w*\b", r"\bpac[\s-]?learn\w*\b",
        r"\bgenerali[sz]ation\s+bound\b", r"\bstatistic\w*\s+learn\w*\b",
        r"\binformation[\s-]?theor\w*\b", r"\boptim\w+\s+theor\w*\b",
        r"\bstochastic\s+optim\w*\b", r"\bconvex\s+optim\w*\b",
        r"\bnon[\s-]?convex\b", r"\bkernel\s+method\w*\b",
        r"\breproducing\s+kernel\b", r"\brkhs\b",
        r"\bbayesian\b", r"\bvariational\s+infer\w*\b",
        r"\bmcmc\b", r"\bcausal\w*\b", r"\bcounterfactual\b",
        r"\bfairness\b", r"\brobustness\b", r"\badversarial\b",
        r"\bdifferential\s+privac\w*\b", r"\bprivacy\b",
    ]),
]

# Compile patterns once for performance
_DOMAIN_COMPILED: List[Tuple[str, List[re.Pattern]]] = [
    (domain, [re.compile(p, re.IGNORECASE) for p in patterns])
    for domain, patterns in DOMAIN_KEYWORDS
]


def classify_domain(title: str, abstract: str) -> str:
    """Classify a paper into a domain based on keyword heuristics.

    Returns the first domain that gets at least one keyword match.
    Domains are checked in priority order (specific before general).
    """
    text = f"{title} {abstract}"
    # Score each domain by number of matching patterns
    scores: Dict[str, int] = {}
    for domain, patterns in _DOMAIN_COMPILED:
        count = sum(1 for p in patterns if p.search(text))
        if count > 0:
            scores[domain] = count

    if not scores:
        return "other"

    # Return domain with highest match count; ties broken by priority order
    best_domain = max(scores, key=lambda d: scores[d])
    return best_domain


# ---------------------------------------------------------------------------
# Code platform detection
# ---------------------------------------------------------------------------

def detect_code_platform(code_url: str) -> str:
    """Detect hosting platform from code URL."""
    url_lower = code_url.lower()
    if "github.com" in url_lower:
        return "github"
    if "gitlab.com" in url_lower or "gitlab." in url_lower:
        return "gitlab"
    if "huggingface.co" in url_lower:
        return "huggingface"
    if "bitbucket.org" in url_lower:
        return "bitbucket"
    return "other"


# ---------------------------------------------------------------------------
# Difficulty scoring skeleton
# ---------------------------------------------------------------------------

# Weight schema for difficulty scoring
DIFFICULTY_WEIGHTS = {
    "framework_complexity": 0.20,
    "dependency_count": 0.15,
    "dataset_requirements": 0.20,
    "hardware_requirements": 0.20,
    "code_quality": 0.10,
    "reproduction_time": 0.15,
}


def compute_difficulty(repo_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute difficulty score from repo analysis results.

    Each sub-score is 0.0-1.0, weighted by DIFFICULTY_WEIGHTS.
    Total difficulty is the weighted sum (0.0 = trivial, 1.0 = extremely hard).

    Args:
        repo_analysis: dict with keys matching DIFFICULTY_WEIGHTS, each
            containing a float score in [0, 1] and optional metadata.

    Returns:
        Dict with per-dimension scores, weights, and total, or None if
        repo_analysis is insufficient.
    """
    if not repo_analysis:
        return None

    sub_scores: Dict[str, float] = {}
    for dimension in DIFFICULTY_WEIGHTS:
        value = repo_analysis.get(dimension)
        if value is None:
            return None  # Incomplete analysis -- cannot score
        sub_scores[dimension] = float(value)

    total = sum(
        sub_scores[dim] * weight
        for dim, weight in DIFFICULTY_WEIGHTS.items()
    )

    return {
        "total": round(total, 4),
        "dimensions": {
            dim: {
                "score": round(sub_scores[dim], 4),
                "weight": weight,
            }
            for dim, weight in DIFFICULTY_WEIGHTS.items()
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_venue(venue_id: str) -> List[Dict[str, Any]]:
    """Load a raw venue JSON file."""
    path = DATA_DIR / f"{venue_id}.json"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_eligible(paper: Dict[str, Any]) -> bool:
    """Check if a paper passes the inclusion filter."""
    return (
        paper.get("_repo_valid") is True
        and bool(paper.get("code_url"))
        and bool(paper.get("code_commit"))
    )


def build_entry(paper: Dict[str, Any], paper_id: str) -> Dict[str, Any]:
    """Build a clean final-schema entry from a raw paper dict."""
    code_url = paper.get("code_url", "")
    return {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "venue": paper.get("venue", ""),
        "year": paper.get("year", 2025),
        "code_url": code_url,
        "code_commit": paper.get("code_commit", ""),
        "arxiv_id": paper.get("arxiv_id", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "paper_url": paper.get("paper_url", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "venue_track": paper.get("venue_track", ""),
        "code_platform": detect_code_platform(code_url),
        "domain": classify_domain(
            paper.get("title", ""),
            paper.get("abstract", ""),
        ),
        "difficulty": None,
        "tags": [],
    }


def finalize() -> List[Dict[str, Any]]:
    """Run the full finalization pipeline."""
    # Collect eligible papers from all venues
    eligible: List[Dict[str, Any]] = []
    for venue_id in VENUE_IDS:
        papers = load_venue(venue_id)
        for p in papers:
            if is_eligible(p):
                eligible.append(p)

    # Sort by venue (uppercase) then title for deterministic ID assignment
    eligible.sort(key=lambda p: (p.get("venue", "").upper(), p.get("title", "")))

    # Assign paper IDs and build final entries
    dataset: List[Dict[str, Any]] = []
    for i, paper in enumerate(eligible, start=1):
        paper_id = f"bamboo-{i:05d}"
        entry = build_entry(paper, paper_id)
        dataset.append(entry)

    return dataset


def print_summary(dataset: List[Dict[str, Any]]) -> None:
    """Print a summary of the finalized dataset."""
    print(f"\n{'='*60}")
    print(f"BAMBOO Dataset Finalized: {len(dataset)} papers")
    print(f"{'='*60}")

    # Venue breakdown
    venue_counts = Counter(e["venue"] for e in dataset)
    print(f"\nVenue breakdown ({len(venue_counts)} venues):")
    for venue, count in sorted(venue_counts.items()):
        print(f"  {venue:>10s}: {count:>5d}")

    # Domain distribution
    domain_counts = Counter(e["domain"] for e in dataset)
    print(f"\nDomain distribution ({len(domain_counts)} domains):")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(dataset) if dataset else 0
        print(f"  {domain:>25s}: {count:>5d}  ({pct:5.1f}%)")

    # Platform breakdown
    platform_counts = Counter(e["code_platform"] for e in dataset)
    print(f"\nCode platform breakdown:")
    for platform, count in sorted(platform_counts.items(), key=lambda x: -x[1]):
        print(f"  {platform:>15s}: {count:>5d}")

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"{'='*60}\n")


def main() -> None:
    dataset = finalize()

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print_summary(dataset)


if __name__ == "__main__":
    main()
