"""BAMBOO evaluation harness.

Takes an agent's result JSON files and the ground truth dataset,
computes per-paper evaluation and aggregate metrics.

Usage:
    python -m scripts.evaluate.evaluate \
        --results-dir data/results/{agent_id}/ \
        --dataset data/bamboo_final.json \
        --output data/results/{agent_id}/report.json \
        [--by venue|tier|domain]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import metrics


# ---------------------------------------------------------------------------
# Claim matching
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Lower-case, strip, collapse whitespace."""
    return " ".join(s.lower().split())


def _fuzzy_claim_match(
    agent_claim: dict,
    gt_claim: dict,
    *,
    threshold: float = 0.5,
) -> bool:
    """Heuristic fuzzy match when claim_id is not available.

    Checks overlap of description words + exact metric_name match.
    """
    if agent_claim.get("metric_name") and gt_claim.get("metric_name"):
        if _normalize(agent_claim["metric_name"]) != _normalize(gt_claim["metric_name"]):
            return False

    desc_a = set(_normalize(agent_claim.get("description", "")).split())
    desc_b = set(_normalize(gt_claim.get("description", "")).split())
    if not desc_a or not desc_b:
        return False
    overlap = len(desc_a & desc_b) / max(len(desc_a), len(desc_b))
    return overlap >= threshold


def match_claims(
    agent_claims: list[dict],
    gt_claims: list[dict],
) -> list[dict]:
    """Match agent claims to ground truth claims.

    Returns a list of match dicts:
    {
        "claim_id": str,           # ground truth claim_id
        "gt_claim": dict,
        "agent_claim": dict|None,  # None if unmatched
        "matched_by": "claim_id"|"fuzzy"|None
    }
    """
    # Index agent claims by claim_id for fast exact matching
    agent_by_id: dict[str, dict] = {}
    agent_unmatched: list[dict] = []
    for ac in agent_claims:
        cid = ac.get("claim_id")
        if cid:
            agent_by_id[cid] = ac
        else:
            agent_unmatched.append(ac)

    matches: list[dict] = []
    used_agent_indices: set[int] = set()

    for gt in gt_claims:
        cid = gt["claim_id"]
        # Try exact claim_id match first
        if cid in agent_by_id:
            matches.append({
                "claim_id": cid,
                "gt_claim": gt,
                "agent_claim": agent_by_id[cid],
                "matched_by": "claim_id",
            })
            continue

        # Try fuzzy match against remaining unmatched agent claims
        best_idx = None
        for i, ac in enumerate(agent_unmatched):
            if i in used_agent_indices:
                continue
            if _fuzzy_claim_match(ac, gt):
                best_idx = i
                break

        if best_idx is not None:
            used_agent_indices.add(best_idx)
            matches.append({
                "claim_id": cid,
                "gt_claim": gt,
                "agent_claim": agent_unmatched[best_idx],
                "matched_by": "fuzzy",
            })
        else:
            matches.append({
                "claim_id": cid,
                "gt_claim": gt,
                "agent_claim": None,
                "matched_by": None,
            })

    return matches


# ---------------------------------------------------------------------------
# Per-claim evaluation
# ---------------------------------------------------------------------------

def evaluate_claim(
    gt_claim: dict,
    agent_claim: dict | None,
) -> dict:
    """Evaluate a single claim against ground truth.

    Returns:
        {
            "claim_id": str,
            "expected_value": float,
            "actual_value": float|None,
            "relative_deviation": float|None,
            "pass": bool,
            "category": str,
        }
    """
    cid = gt_claim["claim_id"]
    expected = gt_claim["metric_value"]
    tolerance = gt_claim.get("tolerance", 0.05)
    category = gt_claim.get("category", "main")

    if agent_claim is None:
        return {
            "claim_id": cid,
            "expected_value": expected,
            "actual_value": None,
            "relative_deviation": None,
            "pass": False,
            "category": category,
        }

    # Agent may report reproduced value under various field names
    actual = (
        agent_claim.get("actual_value")
        or agent_claim.get("metric_value_reproduced")
        or agent_claim.get("metric_value_actual")
    )
    if actual is None:
        return {
            "claim_id": cid,
            "expected_value": expected,
            "actual_value": None,
            "relative_deviation": None,
            "pass": False,
            "category": category,
        }

    try:
        actual = float(actual)
    except (TypeError, ValueError):
        return {
            "claim_id": cid,
            "expected_value": expected,
            "actual_value": None,
            "relative_deviation": None,
            "pass": False,
            "category": category,
        }

    # Compute relative deviation
    if abs(expected) > 1e-12:
        rel_dev = abs(actual - expected) / abs(expected)
    else:
        # Avoid division by zero -- use absolute difference
        rel_dev = abs(actual - expected)

    passed = rel_dev <= tolerance

    return {
        "claim_id": cid,
        "expected_value": expected,
        "actual_value": actual,
        "relative_deviation": rel_dev,
        "pass": passed,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Per-paper evaluation
# ---------------------------------------------------------------------------

def evaluate_paper(
    result: dict,
    gt_entry: dict,
    judge_claims: list[dict] | None = None,
) -> dict:
    """Evaluate a single agent result against ground truth.

    Args:
        result: agent output per result.schema.json
        gt_entry: paper entry from bamboo_final.json (with ground_truth_claims)
        judge_claims: Independent judge's extracted claim results (preferred over
                      agent self-report). Each entry has claim_id + actual_value.

    Returns:
        Enriched result dict with evaluation metrics attached.
    """
    gt_claims = gt_entry.get("ground_truth_claims") or []
    # Filter to non-baseline for primary scoring
    gt_scoring = [c for c in gt_claims if c.get("category", "main") != "baseline"]
    gt_all = gt_claims  # keep all for recall computation

    # Use judge results if available (independent extraction),
    # otherwise fall back to agent self-report (less trustworthy)
    if judge_claims is not None:
        agent_claims = judge_claims
    else:
        agent_claims = (
            result.get("pass4", {}).get("l3_reproduce", {}).get("claim_results")
            or result.get("claims_extracted")
            or []
        )

    # Match claims
    matches = match_claims(agent_claims, gt_scoring)

    # Evaluate each matched claim
    claim_evals = [evaluate_claim(m["gt_claim"], m["agent_claim"]) for m in matches]

    # Compute L3 status from claim evaluations
    non_baseline_evals = [ce for ce in claim_evals if ce["category"] != "baseline"]
    if non_baseline_evals:
        pass_count = sum(1 for ce in non_baseline_evals if ce["pass"])
        pass_frac = pass_count / len(non_baseline_evals)
    else:
        pass_frac = 0.0

    # Determine L3 status from our own evaluation
    if non_baseline_evals and pass_frac >= 0.8:
        l3_status = "pass"
    elif non_baseline_evals and pass_frac >= 0.5:
        l3_status = "partial"
    elif not gt_scoring:
        # No ground truth claims -- fall back to agent's self-report
        l3_status = result.get("pass4", {}).get("l3_reproduce", {}).get("status", "skip")
    else:
        l3_status = "fail"

    # Build evaluated pass4
    agent_p4 = result.get("pass4", {})
    evaluated_p4 = {
        "l1_build": agent_p4.get("l1_build", {"status": "fail"}),
        "l2_run": agent_p4.get("l2_run", {"status": "fail"}),
        "l3_reproduce": {
            "status": l3_status,
            "claim_results": claim_evals,
            "reproduction_score": pass_frac if non_baseline_evals else None,
        },
        "l4_cross": agent_p4.get("l4_cross", {"status": "skip"}),
        "overall_level": _compute_overall_level(
            agent_p4.get("l1_build", {}).get("status", "fail"),
            agent_p4.get("l2_run", {}).get("status", "fail"),
            l3_status,
            agent_p4.get("l4_cross", {}).get("status", "skip"),
        ),
    }

    # Claim recall: how many GT claims did the agent find at all?
    # (matched agent_claim is not None)
    matched_count = sum(1 for m in match_claims(agent_claims, gt_all) if m["agent_claim"] is not None)
    claim_recall = matched_count / len(gt_all) if gt_all else None

    # Claim precision: how many agent claims map to a GT claim?
    if agent_claims:
        # Reverse match: for each agent claim, check if it matches any GT claim
        reverse_matches = match_claims(gt_all, agent_claims)
        valid_agent = sum(1 for m in reverse_matches if m["agent_claim"] is not None)
        claim_precision = valid_agent / len(agent_claims) if agent_claims else None
    else:
        claim_precision = None

    # Mean relative deviation for this paper
    deviations = [ce["relative_deviation"] for ce in claim_evals
                  if ce["relative_deviation"] is not None]
    mean_dev = (sum(deviations) / len(deviations)) if deviations else None

    return {
        "paper_id": result.get("paper_id", gt_entry.get("paper_id")),
        "agent_id": result.get("agent_id"),
        "timestamp": result.get("timestamp"),
        "pass4": evaluated_p4,
        "claim_recall": claim_recall,
        "claim_precision": claim_precision,
        "relative_deviation": mean_dev,
        "barriers": result.get("barriers"),
        "failure_attribution": result.get("failure_attribution"),
        "resource_usage": result.get("resource_usage", {"total_time_ms": 0}),
    }


def _compute_overall_level(
    l1: str, l2: str, l3: str, l4: str,
) -> int:
    """Highest level passed (0-4)."""
    level = 0
    if l1 == "pass":
        level = 1
    else:
        return level
    if l2 == "pass":
        level = 2
    else:
        return level
    if l3 in ("pass",):
        level = 3
    else:
        return level
    if l4 == "pass":
        level = 4
    return level


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    evaluated: list[dict],
    dataset_index: dict[str, dict],
    stratify_by: str | None = None,
) -> dict[str, Any]:
    """Generate full evaluation report."""
    report: dict[str, Any] = {
        "summary": {
            "total_papers": len(evaluated),
            **metrics._all_metrics(evaluated),
        },
        "per_paper": evaluated,
    }

    if stratify_by:
        report["stratified"] = {
            stratify_by: metrics.stratified_metrics(
                evaluated, dataset_index, by=stratify_by
            ),
        }
    else:
        # Include all three stratifications by default
        for by in ("venue", "tier", "domain"):
            report.setdefault("stratified", {})[by] = metrics.stratified_metrics(
                evaluated, dataset_index, by=by
            )

    return report


def print_summary(report: dict) -> None:
    """Print human-readable summary to stdout."""
    s = report["summary"]
    n = s["total_papers"]

    print("=" * 64)
    print("  BAMBOO Evaluation Report")
    print("=" * 64)
    print()
    print(f"  Papers evaluated: {n}")
    print()

    print("  pass^4 Rates:")
    print(f"    L1 Build Rate:           {s['build_rate']:.1%}")
    print(f"    L2 Run Rate:             {s['run_rate']:.1%}")
    print(f"    L3 Reproduce Rate:       {s['reproduce_rate']:.1%}")
    print(f"    L3 Reproduce (partial+): {s['reproduce_rate_partial']:.1%}")
    print()

    print("  Claim Metrics:")
    print(f"    Claim Coverage (recall): {s['claim_coverage']:.3f}")
    print(f"    Mean Reproduction Score: {s['mean_reproduction_score']:.3f}")
    print(f"    Mean Relative Deviation: {s['mean_relative_deviation']:.4f}")
    print()

    print("  Failure Analysis:")
    bd = s.get("barrier_distribution", {})
    if bd:
        for barrier, frac in bd.items():
            print(f"    {barrier}: {frac:.1%}")
    else:
        print("    (no failures)")
    print(f"    Paper Issue Rate: {s['paper_issue_rate']:.1%}")
    print()

    cost = s["cost"]
    time = s["time"]
    print("  Resources:")
    if cost["count"] > 0:
        print(f"    LLM Cost: mean=${cost['mean']:.2f}  "
              f"median=${cost['median']:.2f}  "
              f"total=${cost['total']:.2f}  ({cost['count']} papers)")
    else:
        print("    LLM Cost: (not reported)")
    if time["count"] > 0:
        print(f"    Wall Time: mean={time['mean_s']:.0f}s  "
              f"median={time['median_s']:.0f}s  "
              f"total={time['total_s']:.0f}s  ({time['count']} papers)")
    else:
        print("    Wall Time: (not reported)")
    print()

    # Stratified summary
    stratified = report.get("stratified", {})
    for by_name, strata in stratified.items():
        print(f"  Stratified by {by_name}:")
        print(f"    {'Stratum':<20s} {'N':>4s} {'BR':>6s} {'RR':>6s} "
              f"{'ReR':>6s} {'ReR-P':>6s} {'CC':>6s} {'MRS':>6s}")
        print(f"    {'-'*20} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
        for stratum, sm in sorted(strata.items()):
            print(f"    {stratum:<20s} {sm['count']:>4d} "
                  f"{sm['build_rate']:>5.1%} "
                  f"{sm['run_rate']:>5.1%} "
                  f"{sm['reproduce_rate']:>5.1%} "
                  f"{sm['reproduce_rate_partial']:>5.1%} "
                  f"{sm['claim_coverage']:>5.3f} "
                  f"{sm['mean_reproduction_score']:>5.3f}")
        print()

    print("=" * 64)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> dict[str, dict]:
    """Load bamboo_final.json, return {paper_id: entry}."""
    with open(path, "r") as f:
        data = json.load(f)
    # Could be a list or a dict
    if isinstance(data, list):
        return {entry["paper_id"]: entry for entry in data}
    elif isinstance(data, dict):
        # Already keyed by paper_id, or has a "papers" key
        if "papers" in data:
            return {entry["paper_id"]: entry for entry in data["papers"]}
        return data
    else:
        raise ValueError(f"Unexpected dataset format: {type(data)}")


def load_results(results_dir: str) -> list[dict]:
    """Load all result JSON files from a directory."""
    results = []
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        # Skip report.json itself to avoid circular reads
        if name == "report.json":
            continue
        path = os.path.join(results_dir, name)
        with open(path, "r") as f:
            try:
                result = json.load(f)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping {path}: {e}", file=sys.stderr)
                continue
        # Must have paper_id
        if "paper_id" not in result:
            print(f"WARNING: skipping {path}: no paper_id", file=sys.stderr)
            continue
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BAMBOO evaluation harness",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory containing agent result JSON files",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to bamboo_final.json",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for report JSON",
    )
    parser.add_argument(
        "--by", choices=["venue", "tier", "domain"], default=None,
        help="Stratify by a single dimension (default: all three)",
    )
    args = parser.parse_args()

    # Load data
    dataset_index = load_dataset(args.dataset)
    agent_results = load_results(args.results_dir)

    if not agent_results:
        print("ERROR: no valid result files found in "
              f"{args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(agent_results)} result(s) from {args.results_dir}",
          file=sys.stderr)

    # Load judge results if available
    judge_dir = Path(args.results_dir) / "judge"
    judge_index: dict[str, list[dict]] = {}
    if judge_dir.is_dir():
        for jf in judge_dir.glob("bamboo-*.json"):
            try:
                jdata = json.loads(jf.read_text())
                jpid = jdata.get("paper_id", jf.stem)
                judge_index[jpid] = jdata.get("claim_results", [])
            except (json.JSONDecodeError, OSError):
                pass
        if judge_index:
            print(f"Loaded {len(judge_index)} judge result(s) from {judge_dir}",
                  file=sys.stderr)

    # Evaluate each paper
    evaluated = []
    missing_gt = 0
    for result in agent_results:
        pid = result["paper_id"]
        gt_entry = dataset_index.get(pid)
        if gt_entry is None:
            print(f"WARNING: no ground truth for {pid}, "
                  "using agent self-report only", file=sys.stderr)
            missing_gt += 1
            gt_entry = {"paper_id": pid}

        # Use judge results (independent) over agent self-report
        jclaims = judge_index.get(pid)
        evaluated.append(evaluate_paper(result, gt_entry, judge_claims=jclaims))

    if missing_gt:
        print(f"WARNING: {missing_gt}/{len(agent_results)} papers had no "
              "ground truth entry", file=sys.stderr)

    # Generate report
    report = generate_report(evaluated, dataset_index, stratify_by=args.by)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {args.output}", file=sys.stderr)

    # Print human-readable summary
    print_summary(report)


if __name__ == "__main__":
    main()
