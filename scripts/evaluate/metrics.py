"""Aggregate metric computation for BAMBOO evaluation results.

Each function takes a list of per-paper evaluated result dicts (the output
of evaluate.py's per-paper evaluation, NOT the raw agent output).

Expected per-paper dict shape (fields used by these functions):
{
    "paper_id": str,
    "pass4": {
        "l1_build": {"status": str},
        "l2_run": {"status": str},
        "l3_reproduce": {"status": str, "reproduction_score": float|None,
                         "claim_results": [...]},
        "l4_cross": {"status": str},
        "overall_level": int
    },
    "claim_recall": float|None,
    "claim_precision": float|None,
    "relative_deviation": float|None,
    "failure_attribution": {"likely_paper_issue": bool, ...} | None,
    "barriers": [...],
    "resource_usage": {"total_time_ms": int, "llm_cost_usd": float|None, ...}
}
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

# Sentinel for missing values -- lets us distinguish "not reported" from 0.
_MISSING = object()


def _safe_mean(values: list[float]) -> float:
    """Mean that returns 0.0 for empty lists."""
    return statistics.mean(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _level_passes(status: str, *, include_partial: bool = False) -> bool:
    if status == "pass":
        return True
    if include_partial and status == "partial":
        return True
    return False


# ---------------------------------------------------------------------------
# Core pass^4 rates
# ---------------------------------------------------------------------------

def build_rate(results: list[dict]) -> float:
    """% papers reaching L1 (build pass)."""
    if not results:
        return 0.0
    passing = sum(
        1 for r in results
        if _level_passes(r["pass4"]["l1_build"]["status"])
    )
    return passing / len(results)


def run_rate(results: list[dict]) -> float:
    """% papers reaching L2 (run pass)."""
    if not results:
        return 0.0
    passing = sum(
        1 for r in results
        if _level_passes(r["pass4"]["l2_run"]["status"])
    )
    return passing / len(results)


def reproduce_rate(results: list[dict]) -> float:
    """% papers reaching L3 full pass."""
    if not results:
        return 0.0
    passing = sum(
        1 for r in results
        if r["pass4"]["l3_reproduce"]["status"] == "pass"
    )
    return passing / len(results)


def reproduce_rate_partial(results: list[dict]) -> float:
    """% papers reaching L3 partial or full."""
    if not results:
        return 0.0
    passing = sum(
        1 for r in results
        if _level_passes(r["pass4"]["l3_reproduce"]["status"],
                         include_partial=True)
    )
    return passing / len(results)


# ---------------------------------------------------------------------------
# Claim-level metrics
# ---------------------------------------------------------------------------

def claim_coverage(results: list[dict]) -> float:
    """Mean claim recall across all papers."""
    vals = [r["claim_recall"] for r in results
            if r.get("claim_recall") is not None]
    return _safe_mean(vals)


def mean_reproduction_score(results: list[dict]) -> float:
    """Mean (claims matched / claims attempted)."""
    vals = []
    for r in results:
        l3 = r["pass4"]["l3_reproduce"]
        score = l3.get("reproduction_score")
        if score is not None:
            vals.append(score)
    return _safe_mean(vals)


def mean_relative_deviation(results: list[dict]) -> float:
    """Mean |actual-expected|/|expected| across all papers."""
    vals = [r["relative_deviation"] for r in results
            if r.get("relative_deviation") is not None]
    return _safe_mean(vals)


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def barrier_distribution(results: list[dict]) -> dict[str, float]:
    """% failures at each barrier level.

    Only considers papers that did NOT reach L3 full pass.
    """
    failed = [r for r in results
              if r["pass4"]["l3_reproduce"]["status"] not in ("pass",)]
    if not failed:
        return {}
    counter: Counter = Counter()
    for r in failed:
        barriers = r.get("barriers") or []
        if barriers:
            # Use the first (primary) barrier
            counter[barriers[0]["level"]] += 1
        else:
            # Determine implicit barrier from pass^4 levels
            p4 = r["pass4"]
            if not _level_passes(p4["l1_build"]["status"]):
                counter["L1_environment"] += 1
            elif not _level_passes(p4["l2_run"]["status"]):
                counter["L2_build"] += 1
            else:
                counter["L3_framework"] += 1
    total = len(failed)
    return {k: v / total for k, v in sorted(counter.items())}


def paper_issue_rate(results: list[dict]) -> float:
    """% failures attributed to paper's code quality."""
    failed = [r for r in results
              if r["pass4"]["l3_reproduce"]["status"] not in ("pass",)]
    if not failed:
        return 0.0
    attributed = sum(
        1 for r in failed
        if (r.get("failure_attribution") or {}).get("likely_paper_issue", False)
    )
    return attributed / len(failed)


# ---------------------------------------------------------------------------
# Resource metrics
# ---------------------------------------------------------------------------

def cost_summary(results: list[dict]) -> dict[str, float]:
    """Mean/median/total LLM cost."""
    costs = [r["resource_usage"].get("llm_cost_usd")
             for r in results
             if r.get("resource_usage", {}).get("llm_cost_usd") is not None]
    if not costs:
        return {"mean": 0.0, "median": 0.0, "total": 0.0, "count": 0}
    return {
        "mean": statistics.mean(costs),
        "median": statistics.median(costs),
        "total": sum(costs),
        "count": len(costs),
    }


def time_summary(results: list[dict]) -> dict[str, float]:
    """Mean/median/total wall-clock time (seconds)."""
    times_s = [r["resource_usage"]["total_time_ms"] / 1000.0
               for r in results
               if r.get("resource_usage", {}).get("total_time_ms") is not None]
    if not times_s:
        return {"mean_s": 0.0, "median_s": 0.0, "total_s": 0.0, "count": 0}
    return {
        "mean_s": statistics.mean(times_s),
        "median_s": statistics.median(times_s),
        "total_s": sum(times_s),
        "count": len(times_s),
    }


# ---------------------------------------------------------------------------
# Stratified metrics
# ---------------------------------------------------------------------------

def _all_metrics(results: list[dict]) -> dict[str, Any]:
    """Compute all aggregate metrics for a set of results."""
    return {
        "count": len(results),
        "build_rate": build_rate(results),
        "run_rate": run_rate(results),
        "reproduce_rate": reproduce_rate(results),
        "reproduce_rate_partial": reproduce_rate_partial(results),
        "claim_coverage": claim_coverage(results),
        "mean_reproduction_score": mean_reproduction_score(results),
        "mean_relative_deviation": mean_relative_deviation(results),
        "barrier_distribution": barrier_distribution(results),
        "paper_issue_rate": paper_issue_rate(results),
        "cost": cost_summary(results),
        "time": time_summary(results),
    }


def stratified_metrics(
    results: list[dict],
    dataset: dict[str, dict],
    by: str = "venue",
) -> dict[str, dict]:
    """All metrics stratified by venue/tier/domain.

    Args:
        results: evaluated per-paper result dicts
        dataset: mapping paper_id -> paper entry (from bamboo_final.json)
        by: one of 'venue', 'tier', 'domain'

    Returns:
        {stratum_name: {metric: value, ...}, ...}
    """
    buckets: dict[str, list[dict]] = {}
    for r in results:
        pid = r["paper_id"]
        entry = dataset.get(pid, {})
        if by == "venue":
            key = entry.get("venue", "unknown")
        elif by == "tier":
            diff = entry.get("difficulty") or {}
            tier = diff.get("tier")
            key = f"tier_{tier}" if tier is not None else "unknown"
        elif by == "domain":
            key = entry.get("domain", "unknown")
        else:
            raise ValueError(f"Unknown stratification: {by!r}")
        buckets.setdefault(key, []).append(r)

    return {k: _all_metrics(v) for k, v in sorted(buckets.items())}
