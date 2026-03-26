#!/usr/bin/env python3
"""Compare evaluation reports across agents.

Usage:
    python -m scripts.evaluate.compare \
        --reports data/results/panda/report.json \
                  data/results/claude-code/report.json \
                  data/results/opencode/report.json \
                  data/results/codex/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BAMBOO agent reports")
    parser.add_argument("--reports", nargs="+", required=True, help="Report JSON paths")
    parser.add_argument("--output", help="Output comparison JSON path")
    args = parser.parse_args()

    reports = {}
    for path in args.reports:
        report = load_report(path)
        # Infer agent_id from directory name or first result
        per_paper = report.get("per_paper", [])
        if per_paper:
            agent_id = per_paper[0].get("agent_id", Path(path).parent.name)
        else:
            agent_id = Path(path).parent.name
        reports[agent_id] = report

    if not reports:
        print("No reports loaded", file=sys.stderr)
        sys.exit(1)

    agent_ids = sorted(reports.keys())

    # Header
    print("=" * 72)
    print("  BAMBOO Agent Comparison")
    print("=" * 72)
    print()

    # Summary metrics table
    metrics = [
        ("Papers", "total_papers"),
        ("Build Rate (L1)", "build_rate"),
        ("Run Rate (L2)", "run_rate"),
        ("Reproduce Rate (L3)", "reproduce_rate"),
        ("Reproduce Partial+", "reproduce_rate_partial"),
        ("Claim Coverage", "claim_coverage"),
        ("Mean Repro Score", "mean_reproduction_score"),
        ("Mean Rel Deviation", "mean_relative_deviation"),
    ]

    col_w = 18
    header = f"{'Metric':<24s}" + "".join(f"{aid:>{col_w}s}" for aid in agent_ids)
    print(header)
    print("-" * len(header))

    for label, key in metrics:
        row = f"{label:<24s}"
        for aid in agent_ids:
            val = reports[aid].get("summary", {}).get(key, None)
            if val is None:
                row += f"{'N/A':>{col_w}s}"
            elif isinstance(val, float):
                if key in ("mean_relative_deviation",):
                    row += f"{val:>{col_w}.4f}"
                elif key in ("claim_coverage", "mean_reproduction_score"):
                    row += f"{val:>{col_w}.3f}"
                else:
                    row += f"{val:>{col_w - 1}.1%} "
                    row = row[:-1]  # trim trailing space
            else:
                row += f"{val:>{col_w}}"
        print(row)

    # Cost comparison
    print()
    print("  Resource Usage:")
    for aid in agent_ids:
        cost_info = reports[aid].get("summary", {}).get("cost", {})
        time_info = reports[aid].get("summary", {}).get("time", {})
        cost_str = f"${cost_info.get('total', 0):.2f}" if cost_info.get("count", 0) > 0 else "N/A"
        time_str = f"{time_info.get('total_s', 0):.0f}s" if time_info.get("count", 0) > 0 else "N/A"
        print(f"    {aid:<16s}  cost={cost_str:<10s}  time={time_str}")

    # Per-paper comparison
    print()
    print("  Per-Paper Level Comparison:")
    paper_header = f"{'Paper':<16s}" + "".join(f"{aid:>{col_w}s}" for aid in agent_ids)
    print(f"  {paper_header}")
    print(f"  {'-' * len(paper_header)}")

    # Collect all paper IDs across reports
    all_paper_ids = set()
    paper_levels: dict[str, dict[str, int]] = {}
    for aid in agent_ids:
        for pp in reports[aid].get("per_paper", []):
            pid = pp.get("paper_id", "?")
            all_paper_ids.add(pid)
            paper_levels.setdefault(pid, {})[aid] = pp.get("pass4", {}).get("overall_level", 0)

    for pid in sorted(all_paper_ids):
        row = f"  {pid:<16s}"
        for aid in agent_ids:
            level = paper_levels.get(pid, {}).get(aid, "-")
            row += f"{'L' + str(level) if isinstance(level, int) else level:>{col_w}s}"
        print(row)

    print()
    print("=" * 72)

    # Optionally save comparison JSON
    if args.output:
        comparison = {
            "agents": agent_ids,
            "summary": {
                aid: reports[aid].get("summary", {}) for aid in agent_ids
            },
            "per_paper_levels": paper_levels,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison written to {args.output}")


if __name__ == "__main__":
    main()
