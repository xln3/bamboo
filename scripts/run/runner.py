#!/usr/bin/env python3
"""BAMBOO comparative agent runner.

Runs multiple agents on the same set of papers and collects results
for evaluation by scripts/evaluate/evaluate.py.

Usage:
    # Run panda with glm-5 on specific papers
    python -m scripts.run.runner --agents panda --model glm-5 --papers bamboo-06079

    # Run panda with claude-sonnet on pilot papers
    python -m scripts.run.runner --agents panda --model claude-sonnet

    # Compare two models on the same papers
    python -m scripts.run.runner --agents panda --model glm-5 --papers bamboo-06079
    python -m scripts.run.runner --agents panda --model claude-sonnet --papers bamboo-06079

    # Use a different dataset file
    python -m scripts.run.runner --agents panda --model glm-5 --dataset data/bamboo_curated.json

    # Run on N random papers with claims
    python -m scripts.run.runner --agents panda --model glm-5 --sample 10

    # Dry run (print prompts only)
    python -m scripts.run.runner --agents panda --model glm-5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path for judge import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Project root
BAMBOO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BAMBOO_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
CONFIGS_DIR = BAMBOO_ROOT / "configs"
DEFAULT_DATASET = DATA_DIR / "bamboo_final.json"
MODELS_CONFIG = CONFIGS_DIR / "models.json"
WORKDIR_BASE = Path("/tmp/bamboo")

# Agents
from .agents.panda import PandaAdapter
from .agents.claude_code import ClaudeCodeAdapter
from .agents.opencode import OpenCodeAdapter
from .agents.codex import CodexAdapter
from .agents.base import AgentAdapter, RunResult
from .prompt_builder import build_prompt

AGENT_REGISTRY: dict[str, type[AgentAdapter]] = {
    "panda": PandaAdapter,
    "claude-code": ClaudeCodeAdapter,
    "opencode": OpenCodeAdapter,
    "codex": CodexAdapter,
}

# Default pilot papers: diverse domains, have claims, tier 2
DEFAULT_PILOTS = [
    "bamboo-00003",  # nlp, 30 claims, time series
    "bamboo-00110",  # vision, 9 claims, adversarial defense
    "bamboo-00340",  # nlp, 3 claims, chart coding
    "bamboo-00021",  # tabular, 28 claims, diffusion time series
    "bamboo-00076",  # tabular, 37 claims, anomaly detection
]


CLAIMS_V2_DIR = DATA_DIR / "paper_claims_v2"


def load_model_config(profile_name: str) -> dict[str, Any]:
    """Load a model profile from configs/models.json.

    Returns a dict with keys: provider, model, base_url, api_key,
    plus _profile_name for agent_id generation.
    """
    if not MODELS_CONFIG.exists():
        print(f"ERROR: Model config not found: {MODELS_CONFIG}")
        print(f"  Copy configs/models.example.json → configs/models.json and fill in API keys.")
        sys.exit(1)

    with open(MODELS_CONFIG) as f:
        all_profiles = json.load(f)

    if profile_name not in all_profiles:
        available = ", ".join(sorted(all_profiles.keys()))
        print(f"ERROR: Model profile '{profile_name}' not found.")
        print(f"  Available profiles: {available}")
        print(f"  Edit {MODELS_CONFIG} to add it.")
        sys.exit(1)

    config = all_profiles[profile_name]
    config["_profile_name"] = profile_name
    return config


def list_model_profiles() -> None:
    """Print available model profiles."""
    if not MODELS_CONFIG.exists():
        print(f"No model config found at {MODELS_CONFIG}")
        print(f"Copy configs/models.example.json → configs/models.json and fill in API keys.")
        return

    with open(MODELS_CONFIG) as f:
        profiles = json.load(f)

    print(f"Available model profiles ({MODELS_CONFIG}):\n")
    for name, cfg in sorted(profiles.items()):
        model = cfg.get("model", "?")
        url = cfg.get("base_url", "?")
        has_key = "yes" if cfg.get("api_key") else "NO"
        print(f"  {name:<20s}  model={model:<30s}  url={url}")
        print(f"  {'':20s}  api_key={'***' if has_key == 'yes' else 'MISSING'}")


def load_dataset(dataset_path: Path | None = None) -> dict[str, dict]:
    """Load the BAMBOO dataset keyed by paper_id.

    Overlays paper_claims_v2/ individual claim files onto the dataset,
    preferring v2 claims (more complete, with dataset field) over the
    older claims in the base dataset.
    """
    path = dataset_path or DEFAULT_DATASET
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        index = {p["paper_id"]: p for p in data}
    else:
        index = data

    # Overlay v2 claims
    if CLAIMS_V2_DIR.is_dir():
        upgraded = 0
        for cf in CLAIMS_V2_DIR.glob("bamboo-*.json"):
            pid = cf.stem
            if pid not in index:
                continue
            try:
                claims = json.loads(cf.read_text())
                if isinstance(claims, list) and claims:
                    index[pid]["ground_truth_claims"] = claims
                    upgraded += 1
            except (json.JSONDecodeError, OSError):
                pass
        if upgraded:
            print(f"Overlaid {upgraded} paper_claims_v2 files onto dataset")

    return index


def make_fallback_result(
    paper_id: str,
    agent_id: str,
    run: RunResult,
) -> dict[str, Any]:
    """Create a minimal result JSON when the agent didn't write one."""
    detail = ""
    if run.error:
        detail = run.error
    elif run.exit_code != 0:
        detail = (run.stderr or "")[-500:]
    else:
        detail = "Agent completed but did not write result JSON"

    return {
        "paper_id": paper_id,
        "agent_id": agent_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pass4": {
            "l1_build": {"status": "fail", "detail": detail},
            "l2_run": {"status": "skip"},
            "l3_reproduce": {"status": "skip"},
            "l4_cross": {"status": "skip"},
            "overall_level": 0,
        },
        "barriers": [
            {
                "level": "L1_environment",
                "description": "Agent did not produce result JSON",
                "evidence": detail[:1000],
                "auto_fixable": False,
            }
        ],
        "resource_usage": {
            "total_time_ms": run.wall_time_ms,
        },
    }


def run_single(
    agent: AgentAdapter,
    paper: dict[str, Any],
    timeout_s: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single agent on a single paper. Returns the result dict."""
    paper_id = paper["paper_id"]
    agent_id = agent.agent_id

    workdir = WORKDIR_BASE / agent_id / paper_id
    result_dir = RESULTS_DIR / agent_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{paper_id}.json"

    prompt = build_prompt(paper, agent_id, result_path, workdir)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"[DRY RUN] Agent={agent_id}  Paper={paper_id}")
        print(f"Workdir: {workdir}")
        print(f"Result:  {result_path}")
        print(f"Prompt ({len(prompt)} chars):")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        return {"paper_id": paper_id, "agent_id": agent_id, "dry_run": True}

    print(f"\n{'='*60}")
    print(f"[RUN] Agent={agent_id}  Paper={paper_id}  Timeout={timeout_s}s")
    print(f"  Title: {paper['title'][:70]}")
    print(f"  Repo:  {paper.get('code_url') or 'N/A'}")
    print(f"  Claims: {len(paper.get('ground_truth_claims', []))}")
    sys.stdout.flush()

    # Remove stale result
    if result_path.exists():
        result_path.unlink()

    # Stream logs to files in real time (allows `tail -f` during run)
    log_dir = RESULTS_DIR / agent_id / "logs" / paper_id
    print(f"  Logs:  tail -f {log_dir}/stdout.txt")
    sys.stdout.flush()

    start = time.time()
    run_result = agent.run(prompt, workdir, result_path, timeout_s, log_dir=log_dir)
    elapsed = time.time() - start

    run_result.paper_id = paper_id

    if run_result.result_json:
        result = run_result.result_json
        result.setdefault("paper_id", paper_id)
        result.setdefault("agent_id", agent_id)
        result.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        result.setdefault("resource_usage", {}).setdefault(
            "total_time_ms", run_result.wall_time_ms
        )
        status = "OK (agent wrote result)"
    else:
        result = make_fallback_result(paper_id, agent_id, run_result)
        status = "FALLBACK (agent did not write result)"

    # Write/overwrite the result file
    result_path.write_text(json.dumps(result, indent=2))

    level = result.get("pass4", {}).get("overall_level", "?")
    print(f"  [{status}] exit={run_result.exit_code} "
          f"time={elapsed:.0f}s level=L{level}")
    if run_result.error:
        print(f"  ERROR: {run_result.error[:200]}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BAMBOO comparative agent runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run panda + glm-5 on 12 papers from curated dataset
  python -m scripts.run.runner --agents panda --model glm-5 \\
    --dataset data/bamboo_curated.json \\
    --papers bamboo-06079 bamboo-06080 bamboo-06081

  # List available model profiles
  python -m scripts.run.runner --list-models

  # Dry run to preview prompts
  python -m scripts.run.runner --agents panda --model glm-5 --dry-run
""",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENT_REGISTRY.keys()),
        default=["panda"],
        help="Which agents to run (default: panda)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Model profile name from configs/models.json (e.g. glm-5, claude-sonnet)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model profiles and exit",
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        help="Specific paper IDs to run (default: pilot set)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Run on N randomly selected papers with claims",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per paper in seconds (default: 1800 = 30min)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without running agents",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip independent judge after agent runs",
    )
    parser.add_argument(
        "--judge-model",
        default="opus",
        help="Model for independent judge (default: opus)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to dataset JSON (default: {DEFAULT_DATASET})",
    )
    args = parser.parse_args()

    # List models mode
    if args.list_models:
        list_model_profiles()
        return

    # Load model config
    if not args.model:
        parser.error("--model is required. Use --list-models to see available profiles.")

    model_config = load_model_config(args.model)
    print(f"Model: {args.model} → {model_config['model']} @ {model_config['base_url']}")

    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} papers from {args.dataset}")

    # Select papers
    if args.papers:
        paper_ids = args.papers
    elif args.sample:
        with_claims = [
            pid for pid, p in dataset.items()
            if len(p.get("ground_truth_claims", [])) >= 3
        ]
        paper_ids = random.sample(with_claims, min(args.sample, len(with_claims)))
    else:
        paper_ids = DEFAULT_PILOTS

    papers = []
    for pid in paper_ids:
        if pid not in dataset:
            print(f"WARNING: paper {pid} not in dataset, skipping")
            continue
        papers.append(dataset[pid])

    print(f"Selected {len(papers)} paper(s): {[p['paper_id'] for p in papers]}")

    # Instantiate agents with model config
    agents = [AGENT_REGISTRY[name](model_config=model_config) for name in args.agents]
    print(f"Agents: {[a.agent_id for a in agents]}")
    print(f"Timeout: {args.timeout}s per paper")

    if args.dry_run:
        print("\n*** DRY RUN MODE ***\n")

    # Run matrix: agents × papers, then judge
    summary: list[dict] = []
    total_start = time.time()

    for paper in papers:
        for agent in agents:
            result = run_single(agent, paper, args.timeout, args.dry_run)

            # Run independent judge (unless skipped or dry run)
            judge_found = 0
            if not args.dry_run and not args.skip_judge:
                claims = paper.get("ground_truth_claims", [])
                if claims:
                    from scripts.evaluate.judge import judge_paper

                    paper_id = paper["paper_id"]
                    log_dir = RESULTS_DIR / agent.agent_id / "logs" / paper_id
                    workdir = WORKDIR_BASE / agent.agent_id / paper_id
                    judge_dir = RESULTS_DIR / agent.agent_id / "judge"
                    judge_dir.mkdir(parents=True, exist_ok=True)

                    print(f"  [JUDGE] Evaluating {len(claims)} claims...")
                    sys.stdout.flush()

                    judge_result = judge_paper(
                        paper_id=paper_id,
                        agent_id=agent.agent_id,
                        claims=claims,
                        logs_dir=log_dir,
                        workdir=workdir,
                        model=args.judge_model,
                    )

                    judge_path = judge_dir / f"{paper_id}.json"
                    judge_path.write_text(
                        json.dumps(judge_result.to_dict(), indent=2, ensure_ascii=False)
                    )

                    judge_found = sum(
                        1 for cr in judge_result.claim_results
                        if cr.get("actual_value") is not None
                    )
                    print(f"  [JUDGE] {judge_found}/{len(claims)} values extracted "
                          f"({judge_result.judge_duration_ms}ms)")

            summary.append({
                "paper_id": paper["paper_id"],
                "agent_id": agent.agent_id,
                "overall_level": result.get("pass4", {}).get("overall_level", 0),
                "judge_extracted": judge_found,
            })

    total_elapsed = time.time() - total_start

    # Print summary matrix
    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  BAMBOO Run Summary")
        print(f"  Model: {args.model}")
        print(f"  Total time: {total_elapsed:.0f}s")
        print(f"{'='*60}\n")

        # Build matrix
        agent_ids = [a.agent_id for a in agents]
        header = f"{'Paper':<16s}" + "".join(f"{aid:>20s}" for aid in agent_ids)
        print(header)
        print("-" * len(header))

        by_paper: dict[str, dict[str, int]] = {}
        for s in summary:
            by_paper.setdefault(s["paper_id"], {})[s["agent_id"]] = s["overall_level"]

        for pid in paper_ids:
            if pid not in by_paper:
                continue
            row = f"{pid:<16s}"
            for aid in agent_ids:
                level = by_paper[pid].get(aid, "?")
                row += f"{'L' + str(level):>20s}"
            print(row)

        print()
        print("Next: run evaluation with:")
        for aid in agent_ids:
            print(f"  python -m scripts.evaluate.evaluate "
                  f"--results-dir data/results/{aid}/ "
                  f"--dataset {args.dataset} "
                  f"--output data/results/{aid}/report.json")


if __name__ == "__main__":
    main()
