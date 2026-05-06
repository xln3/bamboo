"""G5 — Baseline-vs-Evolved acceptance gate.

Same paper, same model, two consecutive runs:

    Cold run:  PANDA_KNOWLEDGE_ROOT=$(mktemp -d) + PANDA_CROSS_RUN_LEARNING=1
                (empty wiki/skills/experiences — agent starts from scratch)
    Warm run:  PANDA_KNOWLEDGE_ROOT=<real knowledge dir> + PANDA_CROSS_RUN_LEARNING=1
                (whatever the agent has accumulated to date)

Captured per run: overall_level (pass4), wall_time_ms, turn_count (parsed
from agent stderr), exit_code.

Acceptance gate (warm must satisfy AT LEAST ONE):
  1. outcome_warm > outcome_cold (strict improvement on level)
  2. outcome_warm >= outcome_cold and turn_count_warm <= 0.7 * turn_count_cold
  3. outcome_warm >= outcome_cold and wall_time_warm <= 0.7 * wall_time_cold

If none hold, the gate FAILS — the wiki content has not paid for itself.
The plan calls for ablating G1/G2/G3/G4 to find the regression source.

This module is invoked by `runner.py --baseline-vs-evolved`. It can also be
unit-tested independently of any agent invocation; see test_baseline_vs_evolved.py.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Names preserved across the cold→warm reset. Reusing these is exactly what
# wiki knowledge should let the agent do quickly; reusing cold's source
# edits or trained checkpoints would confound the gate.
_PRESERVE_NAMES = (
    ".venv", "venv",
    "dataset", "datasets", "data",
    "pretrained", "pretrained_weights", "ckpt_pretrained",
)

# Default warm root mirrors panda's compile-time default
# (~/.local/share/panda/knowledge). Override via env var if the user moved it.
DEFAULT_WARM_ROOT = Path.home() / ".local" / "share" / "panda" / "knowledge"

# Speed/turn-count threshold from the cross-run-learning plan.
EFFICIENCY_RATIO = 0.7

TURN_COUNT_RE = re.compile(
    r"Session ended after \d+ messages,\s*(\d+)\s*turns",
)


@dataclass
class RunMetrics:
    """Captured metrics from one cold or warm run."""

    overall_level: int
    turn_count: int
    wall_time_ms: int
    exit_code: int
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_level": self.overall_level,
            "turn_count": self.turn_count,
            "wall_time_ms": self.wall_time_ms,
            "exit_code": self.exit_code,
            "error": self.error,
        }


def extract_metrics(
    result_json: dict[str, Any] | None,
    stderr_text: str,
    wall_time_ms: int,
    exit_code: int,
    error: str | None = None,
) -> RunMetrics:
    """Pull G5 metrics out of a finished run."""
    overall_level = 0
    if result_json:
        overall_level = int(
            result_json.get("pass4", {}).get("overall_level", 0) or 0
        )

    # turn_count comes from the agent's own session-end log line.
    turn_count = 0
    if stderr_text:
        m = TURN_COUNT_RE.search(stderr_text)
        if m:
            turn_count = int(m.group(1))

    return RunMetrics(
        overall_level=overall_level,
        turn_count=turn_count,
        wall_time_ms=wall_time_ms,
        exit_code=exit_code,
        error=error,
    )


def evaluate_gate(cold: RunMetrics, warm: RunMetrics) -> dict[str, Any]:
    """Apply the three acceptance criteria.

    Returns a dict ready to be serialized:
      {"passed": bool, "criteria_met": [...], "rationale": str}
    """
    criteria: list[str] = []
    notes: list[str] = []

    if warm.overall_level > cold.overall_level:
        criteria.append("strict-outcome-improvement")
        notes.append(
            f"warm L{warm.overall_level} > cold L{cold.overall_level}",
        )

    same_or_better_outcome = warm.overall_level >= cold.overall_level

    if (
        same_or_better_outcome
        and cold.turn_count > 0
        and warm.turn_count > 0
        and warm.turn_count <= EFFICIENCY_RATIO * cold.turn_count
    ):
        criteria.append("turn-count-cut")
        notes.append(
            f"turns warm={warm.turn_count} ≤ 0.7×{cold.turn_count}"
            f" (= {EFFICIENCY_RATIO * cold.turn_count:.1f})"
            f" with outcome ≥ cold",
        )

    if (
        same_or_better_outcome
        and cold.wall_time_ms > 0
        and warm.wall_time_ms > 0
        and warm.wall_time_ms <= EFFICIENCY_RATIO * cold.wall_time_ms
    ):
        criteria.append("wall-time-cut")
        notes.append(
            f"wall_time warm={warm.wall_time_ms}ms ≤ 0.7×{cold.wall_time_ms}ms"
            f" with outcome ≥ cold",
        )

    passed = len(criteria) > 0
    if passed:
        rationale = "PASS: " + "; ".join(notes)
    else:
        rationale = (
            f"FAIL: cold L{cold.overall_level}/{cold.turn_count}t/"
            f"{cold.wall_time_ms}ms vs warm L{warm.overall_level}/"
            f"{warm.turn_count}t/{warm.wall_time_ms}ms — wiki did not pay "
            f"for itself; ablate G1/G2/G3/G4 to find the regression source."
        )

    return {
        "passed": passed,
        "criteria_met": criteria,
        "rationale": rationale,
    }


def _reset_workdir_preserving_env(workdir: Path) -> None:
    """Revert source-code edits and remove training artifacts left by the
    cold pass, while preserving the Python venv, downloaded datasets, and
    pretrained weights.

    G5 measures whether the wiki helps a fresh attempt. Reusing cold's
    environment / datasets / pretrained weights is fair — those are what
    wiki should teach the agent to obtain efficiently. Reusing cold's
    source-code edits or its trained checkpoints would let warm inherit
    cold's actual labor, so we git-reset every repo under the workdir and
    git-clean it with the preserve list excluded.
    """
    if not workdir.exists():
        return
    repos: list[Path] = []
    if (workdir / ".git").is_dir():
        repos.append(workdir)
    for child in workdir.iterdir():
        if child.is_dir() and (child / ".git").is_dir():
            repos.append(child)
    for repo in repos:
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=repo, check=False, capture_output=True,
        )
        clean_args = ["git", "clean", "-fdx"]
        for name in _PRESERVE_NAMES:
            clean_args += ["-e", name]
        subprocess.run(clean_args, cwd=repo, check=False, capture_output=True)


def compare_paper(
    agent: Any,  # AgentAdapter
    paper: dict[str, Any],
    timeout_s: int,
    prompt_tier: str,
    warm_root: Path,
    results_dir: Path,
    run_single: Any = None,  # unused; kept for caller-symmetry with runner.run_single
) -> dict[str, Any]:
    """Run cold + warm passes for a single paper and write the comparison JSON.

    The two runs are sequential (not parallel — they share filesystem
    workdirs and we want a clean cold start). We isolate the cold run's
    knowledge into a tempdir so the agent's own reflection writes do not
    pollute the real wiki, then run warm with the persistent root.
    """
    del run_single  # not needed: we drive agent.run directly via _run_with_log_dir
    paper_id = paper["paper_id"]
    print(f"\n{'='*60}\n[BASELINE-vs-EVOLVED] paper={paper_id}\n{'='*60}")

    # Each run needs its own logs dir so we can extract turn_count.
    base_log_dir = results_dir / agent.agent_id / "logs"

    cold_root = Path(tempfile.mkdtemp(prefix=f"panda-cold-{paper_id}-"))
    cold_metrics: RunMetrics
    warm_metrics: RunMetrics
    try:
        # Cold pass: empty knowledge.
        os.environ["PANDA_KNOWLEDGE_ROOT"] = str(cold_root)
        os.environ["PANDA_CROSS_RUN_LEARNING"] = "1"
        cold_log_dir = base_log_dir / paper_id / "cold"
        cold_log_dir.mkdir(parents=True, exist_ok=True)
        cold_result = _run_with_log_dir(
            agent, paper, timeout_s, prompt_tier, cold_log_dir,
        )
        cold_stderr = _read_log(cold_log_dir / "stderr.txt")
        cold_metrics = extract_metrics(
            cold_result.get("result_json"),
            cold_stderr,
            cold_result["wall_time_ms"],
            cold_result["exit_code"],
            cold_result.get("error"),
        )

        # Reset workdir before warm pass: revert cold's source edits and
        # drop its trained checkpoints, but keep the venv / datasets /
        # pretrained weights so warm doesn't waste time re-downloading.
        from .runner import WORKDIR_BASE as _WB
        _reset_workdir_preserving_env(_WB / agent.agent_id / paper_id)

        # Warm pass: real knowledge.
        os.environ["PANDA_KNOWLEDGE_ROOT"] = str(warm_root)
        warm_log_dir = base_log_dir / paper_id / "warm"
        warm_log_dir.mkdir(parents=True, exist_ok=True)
        warm_result = _run_with_log_dir(
            agent, paper, timeout_s, prompt_tier, warm_log_dir,
        )
        warm_stderr = _read_log(warm_log_dir / "stderr.txt")
        warm_metrics = extract_metrics(
            warm_result.get("result_json"),
            warm_stderr,
            warm_result["wall_time_ms"],
            warm_result["exit_code"],
            warm_result.get("error"),
        )
    finally:
        # Always clean cold root, restore env to whatever it was before.
        os.environ.pop("PANDA_KNOWLEDGE_ROOT", None)
        try:
            shutil.rmtree(cold_root)
        except OSError:
            pass

    gate = evaluate_gate(cold_metrics, warm_metrics)
    comparison = {
        "paper_id": paper_id,
        "agent_id": agent.agent_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "warm_root": str(warm_root),
        "cold": cold_metrics.as_dict(),
        "warm": warm_metrics.as_dict(),
        "gate": gate,
    }

    out_dir = results_dir / agent.agent_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline-vs-evolved-{paper_id}.json"
    out_path.write_text(json.dumps(comparison, indent=2))
    print(f"\n[GATE] {gate['rationale']}")
    print(f"[GATE] wrote {out_path}")
    return comparison


def _read_log(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _run_with_log_dir(
    agent: Any,
    paper: dict[str, Any],
    timeout_s: int,
    prompt_tier: str,
    log_dir: Path,
) -> dict[str, Any]:
    """Run the agent with a custom per-pass log_dir.

    runner.run_single computes log_dir from agent_id+paper_id, so it cannot
    distinguish cold from warm. This helper invokes the agent directly with
    the supplied log_dir and returns a dict shaped like run_single's.
    """
    from .runner import (
        WORKDIR_BASE,
        RESULTS_DIR,
        make_fallback_result,
    )
    from .prompt_builder import build_prompt

    paper_id = paper["paper_id"]
    workdir = WORKDIR_BASE / agent.agent_id / paper_id
    workdir.mkdir(parents=True, exist_ok=True)
    result_dir = RESULTS_DIR / agent.agent_id
    result_dir.mkdir(parents=True, exist_ok=True)
    # Each pass gets its own result.json so the warm pass does not read
    # the cold pass's pass4 by accident.
    result_path = log_dir / "result.json"
    if result_path.exists():
        result_path.unlink()
    prompt = build_prompt(
        paper, agent.agent_id, result_path, workdir, timeout_s,
        tier=prompt_tier,
    )

    print(
        f"  [pass={log_dir.name}] tail -f {log_dir}/stdout.txt",
        flush=True,
    )
    start = time.time()
    run_result = agent.run(
        prompt, workdir, result_path, timeout_s, log_dir=log_dir,
    )
    elapsed = time.time() - start

    if run_result.result_json:
        result = run_result.result_json
        result.setdefault("paper_id", paper_id)
        result.setdefault("agent_id", agent.agent_id)
    else:
        result = make_fallback_result(paper_id, agent.agent_id, run_result)

    print(
        f"  [pass={log_dir.name}] exit={run_result.exit_code} "
        f"time={elapsed:.0f}s level=L"
        f"{result.get('pass4', {}).get('overall_level', '?')}",
        flush=True,
    )

    return {
        "result_json": result,
        "wall_time_ms": run_result.wall_time_ms,
        "exit_code": run_result.exit_code,
        "error": run_result.error,
    }
