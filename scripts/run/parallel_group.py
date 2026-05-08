#!/usr/bin/env python3
"""Run a group of BAMBOO papers in parallel through panda + auto-reflection.

Usage:
    python -m scripts.run.parallel_group \
        --model deep-deepseek-v4-pro \
        --timeout 1800 \
        bamboo-00003 bamboo-00021 bamboo-00076

Each paper is launched as its own runner.py subprocess. PANDA_CROSS_RUN_LEARNING=1
is exported so panda's session-end reflection writes to the shared wiki at
~/.local/share/panda/knowledge. The independent judge (gold-vs-claimed answer
extraction) runs at the end of each subprocess. After all subprocesses exit,
this script prints:

  - per-paper exit code, wall time, overall_level (from result.json)
  - judge gaps: how many of the paper's claims the judge could extract
  - wiki diff: which new articles / experiences / reflections appeared
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BAMBOO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = BAMBOO_ROOT / "data" / "results"
WIKI_ROOT = Path.home() / ".local" / "share" / "panda" / "knowledge"


def snapshot_wiki() -> set[str]:
    """Set of relative paths under wiki, for diffing before vs after."""
    if not WIKI_ROOT.is_dir():
        return set()
    return {
        str(p.relative_to(WIKI_ROOT))
        for p in WIKI_ROOT.rglob("*.md")
        if p.is_file()
    }


def launch_one(paper_id: str, model: str, timeout_s: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PANDA_CROSS_RUN_LEARNING"] = "1"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "scripts.run.runner",
            "--agents", "panda",
            "--model", model,
            "--papers", paper_id,
            "--timeout", str(timeout_s),
            "--prompt-tier", "guided",
        ],
        stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=BAMBOO_ROOT, env=env,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]  (keep handle alive)
    return proc


def read_result(agent_id: str, paper_id: str) -> dict:
    p = RESULTS_DIR / agent_id / f"{paper_id}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def read_judge(agent_id: str, paper_id: str) -> dict:
    p = RESULTS_DIR / agent_id / "judge" / f"{paper_id}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model profile from configs/models.json")
    ap.add_argument("--timeout", type=int, default=1800, help="Per-paper timeout in seconds")
    ap.add_argument("--log-dir", type=Path,
                    default=Path(os.environ.get("BAMBOO_LOG_DIR", "/tmp/bamboo-group-logs")),
                    help="Where to write per-paper subprocess logs")
    ap.add_argument("papers", nargs="+", help="Paper IDs (e.g. bamboo-00003 bamboo-00021 ...)")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    agent_id = f"panda-deep-{args.model}" if not args.model.startswith("panda-") else args.model
    # The actual agent_id is built inside runner.py from the model profile;
    # we recover it later from result paths. For now, scan results dir at end.

    wiki_before = snapshot_wiki()
    print(f"[group] wiki snapshot: {len(wiki_before)} files before")
    print(f"[group] launching {len(args.papers)} papers in parallel: {args.papers}")
    print(f"[group] timeout={args.timeout}s per paper, model={args.model}")
    print(f"[group] logs: {args.log_dir}/<paper_id>.log")
    sys.stdout.flush()

    start = time.time()
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for pid in args.papers:
        log_path = args.log_dir / f"{pid}.log"
        proc = launch_one(pid, args.model, args.timeout, log_path)
        procs.append((pid, proc, log_path))
        print(f"[group] launched {pid} pid={proc.pid} log={log_path}")
        sys.stdout.flush()

    # Wait for all to finish.
    results: list[dict] = []
    for pid, proc, log_path in procs:
        proc.wait()
        proc._log_fh.close()  # type: ignore[attr-defined]
        rc = proc.returncode
        elapsed = time.time() - start
        print(f"[group] {pid} exited rc={rc} (t+{elapsed:.0f}s)")
        results.append({"paper_id": pid, "exit_code": rc, "log": str(log_path)})

    total = time.time() - start
    print(f"\n[group] all done in {total:.0f}s")

    # Auto-detect agent_id by scanning results dir for any matching paper.
    agent_dirs = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()] if RESULTS_DIR.is_dir() else []
    aid_match = next((d for d in agent_dirs if "panda" in d and args.model.split("-")[-1] in d), None)
    if aid_match is None and agent_dirs:
        aid_match = agent_dirs[0]
    print(f"[group] agent_id={aid_match}")

    # Per-paper summary.
    print(f"\n{'='*70}")
    print(f"  GROUP SUMMARY")
    print(f"{'='*70}")
    for r in results:
        pid = r["paper_id"]
        result = read_result(aid_match, pid) if aid_match else {}
        level = result.get("pass4", {}).get("overall_level", "?")
        wall = result.get("resource_usage", {}).get("total_time_ms", 0)
        judge = read_judge(aid_match, pid) if aid_match else {}
        claim_results = judge.get("claim_results", []) if judge else []
        extracted = sum(1 for cr in claim_results if cr.get("actual_value") is not None)
        total_claims = len(claim_results)
        print(f"  {pid}: rc={r['exit_code']} L{level} wall={wall/1000:.0f}s judge={extracted}/{total_claims}")

    # Wiki diff.
    wiki_after = snapshot_wiki()
    new_files = sorted(wiki_after - wiki_before)
    print(f"\n[wiki] {len(wiki_after)} files now (+{len(new_files)} new)")
    for f in new_files:
        print(f"  + {f}")

    # Dump a top-of-judge sample per paper so we can eyeball gaps.
    print(f"\n{'='*70}")
    print(f"  JUDGE GAPS (first 5 claims per paper)")
    print(f"{'='*70}")
    for r in results:
        pid = r["paper_id"]
        judge = read_judge(aid_match, pid) if aid_match else {}
        crs = judge.get("claim_results", [])[:5]
        if not crs:
            print(f"\n  {pid}: no judge data")
            continue
        print(f"\n  {pid}:")
        for cr in crs:
            cid = cr.get("claim_id", "?")
            actual = cr.get("actual_value")
            target = cr.get("target_value", cr.get("expected_value", "?"))
            print(f"    {cid}: agent={actual}  gold={target}")

    print(f"\n[group] done. ts={datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
