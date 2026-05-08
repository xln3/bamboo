#!/usr/bin/env python3
"""Sliding-window batch runner: keep N papers in flight at once.

Unlike parallel_group.py which fires all popens simultaneously, this
maintains a moving window of `--concurrency` in-flight runner.py
subprocesses. As one finishes, the next is launched.

After each paper completes, its workdir is removed (result.json + logs
are kept in BAMBOO_ROOT/data/results/, which is separate).

Usage:
    BAMBOO_WORKDIR_BASE=/home/me/workdirs \
    BAMBOO_LOG_DIR=/home/me/bamboo-logs \
    python -m scripts.run.run_window \
        --model deep-deepseek-v4-pro \
        --concurrency 16 \
        --timeout 1800 \
        --paper-list bin1.papers
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BAMBOO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = BAMBOO_ROOT / "data" / "results"
WORKDIR_BASE = Path(os.environ.get("BAMBOO_WORKDIR_BASE", "/tmp/bamboo"))
LOG_DIR = Path(os.environ.get("BAMBOO_LOG_DIR", "/tmp/bamboo-window-logs"))


def launch_one(pid: str, model: str, timeout_s: int, prompt_tier: str,
               log_dir: Path) -> tuple[str, subprocess.Popen, "object", float]:
    log_path = log_dir / f"{pid}.log"
    log_fh = open(log_path, "w")
    env = os.environ.copy()
    env.setdefault("PANDA_CROSS_RUN_LEARNING", "1")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "scripts.run.runner",
            "--agents", "panda",
            "--model", model,
            "--papers", pid,
            "--timeout", str(timeout_s),
            "--prompt-tier", prompt_tier,
            "--skip-judge",
        ],
        stdout=log_fh, stderr=subprocess.STDOUT,
        cwd=BAMBOO_ROOT, env=env,
        preexec_fn=os.setsid,
    )
    return (pid, proc, log_fh, time.time())


def cleanup_workdir(pid: str, agent_id_prefix: str = "panda-") -> None:
    """rm -rf workdir for this paper across all matching agent_ids."""
    if not WORKDIR_BASE.is_dir():
        return
    for agent_dir in WORKDIR_BASE.iterdir():
        if not agent_dir.name.startswith(agent_id_prefix):
            continue
        wd = agent_dir / pid
        if wd.exists():
            try:
                shutil.rmtree(wd, ignore_errors=True)
            except Exception:
                pass


def read_overall_level(pid: str) -> str:
    """Best-effort: scan results/*/<pid>.json for overall_level."""
    for sub in RESULTS_DIR.glob("*/" + pid + ".json"):
        try:
            d = json.loads(sub.read_text())
            return f"L{d.get('pass4', {}).get('overall_level', '?')}"
        except Exception:
            pass
    return "L?"


def write_progress(progress_path: Path, totals: dict, in_flight: list) -> None:
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        **totals,
        "in_flight": [
            {"paper": pid, "elapsed_s": int(time.time() - t0)}
            for pid, _, _, t0 in in_flight
        ],
    }
    progress_path.write_text(json.dumps(rec, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--prompt-tier", default="guided")
    ap.add_argument("--paper-list", type=Path,
                    help="File with one paper_id per line")
    ap.add_argument("--log-dir", type=Path, default=LOG_DIR)
    ap.add_argument("--progress-file", type=Path,
                    default=Path(os.environ.get("BAMBOO_PROGRESS_FILE",
                                                "/tmp/bamboo-progress.json")))
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Skip workdir cleanup after each paper")
    ap.add_argument("papers", nargs="*")
    args = ap.parse_args()

    if args.paper_list:
        all_papers = [
            ln.strip() for ln in args.paper_list.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        all_papers = list(args.papers)
    if not all_papers:
        sys.exit("No papers given.")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.progress_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[window] {len(all_papers)} papers, concurrency={args.concurrency}, "
          f"timeout={args.timeout}s/paper, model={args.model}")
    print(f"[window] logs={args.log_dir}, workdir_base={WORKDIR_BASE}")
    print(f"[window] results→{RESULTS_DIR}, progress={args.progress_file}")
    sys.stdout.flush()

    queue: list[str] = list(all_papers)
    in_flight: list = []
    completed = 0
    failed = 0
    started_at = time.time()
    totals = {"total": len(all_papers), "completed": 0, "failed": 0,
              "queued": len(queue), "in_flight_count": 0}

    def shutdown(signum, frame):
        print(f"\n[window] caught signal {signum}; killing in-flight...")
        for pid, p, fh, _ in in_flight:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        sys.exit(130)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while queue or in_flight:
        # Top up
        while len(in_flight) < args.concurrency and queue:
            pid = queue.pop(0)
            try:
                in_flight.append(
                    launch_one(pid, args.model, args.timeout,
                               args.prompt_tier, args.log_dir)
                )
                print(f"[launch] {pid}  inflight={len(in_flight)} "
                      f"queue={len(queue)}", flush=True)
            except Exception as e:
                print(f"[launch-error] {pid}: {e}", flush=True)
                failed += 1; completed += 1
        # Reap
        still: list = []
        for pid, p, fh, t0 in in_flight:
            if p.poll() is None:
                # Hard timeout safety: runner has its own timeout, but ours is +120s
                if time.time() - t0 > args.timeout + 120:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except Exception:
                        pass
                still.append((pid, p, fh, t0))
            else:
                fh.close()
                completed += 1
                if p.returncode != 0:
                    failed += 1
                if not args.no_cleanup:
                    cleanup_workdir(pid)
                level = read_overall_level(pid)
                elapsed = int(time.time() - t0)
                wall = int(time.time() - started_at)
                rate = completed / max(wall, 1) * 3600
                eta_h = (len(all_papers) - completed) / max(rate, 1e-6)
                print(f"[done {completed}/{len(all_papers)}] {pid} "
                      f"exit={p.returncode} {level} {elapsed}s "
                      f"(wall {wall}s, rate {rate:.1f}/h, eta {eta_h:.1f}h)",
                      flush=True)
        in_flight = still
        totals.update(completed=completed, failed=failed,
                      queued=len(queue), in_flight_count=len(in_flight))
        write_progress(args.progress_file, totals, in_flight)
        time.sleep(2)

    print(f"\n[window] DONE: {completed}/{len(all_papers)} "
          f"({failed} failed) in {int(time.time()-started_at)}s")


if __name__ == "__main__":
    main()
