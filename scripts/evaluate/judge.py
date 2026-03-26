#!/usr/bin/env python3
"""Independent judge for BAMBOO L3 evaluation.

Reads agent execution logs and output files, uses LLM to extract actual
metric values for each claim. The judge NEVER sees expected (ground truth)
values — it only knows claim descriptions and metric names.

This is the core anti-cheating mechanism: the agent's self-reported values
are ignored; only the judge's extraction counts for scoring.

Usage:
    # Judge a single agent run
    python -m scripts.evaluate.judge \
        --paper bamboo-00003 \
        --agent panda \
        --dataset data/bamboo_curated.json

    # Judge all completed runs for an agent
    python -m scripts.evaluate.judge \
        --agent panda \
        --dataset data/bamboo_curated.json \
        --all
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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("judge")

BAMBOO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BAMBOO_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
CLAIMS_V2_DIR = DATA_DIR / "paper_claims_v2"
WORKDIR_BASE = Path("/tmp/bamboo")

# Maximum evidence chars to send to judge LLM
MAX_EVIDENCE_CHARS = 200_000
# Maximum chars per individual output file
MAX_FILE_CHARS = 20_000
# Maximum number of output files to read from workdir
MAX_OUTPUT_FILES = 30

# File extensions to look for in agent workdir
OUTPUT_EXTENSIONS = (
    "*.csv", "*.json", "*.txt", "*.log", "*.out",
    "*.yaml", "*.yml", "*.md", "*.tsv",
)

# ---------------------------------------------------------------------------
# Judge prompt — the judge NEVER sees expected values
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
You are an independent evaluator for machine learning experiment reproduction.

Your task: extract ACTUAL metric values from experiment execution logs.

STRICT RULES:
1. ONLY report values you can FIND in the evidence text. Do NOT guess, estimate, or infer.
2. If a metric value is not present in the logs, set actual_value to null.
3. Include the EXACT text from the logs as evidence for each value you extract.
4. Be precise: match the correct metric name AND dataset. Do not confuse similar metrics.
5. If multiple values exist for the same metric (e.g., different epochs, checkpoints), use the FINAL or BEST value (whichever the paper's description implies).
6. Numeric values only. Convert percentages to numbers (e.g., "95.3%" → 95.3). Keep original precision.
7. If the logs show the experiment FAILED or CRASHED before producing results, set actual_value to null and note the failure in evidence_text."""


def build_judge_prompt(claims: list[dict], evidence: str) -> str:
    """Build the judge prompt. Claims contain NO expected values."""
    claim_lines = []
    for c in claims:
        desc = c.get("description", "")
        metric = c.get("metric_name", "")
        dataset = c.get("dataset", "N/A")
        category = c.get("category", "main")
        claim_lines.append(
            f"  - {c['claim_id']}: {desc}\n"
            f"    metric: {metric} | dataset: {dataset} | category: {category}"
        )

    return f"""{JUDGE_SYSTEM}

CLAIMS TO EVALUATE ({len(claims)} total):
{chr(10).join(claim_lines)}

EVIDENCE (experiment execution logs and output files):
{evidence}

---

For each claim above, extract the actual metric value from the evidence.
Return a JSON array with one entry per claim:
[
  {{
    "claim_id": "c1",
    "actual_value": 92.3,
    "evidence_text": "Test accuracy: 92.3%",
    "confidence": "high"
  }},
  {{
    "claim_id": "c2",
    "actual_value": null,
    "evidence_text": "Metric not found in logs",
    "confidence": "not_found"
  }}
]

confidence levels:
- "high": value clearly matches the claim's metric and dataset
- "medium": value likely matches but some ambiguity exists
- "low": value found but uncertain if it matches this specific claim
- "not_found": metric not present in the evidence

Return ONLY the JSON array. No markdown fences, no commentary."""


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def collect_evidence(
    logs_dir: Path,
    workdir: Path,
    max_chars: int = MAX_EVIDENCE_CHARS,
) -> str:
    """Collect all text evidence from agent execution.

    Priority: stdout > output files > stderr (stderr is mostly noise but
    sometimes contains metric output).
    """
    parts: list[str] = []
    chars_used = 0

    # 1. stdout — most important, contains experiment output
    stdout_path = logs_dir / "stdout.txt"
    if stdout_path.exists():
        text = stdout_path.read_text(errors="replace")
        budget = max_chars * 2 // 3  # 2/3 of budget for stdout
        if len(text) > budget:
            # Keep both head (setup info) and tail (results)
            head = text[:budget // 5]
            tail = text[-(budget * 4 // 5):]
            text = head + "\n\n...[truncated middle]...\n\n" + tail
        parts.append(f"=== AGENT STDOUT ({len(text)} chars) ===\n{text}")
        chars_used += len(text)

    # 2. Output files in workdir (CSVs, JSONs, logs)
    file_budget = max(max_chars - chars_used, max_chars // 4)
    file_parts = _collect_workdir_files(workdir, file_budget)
    if file_parts:
        parts.append(file_parts)
        chars_used += len(file_parts)

    # 3. stderr — lower priority, take less space
    stderr_path = logs_dir / "stderr.txt"
    if stderr_path.exists():
        text = stderr_path.read_text(errors="replace")
        budget = max(max_chars - chars_used, max_chars // 10)
        if len(text) > budget:
            text = "...[truncated]...\n" + text[-budget:]
        parts.append(f"=== AGENT STDERR ({len(text)} chars) ===\n{text}")

    evidence = "\n\n".join(parts)

    # Final safety truncation
    if len(evidence) > max_chars:
        evidence = evidence[:max_chars] + "\n...[evidence truncated]..."

    return evidence


def _collect_workdir_files(workdir: Path, budget: int) -> str:
    """Read output files from agent workdir."""
    if not workdir.exists():
        return ""

    files_found: list[tuple[Path, int]] = []
    for ext in OUTPUT_EXTENSIONS:
        for f in workdir.rglob(ext):
            # Skip very large files, hidden files, and venv/node_modules
            rel = str(f.relative_to(workdir))
            if any(skip in rel for skip in (
                ".git/", "node_modules/", "__pycache__/", ".venv/",
                "venv/", ".conda/", "site-packages/",
            )):
                continue
            try:
                size = f.stat().st_size
                if 10 < size < 5_000_000:  # skip empty and huge files
                    files_found.append((f, size))
            except OSError:
                continue

    if not files_found:
        return ""

    # Sort by modification time (newest first — likely results)
    files_found.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    files_found = files_found[:MAX_OUTPUT_FILES]

    parts = []
    chars_used = 0
    for fpath, size in files_found:
        if chars_used >= budget:
            break
        try:
            text = fpath.read_text(errors="replace")
            if len(text) > MAX_FILE_CHARS:
                text = text[:MAX_FILE_CHARS] + "\n...[file truncated]..."
            rel = fpath.relative_to(workdir)
            parts.append(f"--- FILE: {rel} ({size} bytes) ---\n{text}")
            chars_used += len(text)
        except (OSError, UnicodeDecodeError):
            continue

    if not parts:
        return ""
    return "=== OUTPUT FILES ===\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_claude(prompt: str, model: str = "opus", timeout: int = 300) -> str | None:
    """Call claude CLI and return text output."""
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", model,
                "--output-format", "text",
                "--no-session-persistence",
                "--max-turns", "1",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.error("Claude CLI timed out")
        return None
    except FileNotFoundError:
        log.error("claude CLI not found")
        return None

    if result.returncode != 0:
        log.error(f"Claude exit {result.returncode}: {result.stderr[:300]}")
        return None

    return result.stdout


def parse_judge_output(text: str, claims: list[dict]) -> list[dict]:
    """Parse judge LLM output into structured claim results."""
    text = text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines)

    # Try direct parse
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(parsed, list):
        log.error(f"Failed to parse judge output as JSON array ({len(text)} chars)")
        # Return all claims as not-found
        return [
            {
                "claim_id": c["claim_id"],
                "actual_value": None,
                "evidence_text": "Judge output parse failure",
                "confidence": "not_found",
                "extraction_method": "llm_failed",
            }
            for c in claims
        ]

    # Build lookup by claim_id
    result_by_id: dict[str, dict] = {}
    for item in parsed:
        if isinstance(item, dict) and "claim_id" in item:
            cid = item["claim_id"]
            # Normalize actual_value
            val = item.get("actual_value")
            if val is not None:
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = None
            result_by_id[cid] = {
                "claim_id": cid,
                "actual_value": val,
                "evidence_text": str(item.get("evidence_text", ""))[:500],
                "confidence": item.get("confidence", "medium"),
                "extraction_method": "llm",
            }

    # Ensure every claim has a result (fill missing with not-found)
    results = []
    for c in claims:
        cid = c["claim_id"]
        if cid in result_by_id:
            results.append(result_by_id[cid])
        else:
            results.append({
                "claim_id": cid,
                "actual_value": None,
                "evidence_text": "Not returned by judge",
                "confidence": "not_found",
                "extraction_method": "llm_missing",
            })

    return results


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    paper_id: str
    agent_id: str
    judge_model: str
    timestamp: str
    evidence_chars: int
    claim_results: list[dict]
    judge_duration_ms: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def judge_paper(
    paper_id: str,
    agent_id: str,
    claims: list[dict],
    logs_dir: Path,
    workdir: Path,
    model: str = "opus",
) -> JudgeResult:
    """Judge a single paper's reproduction.

    Args:
        paper_id: Paper identifier
        agent_id: Agent identifier
        claims: Ground truth claims (only description/metric_name/dataset used;
                expected values are NEVER sent to the judge)
        logs_dir: Directory containing stdout.txt / stderr.txt
        workdir: Agent's working directory (may contain output files)
        model: Claude model to use for judging

    Returns:
        JudgeResult with per-claim actual values extracted from logs.
    """
    start = time.monotonic_ns()
    timestamp = datetime.now(timezone.utc).isoformat()

    # Strip expected values from claims before building prompt (defense in depth)
    safe_claims = []
    for c in claims:
        safe_claims.append({
            "claim_id": c["claim_id"],
            "description": c.get("description", ""),
            "metric_name": c.get("metric_name", ""),
            "dataset": c.get("dataset", "N/A"),
            "category": c.get("category", "main"),
        })

    # Check if agent produced any output
    stdout_exists = (logs_dir / "stdout.txt").exists()
    stderr_exists = (logs_dir / "stderr.txt").exists()
    if not stdout_exists and not stderr_exists:
        elapsed = int((time.monotonic_ns() - start) / 1_000_000)
        return JudgeResult(
            paper_id=paper_id,
            agent_id=agent_id,
            judge_model=model,
            timestamp=timestamp,
            evidence_chars=0,
            claim_results=[
                {
                    "claim_id": c["claim_id"],
                    "actual_value": None,
                    "evidence_text": "No agent output logs found",
                    "confidence": "not_found",
                    "extraction_method": "no_logs",
                }
                for c in safe_claims
            ],
            judge_duration_ms=elapsed,
            error="No agent output logs found",
        )

    # Collect evidence
    evidence = collect_evidence(logs_dir, workdir)

    if len(evidence.strip()) < 50:
        elapsed = int((time.monotonic_ns() - start) / 1_000_000)
        return JudgeResult(
            paper_id=paper_id,
            agent_id=agent_id,
            judge_model=model,
            timestamp=timestamp,
            evidence_chars=len(evidence),
            claim_results=[
                {
                    "claim_id": c["claim_id"],
                    "actual_value": None,
                    "evidence_text": "Agent output too short / empty",
                    "confidence": "not_found",
                    "extraction_method": "no_output",
                }
                for c in safe_claims
            ],
            judge_duration_ms=elapsed,
            error="Agent output too short",
        )

    # For papers with many claims, batch them to stay within context
    # Claude Opus handles ~200k tokens. Each claim ~100 chars in prompt.
    # With 200k evidence, we have room for ~500 claims per call.
    # Most papers have <300 claims, so one call suffices.
    all_results = []
    batch_size = 200  # claims per judge call

    for i in range(0, len(safe_claims), batch_size):
        batch = safe_claims[i:i + batch_size]
        prompt = build_judge_prompt(batch, evidence)

        raw_output = call_claude(prompt, model=model)
        if raw_output is None:
            # LLM call failed — mark all as not found
            all_results.extend([
                {
                    "claim_id": c["claim_id"],
                    "actual_value": None,
                    "evidence_text": "Judge LLM call failed",
                    "confidence": "not_found",
                    "extraction_method": "llm_error",
                }
                for c in batch
            ])
        else:
            batch_results = parse_judge_output(raw_output, batch)
            all_results.extend(batch_results)

    elapsed = int((time.monotonic_ns() - start) / 1_000_000)

    return JudgeResult(
        paper_id=paper_id,
        agent_id=agent_id,
        judge_model=model,
        timestamp=timestamp,
        evidence_chars=len(evidence),
        claim_results=all_results,
        judge_duration_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_claims_for_paper(paper_id: str, dataset_path: Path | None) -> list[dict]:
    """Load ground truth claims for a paper."""
    # Try paper_claims_v2 first
    v2_path = CLAIMS_V2_DIR / f"{paper_id}.json"
    if v2_path.exists():
        try:
            claims = json.loads(v2_path.read_text())
            if isinstance(claims, list) and claims:
                return claims
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to dataset
    if dataset_path and dataset_path.exists():
        with open(dataset_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for p in data:
                if p["paper_id"] == paper_id:
                    return p.get("ground_truth_claims", [])
        elif isinstance(data, dict):
            entry = data.get(paper_id, {})
            return entry.get("ground_truth_claims", [])

    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="BAMBOO independent judge")
    parser.add_argument("--paper", help="Single paper ID to judge")
    parser.add_argument("--agent", required=True, help="Agent ID")
    parser.add_argument("--dataset", default=str(DATA_DIR / "bamboo_curated.json"),
                        help="Path to dataset JSON")
    parser.add_argument("--model", default="opus", help="Judge LLM model")
    parser.add_argument("--all", action="store_true",
                        help="Judge all completed runs for this agent")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    agent_results_dir = RESULTS_DIR / args.agent
    judge_dir = Path(args.output_dir) if args.output_dir else (agent_results_dir / "judge")
    judge_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)

    if args.paper:
        paper_ids = [args.paper]
    elif args.all:
        # Find all papers with result files
        paper_ids = []
        for f in sorted(agent_results_dir.glob("bamboo-*.json")):
            paper_ids.append(f.stem)
    else:
        parser.error("Specify --paper or --all")
        return

    for paper_id in paper_ids:
        judge_path = judge_dir / f"{paper_id}.json"

        # Skip if already judged
        if judge_path.exists():
            log.info(f"{paper_id}: already judged, skipping")
            continue

        claims = load_claims_for_paper(paper_id, dataset_path)
        if not claims:
            log.warning(f"{paper_id}: no claims found, skipping")
            continue

        logs_dir = agent_results_dir / "logs" / paper_id
        workdir = WORKDIR_BASE / args.agent / paper_id

        log.info(f"{paper_id}: judging {len(claims)} claims "
                 f"(logs={logs_dir.exists()}, workdir={workdir.exists()})")

        result = judge_paper(
            paper_id=paper_id,
            agent_id=args.agent,
            claims=claims,
            logs_dir=logs_dir,
            workdir=workdir,
            model=args.model,
        )

        # Save judge result
        judge_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

        found = sum(1 for cr in result.claim_results if cr.get("actual_value") is not None)
        log.info(f"{paper_id}: {found}/{len(claims)} values extracted "
                 f"({result.judge_duration_ms}ms)")

    log.info("Done.")


if __name__ == "__main__":
    main()
