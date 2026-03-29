"""Build the reproduction prompt for a paper entry."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


def probe_environment() -> str:
    """Probe the host environment once and return a structured summary.

    This runs at the runner level (not inside the agent), so the agent
    receives facts instead of having to discover them itself.
    """
    parts: list[str] = []

    # Python
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"],
            capture_output=True, text=True, timeout=5,
        )
        parts.append(f"python: {r.stdout.strip()}")
    except Exception:
        parts.append("python: unknown")

    # Torch + CUDA
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"],
            capture_output=True, text=True, timeout=10,
        )
        parts.append(r.stdout.strip())
    except Exception:
        parts.append("torch: not installed")

    # GPU memory
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts.append(f"gpu: {r.stdout.strip()}")
    except Exception:
        pass

    # Key pre-installed packages
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import importlib; pkgs = ['torchvision','torchaudio','numpy','scipy','transformers','timm','einops'];\n"
             "[print(f'{p}={importlib.import_module(p).__version__}') for p in pkgs if importlib.util.find_spec(p)]"],
            capture_output=True, text=True, timeout=10,
        )
        if r.stdout.strip():
            parts.append("pre-installed: " + ", ".join(r.stdout.strip().split("\n")))
    except Exception:
        pass

    # Proxy / network
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        val = os.environ.get(var, "")
        if val:
            parts.append(f"proxy: {var}={val}")
            parts.append(
                "proxy-hints: use --no-check-certificate for wget; "
                "set HF_HUB_DISABLE_SSL_VERIFY=1 for huggingface_hub; "
                "set REQUESTS_CA_BUNDLE='' if SSL errors persist"
            )
            break

    return "\n".join(parts)


# Cache the probe result (runs once per process)
_ENV_CACHE: str | None = None


def get_env_summary() -> str:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = probe_environment()
    return _ENV_CACHE

RESULT_SCHEMA_SNIPPET = """\
{
  "paper_id": "<paper_id>",
  "agent_id": "<agent_id>",
  "timestamp": "<ISO 8601>",
  "pass4": {
    "l1_build": {"status": "pass|fail|timeout", "duration_ms": <int>, "detail": "<text>"},
    "l2_run":   {"status": "pass|fail|timeout", "duration_ms": <int>, "detail": "<text>"},
    "l3_reproduce": {"status": "skip"},
    "l4_cross": {"status": "skip"},
    "overall_level": <0-2>
  },
  "barriers": [{"level": "<L1_environment|L2_build|L3_framework|L4_hardware_code|L5_microarch>",
                 "description": "<text>", "evidence": "<error msg>",
                 "auto_fixable": <bool>}],
  "resource_usage": {"total_time_ms": <int>}
}"""


def build_prompt(
    paper: dict[str, Any],
    agent_id: str,
    result_path: Path,
    workdir: Path,
    timeout_s: int = 1800,
) -> str:
    """Build the prompt that instructs an agent to reproduce a paper."""
    paper_id = paper["paper_id"]
    title = paper["title"]
    code_url = paper.get("code_url") or ""
    commit = paper.get("code_commit") or "HEAD"
    arxiv_id = paper.get("arxiv_id", "")
    paper_url = paper.get("paper_url", "")
    abstract = paper.get("abstract", "")[:500]

    claims = paper.get("ground_truth_claims") or []
    claims_block = ""
    if claims:
        claim_lines = []
        for c in claims:
            feasibility = c.get("feasibility", "")
            note = c.get("feasibility_note", "")
            tag = f" [INFEASIBLE: {note}]" if feasibility else ""
            claim_lines.append(
                f"  - {c['claim_id']}: {c['description']} "
                f"(metric={c['metric_name']}, dataset={c.get('dataset','N/A')}, "
                f"category={c.get('category','main')}){tag}"
            )
        feasible = [c for c in claims if not c.get("feasibility")]
        infeasible = [c for c in claims if c.get("feasibility")]
        header = f"Experiments to reproduce ({len(feasible)} feasible"
        if infeasible:
            header += f", {len(infeasible)} infeasible — skip infeasible ones"
        header += "):"
        claims_block = header + "\n" + "\n".join(claim_lines)
    else:
        claims_block = (
            "No pre-extracted claims. Identify and run the main experiments from the paper/README."
        )

    schema_snippet = RESULT_SCHEMA_SNIPPET.replace("<paper_id>", paper_id).replace(
        "<agent_id>", agent_id
    )

    # Build repo section based on whether code_url is available
    if code_url:
        repo_info = f"- Repository: {code_url}\n- Commit: {commit}"
        clone_step = f"1. Clone the repository: git clone {code_url} repo && cd repo && git checkout {commit}"
    else:
        repo_info = f"- Repository: NOT PROVIDED (find it from the paper or arxiv page)\n- Paper URL: {paper_url}"
        clone_step = (
            "1. Find the official code repository from the paper URL or arxiv page, "
            "then clone it: git clone <repo_url> repo && cd repo"
        )

    env_summary = get_env_summary()

    prompt = f"""\
TASK: Reproduce the machine learning paper "{title}" using its original code.

PAPER INFO:
- Paper ID: {paper_id}
- ArXiv: {arxiv_id}
{repo_info}
- Abstract: {abstract}

HOST ENVIRONMENT (already installed — do NOT re-download these):
{env_summary}

TIME BUDGET: You have {timeout_s} seconds ({timeout_s // 60} minutes) from now. The runner \
will kill your process at {timeout_s + 60}s with NO warning. Plan accordingly:
- Write a provisional result JSON within the first 5 minutes (after env setup).
- Allocate no more than 30% of total time ({timeout_s * 30 // 100}s) to downloads.
- Reserve the last 10% ({timeout_s * 10 // 100}s) for writing final results.
- If time is short, run experiments with AVAILABLE checkpoints first.

{claims_block}

INSTRUCTIONS:
{clone_step}
2. Read the README and paper to understand how to run experiments.
3. Set up the environment:
   - Create venv with: python3 -m venv .venv --system-site-packages
     This inherits the system torch/CUDA listed above — no need to download them.
   - Read requirements.txt. Only install packages that are MISSING. Do NOT blindly
     pip install the whole file if it would downgrade working system packages.
   - If the paper pins an old torch version, evaluate whether the code actually
     needs it or just needs a compatible version. Prefer the system torch.
4. Run the experiments listed above. Let ALL output print to stdout — do NOT suppress or redirect experiment output.
5. If an experiment produces result files (CSV, JSON, logs), leave them in the working directory.

STRATEGY FOR LARGE EXPERIMENTS:
- If multiple model checkpoints are needed, run experiments with ALREADY-AVAILABLE
  checkpoints FIRST while downloading the rest in parallel.
- After each experiment completes, update the result JSON immediately.
- Do NOT block all experiments on all downloads completing.
- Prioritize: write L1 result → run available experiments → download more → run more.

IMPORTANT RULES:
- Work in directory: {workdir}
- You do NOT know the expected metric values. Run experiments honestly and let the output speak for itself.
- If you encounter an error you cannot fix after 3 attempts, record it as a barrier and move on.
- Do NOT skip steps — always attempt environment setup even if you expect failure.
- Do NOT fabricate, estimate, or hard-code any metric values.
- NETWORK: If a download is slow (< 1MB/s) or fails, do not keep retrying the same source.
  Try alternatives: use a mirror, use uv instead of pip, or ask the user about
  proxy/mirror settings via AskHuman.
  Multi-GB checkpoint downloads through a proxy will be slow (5-20 MB/s). Factor this
  into your time budget — a 2 GB file takes ~2-7 minutes. Do NOT wait for all downloads
  before starting experiments.

OUTPUT: Write the result JSON to exactly this path:
  {result_path}

The JSON must follow this schema:
{schema_snippet}

CRITICAL — Write the result JSON EARLY and UPDATE it as you progress:
1. After environment setup succeeds → write with l1_build="pass", overall_level=1
2. After each experiment completes → update with l2_run status and any metrics
3. Before any long operation (download, training) → ensure the JSON file is current
The runner may kill your process at any time. Only what is ON DISK counts.

Rules for the result JSON:
- overall_level: 0 if L1 (build) failed, 1 if L1 passed but L2 (run) failed, 2 if experiments ran to completion
- L3 (reproduce) is evaluated by an independent judge — set l3 status to "skip"
- Record any barriers encountered with evidence (error messages)
- resource_usage.total_time_ms = your total wall time in milliseconds"""

    return prompt
