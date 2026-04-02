"""Build the reproduction prompt for a paper entry.

Supports three prompt tiers that control how much guidance the agent receives:

  bare    — Task definition only (paper, result schema, workdir, timeout).
            Tests raw agent capability: can it discover the environment, identify
            what to reproduce, manage time, and organise its own workflow?

  neutral — bare + host environment facts + pre-extracted claims.
            Equal-information comparison: all agents start with the same facts,
            but no advice on what to do with them.  Claims coverage (how many
            experiments the agent identifies on its own) is measurable at the
            bare tier.

  guided  — neutral + strategy coaching (time allocation, download tactics,
            dependency management, progressive result writing).
            Maximises success rate but measures instruction-following, not autonomy.

The delta between tiers is itself valuable benchmark data.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Literal

PromptTier = Literal["bare", "neutral", "guided"]

PROMPT_TIERS: list[PromptTier] = ["bare", "neutral", "guided"]


# ── Environment probing ─────────────────────────────────────────

def probe_environment() -> str:
    """Probe the host environment and return a plain-fact summary.

    Returns only observable facts — no advice, no recommendations.
    Used by the 'neutral' and 'guided' tiers.
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

    # Proxy / network — just the address, no hints
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        val = os.environ.get(var, "")
        if val:
            parts.append(f"proxy: {var}={val}")
            break

    return "\n".join(parts)


_ENV_CACHE: str | None = None


def get_env_summary() -> str:
    """Cached environment probe (runs once per process)."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = probe_environment()
    return _ENV_CACHE


# ── Constants ────────────────────────────────────────────────────

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

PROXY_HINTS = (
    "proxy-hints: use --no-check-certificate for wget; "
    "set HF_HUB_DISABLE_SSL_VERIFY=1 for huggingface_hub; "
    "set REQUESTS_CA_BUNDLE='' if SSL errors persist"
)


# ── Prompt builder ───────────────────────────────────────────────

def _build_claims_block(claims: list[dict[str, Any]]) -> str:
    """Format the claims list. Shared across all tiers."""
    if not claims:
        return (
            "No pre-extracted claims. Identify and run the main "
            "experiments from the paper/README."
        )

    lines = []
    for c in claims:
        feasibility = c.get("feasibility", "")
        note = c.get("feasibility_note", "")
        tag = f" [INFEASIBLE: {note}]" if feasibility else ""
        lines.append(
            f"  - {c['claim_id']}: {c['description']} "
            f"(metric={c['metric_name']}, dataset={c.get('dataset', 'N/A')}, "
            f"category={c.get('category', 'main')}){tag}"
        )

    feasible = [c for c in claims if not c.get("feasibility")]
    infeasible = [c for c in claims if c.get("feasibility")]
    header = f"Experiments to reproduce ({len(feasible)} feasible"
    if infeasible:
        header += f", {len(infeasible)} infeasible — skip infeasible ones"
    header += "):"
    return header + "\n" + "\n".join(lines)


def build_prompt(
    paper: dict[str, Any],
    agent_id: str,
    result_path: Path,
    workdir: Path,
    timeout_s: int = 1800,
    tier: PromptTier = "guided",
) -> str:
    """Build the prompt that instructs an agent to reproduce a paper.

    Args:
        tier: Controls how much guidance the agent receives.
              "bare"    — task definition only
              "neutral" — + environment facts
              "guided"  — + strategy coaching (current default)
    """
    paper_id = paper["paper_id"]
    title = paper["title"]
    code_url = paper.get("code_url") or ""
    commit = paper.get("code_commit") or "HEAD"
    arxiv_id = paper.get("arxiv_id", "")
    paper_url = paper.get("paper_url", "")
    abstract = paper.get("abstract", "")[:500]

    schema_snippet = RESULT_SCHEMA_SNIPPET.replace(
        "<paper_id>", paper_id
    ).replace("<agent_id>", agent_id)

    if code_url:
        repo_info = f"- Repository: {code_url}\n- Commit: {commit}"
    else:
        repo_info = (
            f"- Repository: NOT PROVIDED (find it from the paper or arxiv page)\n"
            f"- Paper URL: {paper_url}"
        )

    claims = paper.get("ground_truth_claims") or []

    # ── Assemble sections by tier ────────────────────────────
    sections: list[str] = []

    # — Task header (all tiers) —
    sections.append(
        f'TASK: Reproduce the machine learning paper "{title}" '
        f"using its original code."
    )

    # — Paper info (all tiers) —
    sections.append(
        f"PAPER INFO:\n"
        f"- Paper ID: {paper_id}\n"
        f"- ArXiv: {arxiv_id}\n"
        f"{repo_info}\n"
        f"- Abstract: {abstract}"
    )

    # — Environment facts (neutral, guided) —
    if tier in ("neutral", "guided"):
        env_summary = get_env_summary()
        if tier == "guided":
            sections.append(
                f"HOST ENVIRONMENT (already installed — do NOT re-download these):\n"
                f"{env_summary}\n"
                f"{PROXY_HINTS}"
            )
        else:
            # neutral: facts only, no advice
            sections.append(f"HOST ENVIRONMENT:\n{env_summary}")

    # — Time budget —
    if tier == "guided":
        sections.append(
            f"TIME BUDGET: You have {timeout_s} seconds ({timeout_s // 60} minutes) "
            f"from now. The runner will kill your process at {timeout_s + 60}s "
            f"with NO warning. Plan accordingly:\n"
            f"- Write a provisional result JSON within the first 5 minutes "
            f"(after env setup).\n"
            f"- Allocate no more than 30% of total time "
            f"({timeout_s * 30 // 100}s) to downloads.\n"
            f"- Reserve the last 10% ({timeout_s * 10 // 100}s) for writing "
            f"final results.\n"
            f"- If time is short, run experiments with AVAILABLE checkpoints first."
        )
    else:
        # bare, neutral: just the number
        sections.append(
            f"TIME BUDGET: {timeout_s} seconds ({timeout_s // 60} minutes). "
            f"Hard kill at {timeout_s + 60}s."
        )

    # — Claims (neutral, guided) / open-ended task (bare) —
    if tier == "bare":
        sections.append(
            "Read the paper and its code repository. Identify the main "
            "experiments and reproduce them."
        )
    else:
        sections.append(_build_claims_block(claims))

    # — Instructions —
    if tier == "guided":
        if code_url:
            clone_step = (
                f"1. Clone the repository: git clone {code_url} repo "
                f"&& cd repo && git checkout {commit}"
            )
        else:
            clone_step = (
                "1. Find the official code repository from the paper URL or "
                "arxiv page, then clone it: git clone <repo_url> repo && cd repo"
            )
        sections.append(
            f"INSTRUCTIONS:\n"
            f"{clone_step}\n"
            f"2. Read the README and paper to understand how to run experiments.\n"
            f"3. Set up the environment:\n"
            f"   - Create venv with: python3 -m venv .venv --system-site-packages\n"
            f"     This inherits the system torch/CUDA listed above — no need to "
            f"download them.\n"
            f"   - Read requirements.txt. Only install packages that are MISSING. "
            f"Do NOT blindly\n"
            f"     pip install the whole file if it would downgrade working system "
            f"packages.\n"
            f"   - If the paper pins an old torch version, evaluate whether the "
            f"code actually\n"
            f"     needs it or just needs a compatible version. Prefer the system "
            f"torch.\n"
            f"4. Run the experiments listed above. Let ALL output print to stdout "
            f"— do NOT suppress or redirect experiment output.\n"
            f"5. If an experiment produces result files (CSV, JSON, logs), leave "
            f"them in the working directory."
        )
        sections.append(
            "STRATEGY FOR LARGE EXPERIMENTS:\n"
            "- If multiple model checkpoints are needed, run experiments with "
            "ALREADY-AVAILABLE\n"
            "  checkpoints FIRST while downloading the rest in parallel.\n"
            "- After each experiment completes, update the result JSON "
            "immediately.\n"
            "- Do NOT block all experiments on all downloads completing.\n"
            "- Prioritize: write L1 result → run available experiments → "
            "download more → run more."
        )
        sections.append(
            f"IMPORTANT RULES:\n"
            f"- Work in directory: {workdir}\n"
            f"- You do NOT know the expected metric values. Run experiments "
            f"honestly and let the output speak for itself.\n"
            f"- If you encounter an error you cannot fix after 3 attempts, "
            f"record it as a barrier and move on.\n"
            f"- Do NOT skip steps — always attempt environment setup even if "
            f"you expect failure.\n"
            f"- Do NOT fabricate, estimate, or hard-code any metric values.\n"
            f"- NETWORK: If a download is slow (< 1MB/s) or fails, do not keep "
            f"retrying the same source.\n"
            f"  Try alternatives: use a mirror, use uv instead of pip, or ask "
            f"the user about\n"
            f"  proxy/mirror settings via AskHuman.\n"
            f"  Multi-GB checkpoint downloads through a proxy will be slow "
            f"(5-20 MB/s). Factor this\n"
            f"  into your time budget — a 2 GB file takes ~2-7 minutes. Do NOT "
            f"wait for all downloads\n"
            f"  before starting experiments."
        )
    else:
        # bare, neutral: minimal — let the agent figure out workflow
        sections.append(
            f"INSTRUCTIONS:\n"
            f"Reproduce the experiments listed above.\n"
            f"Work in directory: {workdir}\n"
            f"Do NOT fabricate or hard-code any metric values."
        )

    # — Output spec (all tiers) —
    sections.append(
        f"OUTPUT: Write the result JSON to exactly this path:\n"
        f"  {result_path}\n\n"
        f"The JSON must follow this schema:\n"
        f"{schema_snippet}"
    )

    # — Progressive writing advice (guided only) —
    if tier == "guided":
        sections.append(
            "CRITICAL — Write the result JSON EARLY and UPDATE it as you "
            "progress:\n"
            "1. After environment setup succeeds → write with l1_build=\"pass\","
            " overall_level=1\n"
            "2. After each experiment completes → update with l2_run status "
            "and any metrics\n"
            "3. Before any long operation (download, training) → ensure the "
            "JSON file is current\n"
            "The runner may kill your process at any time. Only what is ON DISK "
            "counts."
        )

    # — Result JSON semantics (all tiers — these are schema docs, not strategy) —
    sections.append(
        "Rules for the result JSON:\n"
        "- overall_level: 0 if L1 (build) failed, 1 if L1 passed but L2 "
        "(run) failed, 2 if experiments ran to completion\n"
        "- L3 (reproduce) is evaluated by an independent judge — set l3 "
        'status to "skip"\n'
        "- Record any barriers encountered with evidence (error messages)\n"
        "- resource_usage.total_time_ms = your total wall time in milliseconds"
    )

    return "\n\n".join(sections)
