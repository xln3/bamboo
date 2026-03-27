"""Build the reproduction prompt for a paper entry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
) -> str:
    """Build the prompt that instructs an agent to reproduce a paper."""
    paper_id = paper["paper_id"]
    title = paper["title"]
    code_url = paper.get("code_url") or ""
    commit = paper.get("code_commit", "HEAD")
    arxiv_id = paper.get("arxiv_id", "")
    paper_url = paper.get("paper_url", "")
    abstract = paper.get("abstract", "")[:500]

    claims = paper.get("ground_truth_claims") or []
    claims_block = ""
    if claims:
        claim_lines = []
        for c in claims:
            claim_lines.append(
                f"  - {c['claim_id']}: {c['description']} "
                f"(metric={c['metric_name']}, dataset={c.get('dataset','N/A')}, "
                f"category={c.get('category','main')})"
            )
        claims_block = "Experiments to reproduce:\n" + "\n".join(claim_lines)
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

    prompt = f"""\
TASK: Reproduce the machine learning paper "{title}" using its original code.

PAPER INFO:
- Paper ID: {paper_id}
- ArXiv: {arxiv_id}
{repo_info}
- Abstract: {abstract}

{claims_block}

INSTRUCTIONS:
{clone_step}
2. Read the README and paper to understand how to run experiments.
3. Set up the environment (conda/venv, install dependencies).
4. Run the experiments listed above. Let ALL output print to stdout — do NOT suppress or redirect experiment output.
5. If an experiment produces result files (CSV, JSON, logs), leave them in the working directory.

IMPORTANT RULES:
- Work in directory: {workdir}
- You do NOT know the expected metric values. Run experiments honestly and let the output speak for itself.
- If you encounter an error you cannot fix after 3 attempts, record it as a barrier and move on.
- Do NOT skip steps — always attempt environment setup even if you expect failure.
- Do NOT fabricate, estimate, or hard-code any metric values.

OUTPUT: When done, write a JSON result file to exactly this path:
  {result_path}

The JSON must follow this schema:
{schema_snippet}

Rules for the result JSON:
- overall_level: 0 if L1 (build) failed, 1 if L1 passed but L2 (run) failed, 2 if experiments ran to completion
- L3 (reproduce) is evaluated by an independent judge — set l3 status to "skip"
- Record any barriers encountered with evidence (error messages)
- resource_usage.total_time_ms = your total wall time in milliseconds

Write the result JSON file BEFORE your final response. This is critical."""

    return prompt
