# Prompt Tier System for BAMBOO Runner

**Date:** 2026-04-03
**Files changed:** `scripts/run/prompt_builder.py`, `scripts/run/runner.py`

## Problem

The BAMBOO runner's prompt mixed two roles:

1. **Benchmark definer** -- telling the agent *what* to do (paper info, claims, result schema)
2. **Strategy coach** -- telling the agent *how* to do it (time allocation, download tactics, dependency management, progressive result writing)

This conflation means the benchmark measures "how well does the agent follow instructions" rather than "how capable is the agent." For example, the Infinity paper L0 failure (agent spent 77 tool calls downloading flan-t5-xl) led to adding time allocation advice in the prompt -- which prevents the failure from surfacing rather than letting the benchmark catch it.

Similarly, environment facts (GPU, CUDA, pre-installed packages) and pre-extracted claims are information the agent could discover on its own. An agent that can't probe its own environment or identify what experiments to reproduce from the paper is objectively weaker -- but the old prompt hid this weakness by providing everything upfront.

## Solution

Introduced `--prompt-tier` with three levels:

| Tier | Includes | Measures |
|------|----------|----------|
| `bare` | Paper info, result schema, workdir, timeout number | Raw autonomy: can the agent discover its environment, identify experiments from the paper, manage its own time? |
| `neutral` | bare + environment facts + pre-extracted claims | Equal-information execution: given the same facts (no advice), which agent reproduces better? |
| `guided` | neutral + strategy coaching + defensive rules | Instruction-following: with full hand-holding, what's the ceiling? |

### What goes where

**bare (task definition only):**
- Paper metadata (title, arxiv, repo URL, abstract)
- Workdir path
- Timeout value (just the number, no allocation advice)
- Result output path + JSON schema
- "Reproduce the experiments" (no claims list)

**neutral adds:**
- Host environment probe (Python, torch, CUDA, GPU, packages) -- facts only, no "do NOT re-download" advice
- Proxy address (no SSL workaround hints)
- Pre-extracted claims list with metrics, datasets, feasibility

**guided adds:**
- "do NOT re-download these" directive
- Proxy workaround hints (HF_HUB_DISABLE_SSL_VERIFY, etc.)
- Time allocation strategy (30% download cap, 10% reserved for results)
- Step-by-step clone/setup/run instructions
- Dependency management advice (--system-site-packages, prefer system torch)
- Download strategy (parallel downloads, run available checkpoints first)
- Progressive result writing (write early, update on each step)
- Error handling policy (3 attempts then record barrier)
- Network advice (try mirrors, factor in proxy speed)

### Cross-tier analysis

The delta between tiers is itself valuable benchmark data:

- Agent scores L2 on `guided` but L0 on `bare` --> strong instruction-follower, weak autonomy
- Agent scores L2 on both --> genuinely autonomous
- **Robustness score** = `score_bare / score_guided` measures prompt-engineering dependency
- **Claims coverage** (bare tier): how many ground-truth claims the agent independently identifies from the paper, without being given the claims list

## Usage

```bash
# Raw capability test
python -m scripts.run.runner --agents panda --model glm-5 --prompt-tier bare

# Fair comparison with equal information
python -m scripts.run.runner --agents panda claude-code --model glm-5 --prompt-tier neutral

# Maximum success rate (default, backward compatible)
python -m scripts.run.runner --agents panda --model glm-5 --prompt-tier guided

# Preview prompt differences
python -m scripts.run.runner --agents panda --model glm-5 --dry-run --prompt-tier bare
```

## Backward Compatibility

- Default tier is `guided`, preserving existing behavior
- `build_prompt()` signature adds `tier="guided"` as the last keyword argument
- All existing callers work without changes
