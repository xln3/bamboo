# Failure Analysis: bamboo-06079 (VAR) — Panda L0

**Paper**: Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction
**Venue**: NeurIPS 2024 Best Paper
**Run ID**: `panda-glm-5-turbo` @ 2026-03-28
**Result**: L0 (complete failure — no result JSON produced)
**Wall time**: 3660s (61 min), killed by runner timeout

---

## 1. Introduction

bamboo-06079 is a high-profile benchmark paper (NeurIPS 2024 Best Paper) with 28
ground-truth claims across three experiment tables. The reproduction task is
evaluation-only: download pretrained VAR checkpoints, sample 50k images per model,
and compute FID/IS/Precision/Recall using OpenAI's guided-diffusion toolkit.

This document analyses why Panda scored L0 on this paper — meaning it failed to
produce even a valid result JSON — despite making substantial progress on
environment setup. We trace the full execution timeline, identify nine distinct
bugs that contributed to the failure, and propose a tiered fix plan.

---

## 2. Run Description

### 2.1 What the benchmark asked

The runner (`scripts/run/runner.py`) invoked Panda via:

```
deno run --allow-all panda/mod.ts \
  -p <prompt> --profile coding --max-tokens 32000 --cost-limit <limit>
```

The prompt instructed the agent to:
1. Clone `https://github.com/FoundationVision/VAR`
2. Set up a venv inheriting the system torch/CUDA
3. Run experiments for claims c1–c28
4. Write a result JSON to a specific path

The runner enforced a hard kill after `timeout_s + 60 = 3660s`.

### 2.2 What the agent actually did (timeline)

| Wall time | Action | Outcome |
|-----------|--------|---------|
| 0–5s | Clone repo | Already existed (from prior run) |
| 5–30s | Read README, explore codebase | Understood eval pipeline correctly |
| 30–60s | Create venv, check pip deps | All deps already installed |
| 60–90s | Check existing checkpoints | Found `vae_ch160v4096z32.pth` (436 MB) + `var_d16.pth` (1.2 GB) |
| 90–95s | `wget` var_d20.pth | SSL error through proxy |
| 95–3548s | `wget --no-check-certificate` var_d20.pth | Timed out (2.4 GB at ~10 MB/s) |
| ~3548s | Switch to `huggingface_hub` Python download | Wrong venv path — "No such file or directory" |
| ~3550s | Retry with correct path, run in background | PID 2740689, later "Task not found" |
| ~3555s | Clone guided-diffusion eval toolkit | Success |
| ~3560s | Install tensorflow | Success |
| ~3565s | Check download status | Only d16 exists (background download failed) |
| ~3570s | Launch second background download | PID 2741564 |
| ~3575s | Start downloading reference NPZ files | In progress |
| 3660s | **Runner kills process** | **No result JSON written** |

### 2.3 What the agent never reached

- Running any experiment (not even VAR-d16, whose checkpoint was ready from minute 1)
- Writing the result JSON (planned as the last step)
- Any L2/L3 evaluation

### 2.4 What the agent had successfully accomplished

- Full L1 (environment setup): repo cloned, venv created, all pip deps installed,
  CUDA verified, eval toolkit ready, 2 of 6 checkpoints downloaded
- This is unambiguously an **L1 pass**, but the agent never recorded it

---

## 3. Bug List

### Bug 1 — No wall-clock deadline communicated to the agent

**Severity**: Critical (root cause of L0)
**Component**: `bamboo/scripts/run/prompt_builder.py`, `bamboo/scripts/run/agents/panda.py`

The prompt never tells the agent how much time it has. The runner allocates
`timeout_s` (default 1800s, this run used 3600s) and kills the process when it
expires, but neither the prompt text nor the Panda CLI args carry this
information.

`PandaAdapter.build_command()` receives `timeout_s` as a parameter and ignores it:

```python
# panda.py:21-41 — timeout_s is never used
def build_command(self, prompt: str, workdir: Path, timeout_s: int) -> list[str]:
    cmd = [str(DENO), "run", "--allow-all", str(PANDA_ROOT / "mod.ts"),
           "-p", prompt, "--profile", "coding", "--max-tokens", "32000", ...]
    return cmd
```

The Panda agent internally has only cost-based ($50) and turn-based (200 turns)
circuit breakers — no wall-clock awareness. The agent therefore has no ability to
budget its time or write results before the deadline.

**Evidence**: Agent spent 61 minutes on setup without urgency.

---

### Bug 2 — No result JSON written before timeout

**Severity**: Critical (the proximate cause of L0 vs L1)
**Component**: Agent behavior / prompt design

The agent planned to write the result JSON as its final step (todo item 9 of 9).
When the runner killed it, no JSON existed. The runner's fallback logic
(`make_fallback_result`) produced:

```json
{
  "pass4": { "l1_build": {"status": "fail", "detail": "Process timed out after 3660s"}, ... },
  "overall_level": 0
}
```

This is wrong — L1 (build) had clearly passed. But the fallback has no way to
know that; it only checks whether the agent wrote a result file.

The prompt does say "Write the result JSON file BEFORE your final response," but
does not instruct the agent to write it incrementally as levels are achieved.

**Evidence**: Result file at `data/results/panda-glm-5-turbo/bamboo-06079.json`
shows the fallback template, not agent-written JSON.

---

### Bug 3 — Greedy "download everything first" strategy

**Severity**: High (caused the timeout)
**Component**: Agent planning (LLM behavior)

The agent's todo list committed to downloading ALL 6 checkpoints (VAE, d16, d20,
d24, d30, d36 — totalling ~15–20 GB) before running any experiments. At observed
proxy speeds (5–20 MB/s), this would take 12–60+ minutes of pure download time.

Meanwhile, `var_d16.pth` was already on disk. The agent could have:
1. Written L1=pass immediately
2. Run VAR-d16 experiments (claims c1–c4) with the available checkpoint
3. Downloaded remaining checkpoints in parallel
4. Updated results after each model

This would have yielded L1 at minimum, likely L2, and possibly partial L3.

**Evidence**: Log shows `var_d16.pth` (1.2 GB) present at the 60-second mark.
Agent did not attempt to use it for 60 more minutes.

---

### Bug 4 — Background download processes fail silently

**Severity**: High
**Component**: `panda/src/tools/bash.ts` (background mode)

The agent launched two background downloads:
- PID 2740689 — later `TaskOutput` returned "Task not found: 2740689"
- PID 2741564 — no evidence of completion

After the agent checked, only `var_d16.pth` existed. The background downloads
failed for multiple reasons:

1. **First attempt**: used relative path `.venv/bin/python` from the wrong cwd
   → "No such file or directory"
2. **No output capture**: background mode calls `process.stdout.cancel()` and
   `process.unref()`, so errors are invisible
3. **No completion polling**: the PID-based `TaskOutput` mechanism is for the
   internal task system, not for tracking detached bash processes
4. **Proxy/SSL issues**: `hf_hub_download` likely hit the same SSL problems as
   wget, but errors were silently discarded

**Evidence**: Log line 1564: `Task not found: 2740689`. Log line 1570: only
`var_d16.pth` present after both background download attempts.

---

### Bug 5 — Proxy/SSL issues not handled proactively

**Severity**: Medium
**Component**: `bamboo/scripts/run/prompt_builder.py` (environment probe)

The system routes traffic through a local proxy at `127.0.0.1:7890`. The first
wget to HuggingFace failed immediately:

```
OpenSSL: error:0A000126:SSL routines::unexpected eof while reading
```

The agent had to waste a turn retrying with `--no-check-certificate`.

The environment probe (`probe_environment()`) detects Python, torch, CUDA, and
pre-installed packages, but does not detect or report proxy configuration. If it
did, the agent could use appropriate flags from the first attempt.

The `huggingface_hub` library may also need `HF_HUB_DISABLE_SSL_VERIFY=1` or
equivalent, which the agent never set — a likely cause of the silent background
download failures (Bug 4).

**Evidence**: Log lines 1134–1137 (SSL error), 1140–1141 (retry with
`--no-check-certificate`).

---

### Bug 6 — Bash tool timeout possibly failed to kill wget

**Severity**: Medium
**Component**: `panda/src/tools/bash.ts`, `panda/src/utils/process-kill.ts`

The wget download was submitted with `timeout=300s` (300,000ms). The tool did
eventually produce a `[TOOL_RESULT: TIMEOUT]` message, but the recorded execution
time was `[Error] [3452.8s]` — 57 minutes, far exceeding the 300s limit.

Two possible explanations:
1. The `[Error] [3452.8s]` is wall time from tool submission to outer process
   death (includes queued LLM inference time after timeout fired). The timeout
   did fire at ~300s and returned the TIMEOUT result, but the outer runner then
   killed the whole Deno process much later while the LLM was processing.
2. `killProcessTree` failed to terminate wget (stuck in kernel I/O through proxy
   tunnel). The 5-second hard timeout in `process-kill.ts` returned a synthetic
   status, but the wget process continued as an orphan consuming bandwidth.

In either case, the agent believed the download had timed out and moved on, but
the wget child process may have continued consuming network I/O and interfering
with subsequent download attempts.

**Evidence**: `(timeout=300s)` vs `[Error] [3452.8s]` on the same tool
invocation (log lines 1140–1215).

---

### Bug 7 — Prompt lacks time-budget and download-strategy guidance

**Severity**: Medium
**Component**: `bamboo/scripts/run/prompt_builder.py`

The prompt's NETWORK section says:

> If a download is slow (< 1MB/s) or fails, do not keep retrying the same source.

The agent observed ~10 MB/s throughput, which is above the 1 MB/s threshold, so
it judged the downloads as "working but slow" and kept going. The prompt gives no
guidance on:
- Total time budget relative to download sizes
- Prioritising experiments over downloads
- That multi-GB checkpoint downloads through proxy are expected to be slow
- Strategies like "run with available checkpoints first"

**Evidence**: Agent's thinking (log line 1226): "Each model is ~2GB, so that's
~6-8 minutes per file. Total for remaining 4 files: ~30 minutes." — It estimated
correctly but had no deadline to compare against.

---

### Bug 8 — PandaAdapter discards the timeout parameter

**Severity**: Medium (architectural gap)
**Component**: `bamboo/scripts/run/agents/panda.py`

`build_command()` accepts `timeout_s` but never injects it into the Panda CLI
invocation or prompt. This is the mechanism-level cause of Bug 1:

```python
def build_command(self, prompt: str, workdir: Path, timeout_s: int) -> list[str]:
    # timeout_s available here but completely unused
    cmd = [str(DENO), "run", "--allow-all", str(PANDA_ROOT / "mod.ts"),
           "-p", prompt, "--profile", "coding", "--max-tokens", "32000", "--log-detail"]
    ...
    return cmd
```

The same gap exists in `ClaudeCodeAdapter` and `OpenCodeAdapter` — no agent
adapter passes the time budget to the agent.

**Evidence**: Source code `bamboo/scripts/run/agents/panda.py:21-41`.

---

### Bug 9 — Ablation claims (c23–c28) are infeasible but not pre-filtered

**Severity**: Low
**Component**: `bamboo/data/paper_claims_v2/bamboo-06079.json`, prompt design

Claims c23–c28 are ablation experiments requiring multi-GPU training from scratch
(the paper used 8 GPUs). The benchmark machine has 1 GPU. These claims are
structurally infeasible for single-GPU evaluation, but they are included in the
prompt as "experiments to reproduce" without any feasibility annotation.

The agent correctly identified this barrier in its thinking but spent cognitive
effort on it. Pre-filtering or annotating infeasible claims would help agents
focus time on achievable experiments.

**Evidence**: Agent thinking (log line 1242): "c23-c28 are ablation experiments.
These require training from scratch with different hyperparameters... We only have
1 GPU. These are likely not feasible."

---

## 4. Fix Plan

### Tier 1 — Prevent L0 (quick wins, high impact)

These fixes ensure the agent writes a valid result JSON and scores at least L1
when environment setup succeeds.

#### Fix 1.1: Inject wall-clock deadline into the prompt

**Files**: `bamboo/scripts/run/prompt_builder.py`, `bamboo/scripts/run/agents/panda.py`

Pass `timeout_s` from the runner into `build_prompt()` and add a section:

```
TIME BUDGET: You have {timeout_s} seconds from now. Plan accordingly:
- Write a provisional result JSON within the first 5 minutes (after env setup).
- Allocate no more than 30% of total time to downloads.
- If time is short, run experiments with available checkpoints first.
```

Also inject it into `PandaAdapter.build_command()` so Panda's own cost/turn
limits are aligned:

```python
cmd.extend(["--cost-limit", str(min(cost_limit, timeout_s * 0.02))])
```

#### Fix 1.2: Instruct incremental result JSON writes in the prompt

**File**: `bamboo/scripts/run/prompt_builder.py`

Add to the OUTPUT section:

```
Write the result JSON EARLY and UPDATE it as you progress:
1. After environment setup succeeds → write with l1_build="pass", overall_level=1
2. After each experiment completes → update with l2_run status and any metrics
3. Before any long operation (download, training) → ensure the file is current

The runner may kill your process at any time. Only what is on disk counts.
```

#### Fix 1.3: Add SIGTERM handler to Panda for graceful shutdown

**File**: `panda/src/core/query.ts` or `panda/mod.ts`

Register a SIGTERM handler that writes a best-effort result JSON before exiting.
The bamboo runner sends SIGTERM (via `proc.kill()`) before SIGKILL, so there is a
brief window to flush state. This is a safety net for when the LLM-level
instructions (Fix 1.2) are not followed.

```typescript
Deno.addSignalListener("SIGTERM", () => {
  writePartialResult(context, resultPath);
  Deno.exit(130);
});
```

### Tier 2 — Improve L1 → L2+ success rate

These fixes address the strategic and infrastructure problems that prevented the
agent from running any experiments.

#### Fix 2.1: Add proxy/network awareness to environment probe

**File**: `bamboo/scripts/run/prompt_builder.py` (`probe_environment()`)

Detect and report proxy settings:

```python
# Proxy
for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    val = os.environ.get(var, "")
    if val:
        parts.append(f"proxy: {var}={val}")
        parts.append("hint: use --no-check-certificate for wget; "
                      "set HF_HUB_DISABLE_SSL_VERIFY=1 for huggingface_hub")
        break
```

#### Fix 2.2: Add incremental experiment strategy to the prompt

**File**: `bamboo/scripts/run/prompt_builder.py`

Add to INSTRUCTIONS:

```
STRATEGY FOR LARGE EXPERIMENTS:
- If multiple model checkpoints are needed, run experiments with already-available
  checkpoints FIRST while downloading the rest in parallel.
- After each experiment completes, update the result JSON immediately.
- Do not block all experiments on all downloads completing.
```

#### Fix 2.3: Fix background process output capture

**File**: `panda/src/tools/bash.ts`

For background processes, redirect stdout/stderr to temp files instead of
cancelling the streams:

```typescript
// Instead of:
process.stdout.cancel();
// Do:
const logPath = `/tmp/panda-bg-${process.pid}.log`;
// Pipe to file so the agent can check later via: cat /tmp/panda-bg-<pid>.log
```

Return the log path alongside the PID so the agent can poll for errors:

```
Process started in background (PID: 12345).
Output: /tmp/panda-bg-12345.log
Check status: kill -0 12345 && echo running || echo done
```

#### Fix 2.4: Pre-annotate infeasible claims

**File**: `bamboo/data/paper_claims_v2/bamboo-06079.json` (and others)

Add a `feasibility` field to claims that require resources beyond the benchmark
machine:

```json
{
  "claim_id": "c23",
  "description": "Table 3: Ablation AR→VAR ...",
  "feasibility": "requires_training",
  "feasibility_note": "Requires multi-GPU training from scratch (8x V100)"
}
```

Update `build_prompt()` to filter or annotate these claims so agents don't waste
time planning for them.

### Tier 3 — Robustness improvements (lower priority)

#### Fix 3.1: Verify Bash tool timeout enforcement under proxy I/O

**File**: `panda/src/utils/process-kill.ts`

Investigate whether `killProcessTree` reliably terminates processes stuck in
network I/O through a proxy. If SIGTERM + SIGKILL fail to kill a wget blocked on
a proxy socket, consider:
- Killing the entire process group (`kill -- -$PGID`)
- Using `timeout` command wrapper as a belt-and-suspenders approach

#### Fix 3.2: Pass `timeout_s` through all agent adapters

**Files**: `bamboo/scripts/run/agents/panda.py`, `claude_code.py`, `opencode.py`

Ensure every adapter passes the time budget to its respective agent CLI. For
Panda, this could be a `--deadline` flag or an environment variable
`PANDA_DEADLINE_EPOCH`.

---

## 5. Expected Impact

| Fix | Prevents | Expected level |
|-----|----------|----------------|
| 1.1 + 1.2 alone | L0 | L1 (agent writes JSON after env setup) |
| 1.1 + 1.2 + 2.2 | Wasted download time | L2 (runs d16 experiments within timeout) |
| Full Tier 1 + 2 | All identified bugs | L2, possibly L3 on d16 claims (c1–c4) |

The VAR-d16 experiment (claims c1–c4: FID, IS, Precision, Recall on ImageNet
256x256) requires only the already-downloaded `var_d16.pth` checkpoint. Sampling
50k images on a single GPU takes approximately 10–20 minutes. With proper time
budgeting, Panda should be able to complete at least this experiment within a
60-minute window.
