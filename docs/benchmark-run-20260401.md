# BAMBOO 12-Paper Benchmark Run Report — 2026-04-01

**Agent**: Panda + GLM-5-Turbo
**Run started**: 2026-03-31 18:21 CST
**Run completed**: 2026-04-01 03:37 CST (~9 hours)
**Judge**: skipped (raw agent scores only)

---

## Context

This run evaluates the impact of recent fixes to the BAMBOO benchmark harness:

1. **Metadata alignment fix**: 11 of 12 benchmark papers had mismatched metadata
   in `bamboo_final.json` and the curated chunks — the titles, code URLs, and
   abstracts pointed to completely different papers than the PDFs/markdowns/claims.
   The root cause was hand-picked benchmark papers (from NeurIPS 2024, CVPR 2023,
   ICML 2023, etc.) overwriting PDF/markdown/claims at IDs already assigned to
   NeurIPS 2025 papers by `finalize_dataset.py`, without updating the metadata.
   **Fixed by patching** both `bamboo_final.json` and the curated chunk.

2. **Stable ID mapping**: Added `data/id_mapping.json` (6,148 entries) and
   `data/benchmark_extras.json` (11 papers) to `finalize_dataset.py` so paper
   IDs are stable across re-runs and benchmark papers from outside tracked venues
   survive re-finalization.

3. **Prior Panda improvements**: Incremental result JSON writes, time budget
   injection, proxy/network hints, download strategy guidance (from earlier
   failure analysis on bamboo-06079).

---

## Results Summary

| Paper ID | Paper Name | Previous | Current | Wall Time | Notes |
|----------|-----------|----------|---------|-----------|-------|
| bamboo-06079 | VAR (Image Generation) | L0 | **L1** | 3 min | Build pass, sampling started |
| bamboo-06080 | Infinity (Image Synthesis) | L0 | L0 | 1 min | Stuck in setup |
| bamboo-06081 | MQ-VAE (Masked VQ) | L0 | **L1** | 5 min | Build pass, no checkpoints/ImageNet |
| bamboo-06082 | ARPC (Image Compression) | L0 | **L1** | 3 min | Build pass, flash_attn mismatch |
| bamboo-06083 | ROI Compression | L0 | **L1** | 5 min | Build pass, slow LSEG checkpoint download |
| bamboo-06084 | SCALAR (Controllable VAR) | L0 | **L1** | 5 min | Build pass, no ImageNet/SCALAR weights |
| bamboo-06085 | TEAL (Activation Sparsity) | L0 | **L1** | 3 min | Build pass, flash_attn + gated HF models |
| bamboo-06086 | CATS (Time Series) | L0 | **L1** | 5 min | Build pass, **experiments actively running** |
| bamboo-06087 | ARCQuant (NVFP4 Quant) | L0 | L0 | 5 min | Stuck in setup |
| bamboo-06088 | FlatQuant (LLM Quant) | L0 | L0 | 0.5 min | Stuck in setup |
| bamboo-06089 | SmoothQuant (PTQ) | L0 | L0 | <1s | **Deno FFI panic** |
| bamboo-03009 | R-Sparse (Activation Sparsity) | L0 | L0 | <1s | **Deno FFI panic** |

### Aggregate

|  | Previous Run | This Run |
|--|-------------|----------|
| L0 (build fail) | 12 | **5** |
| L1 (build pass) | 0 | **7** |
| L2 (experiments ran) | 0 | 0 |

**L1 rate improved from 0% to 58%.** The metadata fix was the primary driver —
agents now clone the correct repos and install the right dependencies.

No paper reached L2 yet. The three blocking issues are analyzed below.

---

## Issue 1: Deno FFI Panic (bamboo-06089, bamboo-03009) — Critical

### Symptom

Both papers show `overall_level: 0` with `total_time_ms: 715` (under 1 second).
The agent never started — `stdout.txt` is empty. `stderr.txt` contains a Deno
panic dump:

```
thread 'main' panicked at libs/core/runtime/jsruntime.rs:2377:3:
internal error: entered unreachable code: Expected at least one stalled top-level await
```

Panic URL (identical for both):
```
https://panic.deno.com/v2.7.7/x86_64-unknown-linux-gnu/i9t1_Gohm3xEg1l3xEg0l3xEgyl3xEgsq3xEgrw2oFg14zoFg33zoF68lgoGgnwgoG6tugoGor2_sG6_v-sGy0kmtG-gxltG0gqx-G0-slyE81_lyEgr-kyEgxuKg8uKgg4u7CA
```

### Root Cause

This is the known Deno 2.7.7 top-level await race condition documented in
`feedback_deno_panic.md`. The panic is non-deterministic — it occurs in the Deno
runtime before any application logic executes. The crash is NOT related to the
paper's content; it is a Deno runtime bug triggered by module loading order and
FFI initialization.

Both papers ran last in the sequence (papers 10 and 11 of 12). The crash may be
more likely after the Deno process has been launched many times in succession
(speculation — needs more data points).

### Impact

- 2 of 12 papers (17%) got L0 due to runtime crashes, not agent logic
- These papers would likely score L1+ with a working runtime

### Fix Options

1. **Retry on panic**: Detect the panic in `runner.py` (check stderr for
   `jsruntime.rs:2377`) and retry the paper up to 2 times
2. **Upgrade Deno**: The panic may be fixed in Deno 2.8+
3. **Workaround**: Use the module deadlock workaround from `feedback_deno_panic.md`
   (avoids the race condition at the cost of slower startup)

---

## Issue 2: flash_attn CUDA Version Mismatch — Medium

### Symptom

Papers bamboo-06082 (ARPC), bamboo-06085 (TEAL), and potentially bamboo-06088
(FlatQuant) report that `flash_attn` cannot be compiled from source:

```
RuntimeError: ('The detected CUDA version (%s) mismatches the version that was
used to compile PyTorch (%s)', '13.2', '12.8')
```

Or:

```
FlashAttention is only supported on CUDA 11.7 and above
```
(when using the system nvcc 11.5)

### Root Cause: Three-Way CUDA Version Mismatch

| Component | Version | Location |
|-----------|---------|----------|
| System nvcc | **11.5** | `/usr/bin/nvcc` (too old for flash_attn) |
| NVIDIA driver | **570.172.18** (CUDA 12.8 runtime) | `nvidia-smi` |
| PyTorch | **2.10.0+cu128** (compiled for CUDA 12.8) | System Python |
| pip nvidia-cuda-nvcc | **13.2** | `~/.local/lib/python3.10/site-packages/nvidia/cu13/bin/nvcc` (too new) |
| GPU | RTX 5880 Ada (48GB) | — |

`flash_attn` requires compiling CUDA kernels from source at install time. It
needs an nvcc that matches the CUDA version PyTorch was compiled with (12.8).
The system has nvcc 11.5 (too old) and pip has nvcc 13.2 (too new). There is no
CUDA 12.8 nvcc on the system.

`flash_attn` does NOT publish prebuilt binary wheels on PyPI — all versions are
source-only distributions that require compilation.

### Impact

- Agents correctly work around this by patching code to use PyTorch's built-in
  `scaled_dot_product_attention` (SDPA) as a fallback
- This costs time (10-15 minutes of agent effort per paper) but doesn't block L1
- May affect L2 experiment accuracy if papers specifically require flash_attn
  behavior (rare — SDPA is functionally equivalent for most use cases)

### Fix Options (system-level)

1. **Install CUDA 12.8 toolkit**: `apt install cuda-toolkit-12-8` from NVIDIA's
   apt repo — provides nvcc 12.8 at `/usr/local/cuda-12.8/bin/nvcc`
2. **Prebuilt wheels from GitHub**: flash-attn publishes `.whl` files for
   specific torch+CUDA combos at
   `https://github.com/Dao-AILab/flash-attention/releases`. Find the wheel for
   `torch2.10+cu128` and pre-install it in the system Python.
3. **Tell agents about the mismatch**: Add to the environment probe output:
   ```
   flash_attn: NOT AVAILABLE (nvcc 11.5 ≠ torch cu128).
   Use torch.nn.functional.scaled_dot_product_attention as fallback.
   ```
   This saves agents 10+ minutes of trial-and-error per paper.

### Recommendation

Option 3 (inform agents) is the quickest fix and costs nothing. Option 1 or 2
should be done if flash_attn performance matters for L3 accuracy validation.

---

## Issue 3: Papers Closest to L2

### bamboo-06086 (CATS — Time Series Forecasting) — Closest to L2

**Status**: L1, experiments actively training when killed by timeout.

**What the agent accomplished:**
- Cloned repo, created venv, installed all dependencies (no flash_attn needed)
- Downloaded all small datasets: ETTm1, ETTm2, ETTh1, ETTh2, Weather, Electricity
- Traffic dataset (136 MB from Google Drive) was extremely slow (~90 KB/s through
  proxy) — multiple timeouts
- **Successfully trained on ETTm1_96_96:**
  ```
  Epoch: 1, Steps: 268 | Train Loss: 0.4654 Vali Loss: 0.6313 Test Loss: 0.5947
  Epoch: 2, Steps: 268 | Train Loss: 0.4226 Vali Loss: 0.6116 Test Loss: 0.5732
  Epoch: 3, Steps: 268 | Train Loss: 0.3462 Vali Loss: 0.3977 Test Loss: 0.3370
  ```
- Created comprehensive `run_all.sh` with 24 experiment configs
- Launched `nohup bash run_all.sh` to run all experiments
- **Killed by timeout** while experiments were still in progress

**Why it didn't reach L2:**
The 1-hour timeout wasn't enough. Each experiment trains for ~3 epochs with early
stopping, taking 5-15 minutes per config. 24 configs × ~10 min = ~4 hours needed.
The agent spent ~20 minutes on setup + downloads, leaving ~40 minutes for
experiments — enough for only a few of the 24 configs.

**Fix to reach L2:**
- Increase timeout to 2+ hours for this paper
- Pre-cache the Traffic dataset (eliminates 25 min download)
- Or reduce claims to just Weather + ETT experiments (skip Traffic)

### bamboo-06085 (TEAL — LLM Activation Sparsity) — Blocked

**Status**: L1, never started experiments.

**What the agent accomplished:**
- Cloned repo (retry needed — initial clone timed out through proxy)
- Created venv, installed dependencies
- Spent ~15 minutes fighting flash_attn (patched to SDPA, see Issue 2)
- Discovered model access issues:
  - Mistral-7B-v0.1: Fully cached (664 MB blobs present) ✓
  - Llama-2-7B: Gated model — only ref cached (12 KB), no weights. No HF token.
  - Llama-3-8B: Download started, slow through proxy, killed
  - Tried `openlm-research/open_llama_7b` as alternative
- **Killed by timeout** while downloading alternative models

**Why it didn't reach L2:**
Two compounding blockers consumed the entire 1-hour budget:
1. flash_attn compilation attempts (~15 min)
2. Gated HuggingFace models require `HF_TOKEN` which isn't configured

**Fix to reach L2:**
- Pre-install flash_attn or inform agent about SDPA fallback (saves 15 min)
- Set `HF_TOKEN` environment variable for gated model access
- Or pre-cache Llama-2-7B weights in the HF cache directory
- The agent could have run experiments on Mistral-7B (which was cached) if it
  had prioritized available models — a prompt improvement opportunity

---

## Comparison with Previous Run (all L0)

The previous run (before metadata fix) scored L0 on all 12 papers because:
- 11 papers cloned the **wrong repository** (e.g., bamboo-06079 cloned UMPL
  instead of VAR)
- Agents could not reconcile the claims (about VAR/ImageNet) with the code
  (about UMPL/surrogate modeling)
- 2 papers crashed due to the Deno FFI panic (same issue, pre-existing)

After the metadata fix, all 10 non-crashing papers now clone the correct repo.
7 of them achieve L1 (environment setup + build success). The remaining 3 L0s
(excluding crashes) appear to be agent-side issues with environment setup, not
data problems.

---

## Action Items

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| P0 | Add panic retry to `runner.py` (detect `jsruntime.rs` in stderr, retry 2×) | +2 papers from L0 → L1+ |
| P1 | Add flash_attn status to env probe output | Saves 10-15 min per LLM paper |
| P1 | Install CUDA 12.8 nvcc or pre-install flash_attn wheel | Unblocks native flash_attn |
| P1 | Set `HF_TOKEN` in runner environment | Unblocks gated model downloads |
| P2 | Increase timeout to 7200s for compute-heavy papers | bamboo-06086 likely reaches L2 |
| P2 | Pre-cache Traffic dataset for bamboo-06086 | Saves 25 min download time |
| P3 | Prompt improvement: prioritize cached models over downloading new ones | Helps bamboo-06085 reach L2 |
