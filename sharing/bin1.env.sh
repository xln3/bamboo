#!/bin/bash
# Source this on the fit1-217 host before running runner.py / run_window.py
# All paths absolute, no /tmp use.

# ── Proxy (clash) ─────────────────────────────────────────
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# ── Bamboo runner overrides ───────────────────────────────
export BAMBOO_RUN_ROOT=/home/fit1-217/workplace/long/bamboo-run
export BAMBOO_WORKDIR_BASE=/home/fit1-217/workplace/long/workdirs
export BAMBOO_LOG_DIR=/home/fit1-217/workplace/long/bamboo-window-logs
export BAMBOO_PAPER_MARKDOWNS=$BAMBOO_RUN_ROOT/data/paper_markdowns
export BAMBOO_PROGRESS_FILE=/home/fit1-217/workplace/long/bamboo-run/progress.json

# ── /tmp ban ──────────────────────────────────────────────
export TMPDIR=/home/fit1-217/workplace/long/tmpdir
mkdir -p "$TMPDIR" "$BAMBOO_WORKDIR_BASE" "$BAMBOO_LOG_DIR"

# ── Panda / Deno ──────────────────────────────────────────
export PANDA_ROOT=/home/fit1-217/.panda
export DENO_BIN=/home/fit1-217/.deno/bin/deno
export PATH=/home/fit1-217/.local/bin:/home/fit1-217/.deno/bin:$PATH

# ── Hugging Face: shared cache on bbdata, mirror endpoint ─
export HF_HOME=/home/fit1-217/bbdata/bin1/hf-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_TELEMETRY=1

# ── Pre-downloaded asset hint ─────────────────────────────
export BAMBOO_ASSET_DIR=/home/fit1-217/bbdata/bin1

# ── Cross-run learning + judge ────────────────────────────
export PANDA_CROSS_RUN_LEARNING=1

cd "$BAMBOO_RUN_ROOT"
echo "env loaded: BAMBOO_RUN_ROOT=$BAMBOO_RUN_ROOT"
echo "             WORKDIR_BASE=$BAMBOO_WORKDIR_BASE"
echo "             HF_HOME=$HF_HOME"
echo "             PANDA_ROOT=$PANDA_ROOT"
