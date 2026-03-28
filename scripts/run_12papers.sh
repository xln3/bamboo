#!/usr/bin/env bash
# =============================================================================
# BAMBOO 12-Paper Sequential Runner
# =============================================================================
# Runs PANDA agent on 12 papers one-by-one to avoid rate limits.
# Full logs are saved to data/results/<agent_id>/logs/<paper_id>/
#
# Usage:
#   # Default: panda + glm-5-turbo, with judge
#   ./scripts/run_12papers.sh
#
#   # Custom model profile
#   ./scripts/run_12papers.sh --model claude-sonnet
#
#   # Custom agent + model
#   ./scripts/run_12papers.sh --agent opencode --model gpt-4o
#
#   # Skip judge (run judge later)
#   ./scripts/run_12papers.sh --skip-judge
#
#   # Resume from specific paper (skip already completed)
#   ./scripts/run_12papers.sh --resume bamboo-06084
#
#   # Dry run (preview only)
#   ./scripts/run_12papers.sh --dry-run
# =============================================================================

set -euo pipefail

BAMBOO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BAMBOO_ROOT"

# ── Defaults ──
AGENT="panda"
MODEL="glm-5-turbo"
DATASET="data/bamboo_curated"
TIMEOUT=3600
SKIP_JUDGE=""
DRY_RUN=""
RESUME_FROM=""
JUDGE_MODEL="opus"

# ── The 12 papers ──
PAPERS=(
  bamboo-06079  # VAR: Visual Autoregressive Modeling (NeurIPS 2024, 28 claims)
  bamboo-06080  # Infinity: Scaling Bitwise AR (arXiv 2024, 37 claims)
  bamboo-06081  # MaskedVQ: Not All Image Regions Matter (CVPR 2023, 28 claims)
  bamboo-06082  # Progressive Coding Ultra-Low Bitrate (ICLR 2025, 54 claims)
  bamboo-06083  # Customizable-ROI Compression (arXiv 2025, 64 claims)
  bamboo-06084  # SCALAR: Scale-wise Controllable VAR (arXiv 2025, 147 claims)
  bamboo-06085  # TEAL: Training-Free Activation Sparsity (NeurIPS 2024, 250 claims)
  bamboo-06086  # CATS: Self-Attentions for Time Series (NeurIPS 2024, 92 claims)
  bamboo-06087  # ARCQuant: NVFP4 Quantization (arXiv 2026, 79 claims)
  bamboo-06088  # FlatQuant: Flatness for LLM Quantization (NeurIPS 2024, 163 claims)
  bamboo-06089  # SmoothQuant: Post-Training Quantization (ICML 2023, 104 claims)
  bamboo-03009  # R-Sparse: Rank-Aware Activation Sparsity (ICLR 2025, 100 claims)
)

# ── Parse args ──
while [[ $# -gt 0 ]]; do
  case $1 in
    --agent)    AGENT="$2";       shift 2 ;;
    --model)    MODEL="$2";       shift 2 ;;
    --dataset)  DATASET="$2";     shift 2 ;;
    --timeout)  TIMEOUT="$2";     shift 2 ;;
    --skip-judge) SKIP_JUDGE="--skip-judge"; shift ;;
    --dry-run)  DRY_RUN="--dry-run"; shift ;;
    --resume)   RESUME_FROM="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    *)          echo "Unknown arg: $1"; exit 1 ;;
  esac
done

AGENT_ID="${AGENT}-${MODEL}"
LOG_ROOT="data/results/${AGENT_ID}"
RUN_LOG="${LOG_ROOT}/run_12papers.log"
mkdir -p "$LOG_ROOT"

# ── Header ──
{
  echo "============================================================"
  echo "  BAMBOO 12-Paper Run"
  echo "  Agent: ${AGENT}  Model: ${MODEL}"
  echo "  Dataset: ${DATASET}"
  echo "  Timeout: ${TIMEOUT}s per paper"
  echo "  Judge: ${JUDGE_MODEL} ${SKIP_JUDGE:+(skipped)}"
  echo "  Started: $(date -Iseconds)"
  echo "============================================================"
} | tee -a "$RUN_LOG"

# ── Sequential execution ──
COMPLETED=0
FAILED=0
SKIPPED=0
RESUMING=true

if [[ -z "$RESUME_FROM" ]]; then
  RESUMING=false
fi

for PAPER_ID in "${PAPERS[@]}"; do
  # Handle --resume: skip papers until we reach the resume point
  if $RESUMING; then
    if [[ "$PAPER_ID" == "$RESUME_FROM" ]]; then
      RESUMING=false
    else
      echo "[SKIP] $PAPER_ID (before resume point $RESUME_FROM)" | tee -a "$RUN_LOG"
      ((SKIPPED++))
      continue
    fi
  fi

  # Skip if result already exists (unless dry-run)
  RESULT_FILE="${LOG_ROOT}/${PAPER_ID}.json"
  if [[ -z "$DRY_RUN" && -f "$RESULT_FILE" ]]; then
    LEVEL=$(python3 -c "import json; r=json.load(open('$RESULT_FILE')); print(r.get('pass4',{}).get('overall_level','?'))" 2>/dev/null || echo "?")
    echo "[EXISTS] $PAPER_ID → L${LEVEL}, skipping (delete $RESULT_FILE to re-run)" | tee -a "$RUN_LOG"
    ((SKIPPED++))
    continue
  fi

  echo "" | tee -a "$RUN_LOG"
  echo "[$(date +%H:%M:%S)] Starting $PAPER_ID ($(( COMPLETED + FAILED + 1 ))/${#PAPERS[@]})" | tee -a "$RUN_LOG"

  START_TS=$(date +%s)

  python3 -m scripts.run.runner \
    --agents "$AGENT" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --papers "$PAPER_ID" \
    --timeout "$TIMEOUT" \
    --judge-model "$JUDGE_MODEL" \
    $SKIP_JUDGE \
    $DRY_RUN \
    2>&1 | tee -a "$RUN_LOG"

  EXIT_CODE=${PIPESTATUS[0]}
  END_TS=$(date +%s)
  ELAPSED=$(( END_TS - START_TS ))

  if [[ $EXIT_CODE -eq 0 ]]; then
    echo "[DONE] $PAPER_ID  (${ELAPSED}s)" | tee -a "$RUN_LOG"
    ((COMPLETED++))
  else
    echo "[FAIL] $PAPER_ID  exit=$EXIT_CODE  (${ELAPSED}s)" | tee -a "$RUN_LOG"
    ((FAILED++))
  fi

  # Brief pause between papers to be gentle on rate limits
  if [[ -z "$DRY_RUN" ]]; then
    echo "[WAIT] 10s cooldown before next paper..." | tee -a "$RUN_LOG"
    sleep 10
  fi
done

# ── Summary ──
{
  echo ""
  echo "============================================================"
  echo "  Run Complete: $(date -Iseconds)"
  echo "  Completed: $COMPLETED  Failed: $FAILED  Skipped: $SKIPPED"
  echo "  Total: ${#PAPERS[@]}"
  echo "============================================================"
  echo ""
  echo "Next steps:"
  echo "  # Evaluate results"
  echo "  python -m scripts.evaluate.evaluate \\"
  echo "    --results-dir data/results/${AGENT_ID}/ \\"
  echo "    --dataset ${DATASET} \\"
  echo "    --output data/results/${AGENT_ID}/report.json"
  echo ""
  echo "  # Run judge on all (if skipped earlier)"
  echo "  python -m scripts.evaluate.judge --agent ${AGENT_ID} --all"
} | tee -a "$RUN_LOG"
