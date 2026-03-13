# BAMBOO Quick Start Guide

BAMBOO (Benchmark for Autonomous ML Build-and-Output Observation) is a benchmark for evaluating AI agents' ability to **reproduce machine learning papers** using their original open-source code. It tests the complete pipeline: paper comprehension → environment setup → experiment execution → result verification.

**Scale**: 6,148 papers across 9 top-tier AI venues (2025), all with verified code repositories.

---

## 1. Project Structure

```
bamboo/
├── data/
│   ├── bamboo_final.json          # Main dataset: 6,148 paper entries
│   ├── papers/                    # Raw per-venue paper metadata (JSON)
│   ├── curated/                   # Human-curated TSVs (for manual review)
│   └── results/                   # Agent evaluation results go here
├── schema/
│   ├── paper-entry.schema.json    # Paper entry schema
│   └── result.schema.json         # Agent output schema (must conform)
├── scripts/
│   ├── evaluate/                  # Evaluation harness
│   │   ├── evaluate.py            # Main evaluator
│   │   └── metrics.py             # Aggregate metrics
│   └── collect/                   # Dataset construction pipeline (already run)
└── docs/
    ├── design.md                  # Full benchmark specification
    └── evaluation-protocol.md     # Detailed evaluation protocol
```

## 2. Dataset Overview

### 2.1 Paper Entry Format

Each entry in `data/bamboo_final.json`:

```json
{
  "paper_id": "bamboo-00001",
  "title": "3D Annotation-Free Learning by ...",
  "venue": "AAAI",
  "year": 2025,
  "code_url": "https://github.com/sbysbysbys/AFOV",
  "code_commit": "a4fcda8b75dfe795b33d502272c7cce84ea168b6",
  "arxiv_id": "2405.15286",
  "authors": ["..."],
  "abstract": "...",
  "paper_url": "https://ojs.aaai.org/...",
  "pdf_url": "...",
  "venue_track": "main",
  "code_platform": "github",
  "domain": "robotics",
  "difficulty": null,
  "tags": []
}
```

### 2.2 Dataset Statistics

| Dimension | Breakdown |
|-----------|-----------|
| **Total papers** | 6,148 |
| **Venues** | ICML (1,758), ICLR (1,318), NeurIPS (947), ICCV (598), CVPR (437), ACL (399), EMNLP (351), AAAI (285), ICRA (55) |
| **Domains** | NLP (2,307), Vision (792), Other (640), Generative (573), Multimodal (522), Theory (452), RL (237), Graph (179), Robotics (137), Systems (126), Tabular (106), Audio (77) |
| **Code coverage** | 100% — all papers have verified `code_url` + pinned `code_commit` |

### 2.3 What Your Agent Receives

For each paper, provide your agent with:

| Input | Source | Notes |
|-------|--------|-------|
| Paper PDF | `pdf_url` or derive from `arxiv_id` | Agent reads the paper |
| Code repository URL | `code_url` | Agent clones the repo |
| Pinned commit hash | `code_commit` | Agent checks out this exact commit |
| Hardware specification | Your test environment | See hardware profiles below |
| Time budget | Per difficulty tier | Agent should respect the limit |

**Important**: Ground truth claims are **NOT** provided to the agent. The agent must independently identify and reproduce the paper's experimental results.

## 3. The pass^4 Evaluation Model

BAMBOO uses a 4-level progressive evaluation called **pass^4**:

```
L1 Build  →  L2 Run  →  L3 Reproduce  →  L4 Cross-Hardware
```

| Level | What it tests | Pass criteria | Timeout |
|-------|---------------|---------------|---------|
| **L1 Build** | Environment setup, imports work | `import <main_module>` succeeds | 30 min |
| **L2 Run** | Experiment runs without crash | Process exits 0, output files generated | 1h–168h |
| **L3 Reproduce** | Results match paper's claims | ≥80% claims within tolerance = full pass, ≥50% = partial | — |
| **L4 Cross-HW** | Consistent across hardware | L3 passes on 2 different hardware configs | (future) |

### Tolerance Thresholds

| Metric type | Tolerance | Examples |
|-------------|-----------|----------|
| Deterministic | ±1% | Exact match scores, parameter counts |
| Standard | ±5% | Accuracy, F1, BLEU, mAP |
| High-variance | ±10% | FID, IS, generative metrics |
| Timing | ±20% | Training time, inference speed |

## 4. Agent Output Format

Your agent must produce one JSON file **per paper**, conforming to `schema/result.schema.json`.

### 4.1 Minimal Result Example

```json
{
  "paper_id": "bamboo-00001",
  "agent_id": "my-agent",
  "agent_version": "1.0.0",
  "timestamp": "2025-03-13T12:00:00Z",
  "hardware_profile": "medium",
  "pass4": {
    "l1_build": {
      "status": "pass",
      "duration_ms": 120000,
      "detail": "All imports successful after installing requirements"
    },
    "l2_run": {
      "status": "pass",
      "duration_ms": 3600000,
      "detail": "Training completed, results saved to output/"
    },
    "l3_reproduce": {
      "status": "pass",
      "claim_results": [
        {
          "claim_id": "c1",
          "expected_value": 76.5,
          "actual_value": 76.2,
          "relative_deviation": 0.0039,
          "pass": true,
          "extraction_method": "log_parse"
        },
        {
          "claim_id": "c2",
          "expected_value": 82.3,
          "actual_value": 81.9,
          "relative_deviation": 0.0049,
          "pass": true,
          "extraction_method": "structured"
        }
      ],
      "reproduction_score": 1.0
    },
    "l4_cross": {
      "status": "skip"
    },
    "overall_level": 3
  },
  "barriers": [],
  "failure_attribution": null,
  "resource_usage": {
    "total_time_ms": 3720000,
    "llm_tokens_input": 150000,
    "llm_tokens_output": 25000,
    "llm_cost_usd": 2.50,
    "gpu_hours": 1.0,
    "peak_gpu_memory_gb": 18.5
  },
  "logs": {
    "setup_log_path": "logs/bamboo-00001/setup.log",
    "run_log_path": "logs/bamboo-00001/run.log",
    "environment_spec_path": "logs/bamboo-00001/pip_freeze.txt",
    "agent_transcript_path": "logs/bamboo-00001/transcript.json"
  }
}
```

### 4.2 Failure Result Example

```json
{
  "paper_id": "bamboo-00042",
  "agent_id": "my-agent",
  "agent_version": "1.0.0",
  "timestamp": "2025-03-13T14:00:00Z",
  "pass4": {
    "l1_build": {
      "status": "pass",
      "duration_ms": 90000
    },
    "l2_run": {
      "status": "fail",
      "duration_ms": 300000,
      "detail": "CUDA out of memory on RTX 3090 (24GB), model requires ~40GB"
    },
    "l3_reproduce": { "status": "skip" },
    "l4_cross": { "status": "skip" },
    "overall_level": 1
  },
  "barriers": [
    {
      "level": "L4_hardware_code",
      "description": "GPU memory insufficient for default batch size",
      "evidence": "RuntimeError: CUDA out of memory. Tried to allocate 2.3 GiB",
      "auto_fixable": true,
      "suggested_fix": "Reduce batch_size from 64 to 16 or use gradient accumulation"
    }
  ],
  "failure_attribution": {
    "primary_barrier": {
      "level": "L4_hardware_code",
      "description": "GPU memory insufficient",
      "evidence": "CUDA OOM"
    },
    "likely_paper_issue": false,
    "summary": "Paper requires >24GB VRAM; code has no memory-saving options"
  },
  "resource_usage": {
    "total_time_ms": 390000,
    "llm_cost_usd": 0.80
  }
}
```

### 4.3 Key Schema Details

**`status` values**: `"pass"`, `"partial"`, `"fail"`, `"skip"`, `"timeout"`

**`barrier.level` values** (5-layer barrier model):
| Barrier | Description |
|---------|-------------|
| `L1_environment` | Missing system libraries, OS-level issues |
| `L2_build` | Compilation errors, dependency conflicts |
| `L3_framework` | Framework API changes, version incompatibilities |
| `L4_hardware_code` | CUDA OOM, dataset download failure, hardcoded paths |
| `L5_microarch` | GPU architecture issues, driver incompatibilities |

**`extraction_method` values**: `"structured"`, `"regex"`, `"log_parse"`, `"llm"`, `"manual"`

## 5. Running Evaluation

### 5.1 Directory Setup

Place your agent's result files in a directory:

```
data/results/my-agent/
├── bamboo-00001.json
├── bamboo-00002.json
├── bamboo-00003.json
└── ...
```

Each file must contain a valid result JSON with `paper_id` matching the filename.

### 5.2 Run the Evaluator

```bash
cd bamboo/

python -m scripts.evaluate.evaluate \
    --results-dir data/results/my-agent/ \
    --dataset data/bamboo_final.json \
    --output data/results/my-agent/report.json
```

### 5.3 Stratified Reports

Stratify by venue, tier, or domain:

```bash
# Stratify by venue only
python -m scripts.evaluate.evaluate \
    --results-dir data/results/my-agent/ \
    --dataset data/bamboo_final.json \
    --output data/results/my-agent/report.json \
    --by venue

# Default (no --by flag) includes all three stratifications
```

### 5.4 Sample Output

```
================================================================
  BAMBOO Evaluation Report
================================================================

  Papers evaluated: 100

  pass^4 Rates:
    L1 Build Rate:           72.0%
    L2 Run Rate:             45.0%
    L3 Reproduce Rate:       18.0%
    L3 Reproduce (partial+): 31.0%

  Claim Metrics:
    Claim Coverage (recall): 0.650
    Mean Reproduction Score: 0.420
    Mean Relative Deviation: 0.0832

  Failure Analysis:
    L1_environment: 28.0%
    L2_build: 22.0%
    L3_framework: 35.0%
    L4_hardware_code: 12.0%
    L5_microarch: 3.0%
    Paper Issue Rate: 15.0%

  Resources:
    LLM Cost: mean=$3.20  median=$2.80  total=$320.00  (100 papers)
    Wall Time: mean=4500s  median=3600s  total=450000s  (100 papers)

  Stratified by venue:
    Stratum                 N     BR     RR    ReR  ReR-P     CC    MRS
    -------------------- ---- ------ ------ ------ ------ ------ ------
    AAAI                   10 80.0% 50.0% 20.0% 30.0% 0.700 0.450
    ICML                   25 68.0% 40.0% 16.0% 28.0% 0.620 0.380
    ...
================================================================
```

### 5.5 Aggregate Metrics Explained

| Metric | Abbr | What it measures |
|--------|------|------------------|
| Build Rate | BR | % of papers where environment setup succeeded |
| Run Rate | RR | % of papers where experiments ran to completion |
| Reproduce Rate | ReR | % of papers where ≥80% claims match (**primary ranking metric**) |
| Reproduce Rate Partial | ReR-P | % of papers where ≥50% claims match |
| Claim Coverage | CC | Mean recall of ground-truth claims found by agent |
| Mean Reproduction Score | MRS | Mean fraction of claims within tolerance (**secondary ranking metric**) |
| Mean Relative Deviation | MRD | Mean |actual−expected|/|expected| across all claims |
| Paper Issue Rate | PIR | % of failures attributed to the paper's code quality |

## 6. Hardware Profiles

| Profile | GPU | VRAM | Recommended for |
|---------|-----|------|-----------------|
| `small` | 1× RTX 3090/4090 | 24 GB | Lightweight models, small datasets |
| `medium` | 1× A100 | 80 GB | Most papers (default) |
| `large` | 4× A100 | 320 GB | Multi-GPU training, large models |
| `xlarge` | 8× A100 | 640 GB | LLM-scale training |

## 7. End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Select papers from bamboo_final.json                    │
│     (all 6,148 or a subset by venue/domain)                 │
├─────────────────────────────────────────────────────────────┤
│  2. For each paper, your agent:                             │
│     a) Downloads the paper PDF                              │
│     b) Clones the repo at the pinned commit                 │
│     c) Reads the paper and code to understand the setup     │
│     d) Sets up the environment (conda/pip/docker)           │
│     e) Runs the experiments                                 │
│     f) Extracts metric values from output                   │
│     g) Reports results in the required JSON schema          │
├─────────────────────────────────────────────────────────────┤
│  3. Collect all per-paper JSONs into a results directory    │
├─────────────────────────────────────────────────────────────┤
│  4. Run the evaluation harness                              │
│     python -m scripts.evaluate.evaluate ...                 │
├─────────────────────────────────────────────────────────────┤
│  5. Review the report: pass^4 rates, claim metrics,         │
│     barrier distribution, cost efficiency                   │
└─────────────────────────────────────────────────────────────┘
```

## 8. Quick Test: Validate Your Integration

To test that your agent output integrates correctly with the evaluation harness, create a mock result and run the evaluator:

```bash
# Create a test results directory
mkdir -p data/results/test-agent

# Create a minimal test result
cat > data/results/test-agent/bamboo-00001.json << 'EOF'
{
  "paper_id": "bamboo-00001",
  "agent_id": "test-agent",
  "timestamp": "2025-03-13T00:00:00Z",
  "pass4": {
    "l1_build": {"status": "pass", "duration_ms": 60000},
    "l2_run": {"status": "pass", "duration_ms": 600000},
    "l3_reproduce": {"status": "pass", "claim_results": [], "reproduction_score": 0.0},
    "l4_cross": {"status": "skip"},
    "overall_level": 3
  },
  "resource_usage": {"total_time_ms": 660000}
}
EOF

# Run evaluation
python -m scripts.evaluate.evaluate \
    --results-dir data/results/test-agent/ \
    --dataset data/bamboo_final.json \
    --output data/results/test-agent/report.json

# Clean up test data
rm -rf data/results/test-agent/
```

## 9. Selecting a Paper Subset

To work with a subset (e.g., for development/debugging):

```python
import json

with open("data/bamboo_final.json") as f:
    papers = json.load(f)

# Filter by venue
icml_papers = [p for p in papers if p["venue"] == "ICML"]

# Filter by domain
vision_papers = [p for p in papers if p["domain"] == "vision"]

# Take first N papers
subset = papers[:50]

# Save subset for testing
with open("data/bamboo_subset.json", "w") as f:
    json.dump(subset, f, indent=2)
```

Then run evaluation against the subset:

```bash
python -m scripts.evaluate.evaluate \
    --results-dir data/results/my-agent/ \
    --dataset data/bamboo_subset.json \
    --output data/results/my-agent/report_subset.json
```

## 10. Current Status & Limitations

### What's Ready
- **Dataset**: 6,148 papers with verified code URLs and pinned commits
- **Evaluation harness**: Fully functional claim matching, metrics, stratified reporting
- **Schema definitions**: Both paper entry and result schemas are complete
- **Documentation**: Design spec, evaluation protocol, related work survey

### Not Yet Populated
- **`difficulty`**: Tier/composite difficulty scores are not yet computed (all null)
- **`ground_truth_claims`**: LLM-based claim extraction has not been run at scale; the evaluator can still function using agent self-reported claims, but claim-level evaluation (L3 Reproduce) will fall back to agent self-report when no ground truth is available
- **`abstract`**: 2,125 papers missing abstracts (35%)
- **L4 Cross-Hardware**: Deferred to future iterations

### Implications for Current Use
Without ground truth claims, the evaluator relies on agent self-report for L3 status. To get full evaluation power:
1. Run `scripts/collect/extract_claims.py` to extract claims (requires OpenAI-compatible LLM API)
2. Or manually curate claims for a small subset of papers

## 11. Leaderboard Ranking

Agents are ranked by:

1. **Primary**: Reproduce Rate (ReR) — higher is better
2. **Secondary**: Mean Reproduction Score (MRS) — higher is better
3. **Tertiary**: Cost per L3 pass — lower is better

---

For the full benchmark specification, see [docs/design.md](design.md). For detailed evaluation rules, see [docs/evaluation-protocol.md](evaluation-protocol.md).
