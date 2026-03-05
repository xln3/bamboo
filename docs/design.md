# BAMBOO Benchmark Design Specification

## 1. Overview

BAMBOO (Benchmark for Autonomous ML Build-and-Output Observation) evaluates AI agents' ability to **fully reproduce** machine learning research using original open-source code. Given only a paper and its code repository, the agent must:

1. Read and understand the paper's full experimental section
2. Identify all quantitative claims (tables, figures, numerical results)
3. Set up the execution environment (dependencies, frameworks, data)
4. Run the experiments described in the paper
5. Verify whether the reproduced results match the paper's reported values

This mirrors the real-world reproduction workflow. Unlike PaperBench (from-scratch implementation) or CORE-Bench (pre-configured environments), BAMBOO tests the complete pipeline from paper comprehension to result verification using the authors' own code.

## 2. Dataset

### 2.1 Source Venues (2025 papers)

| Venue | Area | Conference Date |
|-------|------|-----------------|
| ICLR | ML/AI general | Apr 2025 |
| ICML | ML/AI general | Jul 2025 |
| NeurIPS | ML/AI general | Dec 2025 |
| CVPR | Computer vision | Jun 2025 |
| ICCV | Computer vision | Oct 2025 |
| ACL | NLP | Jul 2025 |
| EMNLP | NLP | Nov 2025 |
| AAAI | AI general | Feb 2025 |
| ICRA | Robotics | May 2025 |

**Scale**: Collect ALL accepted papers with open-source code from these 9 venues. No artificial cap — the dataset includes every qualifying paper.

### 2.2 Paper Selection Criteria

A paper is included if ALL of the following hold:

1. **Accepted at a target venue in 2025**
2. **Has open-source code** linked from the paper, arXiv, venue proceedings, or discoverable via Semantic Scholar / HuggingFace Papers
3. **Code is on a public platform** (GitHub, GitLab, HuggingFace, etc.)
4. **Contains quantitative experimental results** (tables or figures with numerical metrics)

Papers are EXCLUDED if:
- Code repository is empty, stub-only, or contains only model weights without runnable code
- Paper is a survey, position paper, or purely theoretical with no experiments
- Repository has been deleted or made private

Note: Papers requiring large compute, proprietary datasets, or specialized hardware are NOT excluded — they are included and tagged accordingly. The difficulty tier captures these constraints.

### 2.3 Task Definition

The agent's task is **full paper reproduction**:

1. **Paper comprehension**: Read the paper PDF, understand the methodology and experimental setup
2. **Claim discovery**: Identify ALL quantitative experimental results in the paper (every table, every figure with numbers, every inline numerical claim)
3. **Environment setup**: Clone the code, install dependencies, prepare data
4. **Experiment execution**: Run the experiments that produce the claimed results
5. **Result verification**: Compare reproduced values against paper-reported values

The agent receives ONLY:
- Paper PDF
- Code repository URL + pinned commit hash
- Hardware specification of the evaluation machine
- Time budget

The agent does NOT receive:
- Pre-extracted claims or metric values
- Hints about which scripts to run
- Ground truth numbers

### 2.4 Ground Truth Claims (for Evaluation)

For benchmark scoring, we maintain **auto-extracted ground truth claims** for each paper. These are used ONLY by the evaluation harness (never shown to the agent).

**Extraction process** (fully automated with spot-check):
1. LLM extracts ALL tables, figures, and inline numerical results from paper PDF
2. Each result becomes a claim: metric name, value, source location (table/figure/section), context
3. Automated consistency checks (e.g., numbers in abstract match numbers in tables)
4. Spot-check: random 5% sample verified by human

**Claim categories**:
- **Main results**: Primary experimental tables (typically Table 1-3)
- **Ablation**: Ablation study results
- **Analysis**: Additional analysis, scaling experiments, etc.
- **Baseline reproduction**: Re-reported baseline numbers (not original to this paper)

For scoring, all non-baseline claims contribute equally. Baseline claims are tracked but excluded from the primary score (since the agent is reproducing this paper's code, not baseline code).

### 2.5 Tolerance

Default ±5% relative tolerance. Adjusted by metric type:

| Metric type | Tolerance | Examples |
|-------------|-----------|----------|
| Deterministic | ±1% | Exact match, parameter count |
| Standard | ±5% | Accuracy, F1, BLEU, mAP |
| High-variance | ±10% | FID, IS, generative metrics |
| Timing/throughput | ±20% | Inference speed, training time |

### 2.6 Difficulty Stratification

Each paper receives a composite difficulty score (auto-computed from repo analysis):

| Factor | Weight | Scoring |
|--------|--------|---------|
| Framework complexity | 0.20 | 1 (pip install) → 5 (custom CUDA kernels) |
| Dependency count | 0.15 | 1 (< 10) → 5 (> 50) |
| Dataset requirements | 0.20 | 1 (built-in/small) → 5 (> 100GB or gated) |
| Hardware requirements | 0.20 | 1 (CPU-ok) → 5 (multi-GPU required) |
| Code quality | 0.10 | 1 (clean README + scripts) → 5 (no docs, broken paths) |
| Reproduction time | 0.15 | 1 (< 10 min) → 5 (> 24h) |

Tiers:
- **Tier 1 (Easy)**: 1.0-2.0
- **Tier 2 (Medium)**: 2.0-3.0
- **Tier 3 (Hard)**: 3.0-4.0
- **Tier 4 (Expert)**: 4.0-5.0

## 3. Evaluation Protocol

### 3.1 pass^4: Four-Level Progressive Evaluation

#### L1: Build
Can the agent set up a working environment?
- **Pass**: core imports succeed; or smoke test passes
- **Timeout**: 30 minutes

#### L2: Run
Can the agent execute experiments without crashes?
- **Pass**: at least one experiment produces output (exit code 0 or expected output files)
- **Timeout**: per-difficulty (Tier 1: 1h, Tier 2: 8h, Tier 3: 72h, Tier 4: 168h)

#### L3: Reproduce
Do results match the paper?
- **Full pass**: ≥ 80% of non-baseline claims within tolerance
- **Partial pass**: ≥ 50% of non-baseline claims within tolerance
- **Fail**: < 50%

#### L4: Cross-Hardware
Results consistent on different hardware? (Optional, future work)

### 3.2 Claim Coverage Score

In addition to pass^4, we evaluate how well the agent discovers claims:

- **Claim recall**: (claims found by agent) / (ground truth claims)
- **Claim precision**: (valid claims found) / (total claims reported by agent)

This evaluates the agent's paper comprehension ability separately from its reproduction ability.

### 3.3 Five-Barrier Failure Attribution

| Barrier | Examples | Auto-fixable? |
|---------|----------|---------------|
| L1: Environment | Python version, system libs | Often |
| L2: Build | Compilation failures | Sometimes |
| L3: Framework | PyTorch/TF API changes | Sometimes |
| L4: Hardware/Data | OOM, missing dataset | Rarely |
| L5: Microarch | CUDA capability, precision | No |

### 3.4 Aggregate Metrics

| Metric | Definition |
|--------|------------|
| **Build Rate (BR)** | % papers reaching L1 |
| **Run Rate (RR)** | % papers reaching L2 |
| **Reproduce Rate (ReR)** | % papers reaching L3 (full) |
| **Reproduce Rate Partial (ReR-P)** | % papers reaching L3 (partial+) |
| **Claim Coverage (CC)** | Mean claim recall across all papers |
| **Mean Reproduction Score (MRS)** | Mean (claims matched / claims attempted) |
| **Mean Relative Deviation (MRD)** | Mean |actual-expected|/|expected| |
| **Barrier Distribution** | % failures at each barrier level |
| **Paper Issue Rate** | % failures attributed to paper code |
| **Cost** | Mean LLM cost per paper |
| **Time** | Mean wall-clock time per paper |

All metrics reported overall and stratified by tier, venue, and domain.

## 4. Dataset Construction Pipeline

### Phase 1: Paper Collection
1. Scrape accepted paper lists from all 9 venues
2. For each paper, find associated code repositories via:
   - OpenReview metadata (ICLR, ICML, NeurIPS)
   - Venue proceedings pages
   - arXiv abstract/PDF (GitHub links)
   - Semantic Scholar API
   - HuggingFace Papers
3. Output: `(paper_id, title, arxiv_id, venue, code_url)` entries

### Phase 2: Repository Validation
1. Clone each repository, pin commit hash
2. Run PANDA's `analyzeRepo()`: framework, dependencies, entry points, GPU requirements
3. Filter: empty repos, stubs (< 100 LoC), no runnable code
4. Output: enriched paper entries with repo analysis

### Phase 3: Ground Truth Extraction
1. Download paper PDFs
2. LLM extracts all quantitative claims (tables, figures, inline numbers)
3. Auto-categorize claims (main/ablation/analysis/baseline)
4. Spot-check 5% sample
5. Output: paper entries with ground truth claims

### Phase 4: Difficulty Scoring
1. Compute per-factor scores from repo analysis
2. Compute composite score, assign tier
3. Output: final paper entries with all metadata

### Phase 5: Validation
1. Sample ~50 papers across tiers
2. Human reproduction attempt
3. Calibrate tolerance, verify ground truth quality

## 5. Agent Interface

### 5.1 Input
```json
{
  "paper_id": "bamboo-0001",
  "paper_pdf_path": "/data/papers/bamboo-0001.pdf",
  "code_url": "https://github.com/author/repo",
  "code_commit": "abc123def456",
  "hardware": {
    "gpu": "NVIDIA A100 80GB",
    "gpu_count": 1,
    "cpu_cores": 16,
    "ram_gb": 128
  },
  "time_limit_minutes": 480
}
```

### 5.2 Output
```json
{
  "paper_id": "bamboo-0001",
  "agent_id": "panda-v1",
  "claims_extracted": [
    {
      "source": "Table 1, row 3",
      "description": "CIFAR-10 test accuracy of proposed method",
      "metric_name": "accuracy",
      "metric_value_reported": 95.3,
      "metric_value_reproduced": 95.1,
      "reproduced": true
    }
  ],
  "pass4": {
    "l1_build": { "status": "pass", "duration_ms": 120000 },
    "l2_run":   { "status": "pass", "duration_ms": 3600000 },
    "l3_reproduce": { "status": "pass", "reproduction_score": 0.85 },
    "l4_cross": { "status": "skip" },
    "overall_level": 3
  },
  "barriers": [],
  "resource_usage": {
    "total_time_ms": 3720000,
    "llm_tokens_input": 150000,
    "llm_tokens_output": 30000,
    "llm_cost_usd": 2.50
  }
}
```

### 5.3 Constraints
- Isolated environment (Docker container)
- Internet access allowed (packages, datasets, model weights)
- No access to ground truth claims or other agents' results
- Time limit enforced

## 6. Limitations and Scope

### In scope
- 2025 papers with open-source code from 9 top AI venues
- Full reproduction: paper comprehension → claim discovery → environment setup → execution → verification
- Single-machine experiments (≤ 8 GPUs)

### Out of scope (future work)
- Papers without code (→ PaperBench)
- Non-AI papers (→ CORE-Bench)
- Distributed multi-node training
- L4 cross-hardware at scale
- Reproducibility auditing (→ REPRO-Bench)
