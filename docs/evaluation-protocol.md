# BAMBOO Evaluation Protocol

## 1. Execution Environment

### 1.1 Sandboxing

Each evaluation run executes in an isolated environment:
- **Docker container** with a base image (Ubuntu 22.04 + CUDA 12.x + common tools)
- **Resource limits**: configurable per-tier (CPU cores, RAM, GPU count, disk)
- **Network**: allowed (for package installation and dataset download)
- **Time limit**: enforced via wall-clock timeout

### 1.2 Base Environment

The container provides:
- Ubuntu 22.04 LTS
- CUDA 12.4 + cuDNN 9
- Python 3.10, 3.11, 3.12 (via pyenv)
- conda/mamba, uv, pip
- git, wget, curl, unzip, cmake, gcc/g++
- Common system libraries (libgl1, libglib2.0, ffmpeg, etc.)

The agent must handle everything else: Python environment creation, package installation, data preparation.

### 1.3 Hardware Profiles

| Profile | GPU | VRAM | CPU | RAM | Disk | Use |
|---------|-----|------|-----|-----|------|-----|
| **small** | 1x RTX 3090/4090 | 24GB | 8 cores | 64GB | 500GB | Tier 1-2 |
| **medium** | 1x A100 | 80GB | 16 cores | 128GB | 1TB | Tier 2-3 |
| **large** | 4x A100 | 320GB | 64 cores | 512GB | 2TB | Tier 3-4 |
| **xlarge** | 8x A100 | 640GB | 128 cores | 1TB | 4TB | Tier 4 |

Default profile is **medium**. Paper entries specify minimum required profile.

## 2. Level-by-Level Protocol

### 2.1 L1: Build

**Objective**: Environment is set up; core imports succeed.

**Steps**:
1. Agent clones the repository at the pinned commit
2. Agent reads README, requirements, and documentation
3. Agent creates a virtual environment and installs dependencies
4. Agent runs a smoke test (import check or designated test)

**Pass criteria** (any of):
- `python -c "import <main_module>"` exits with code 0
- A designated smoke test script passes
- Agent reports successful setup with evidence (pip freeze showing key packages)

**Fail signals**:
- ImportError, ModuleNotFoundError
- Compilation failure (C/C++/CUDA extension build)
- Dependency conflict that cannot be resolved

**Timeout**: 30 minutes

**Barrier classification on failure**:
- Missing system library → L1_environment
- Compilation error → L2_build
- Framework version conflict → L3_framework
- CUDA/driver mismatch → L5_microarch

### 2.2 L2: Run

**Objective**: The experiment runs to completion without crashing.

**Steps**:
1. Agent identifies the correct command to run (from README, scripts, or paper)
2. Agent prepares data (downloads datasets, generates synthetic data, etc.)
3. Agent executes the experiment
4. Agent monitors for crashes, OOM, and hangs

**Pass criteria** (any of):
- Process exits with code 0
- Expected output files are generated (model checkpoints, result CSVs, figures)
- Output contains expected completion markers ("Training complete", "Results saved to")

**Fail signals**:
- Non-zero exit code
- Segfault, OOM kill, CUDA error
- Timeout exceeded
- No output files generated

**Timeout**: per-claim (quick: 1h, medium: 8h, long: 72h, prohibitive: 168h)

**Barrier classification on failure**:
- CUDA OOM → L4_hardware_code
- Dataset download failure → L4_hardware_code
- Runtime error in framework API → L3_framework
- FileNotFoundError for hardcoded paths → likelyPaperIssue

### 2.3 L3: Reproduce

**Objective**: Output metrics match paper-reported values within tolerance.

**Steps**:
1. Extract metrics from experiment output (automated)
2. Compare each claim's expected value with actual value
3. Compute relative deviation: `|actual - expected| / |expected|`
4. Determine pass/fail per claim

**Pass criteria**:
- **Full pass**: ALL primary claims within tolerance
- **Partial pass**: ≥ 50% of primary claims within tolerance
- **Fail**: < 50% of primary claims within tolerance

**Metric extraction cascade**:
1. Agent's structured output (if provided)
2. Regex patterns for common metric formats
3. Log file parsing (TensorBoard, W&B, CSV output)
4. LLM-based extraction as fallback

**Edge cases**:
- If metric is reported as a range in the paper, use the midpoint
- If paper reports "best of N runs", use ±10% tolerance instead of ±5%
- If paper uses a different evaluation protocol than what's in the code, flag for manual review

### 2.4 L4: Cross-Hardware

**Objective**: Results are consistent across different hardware.

**Steps**:
1. Re-execute L1-L3 on a different hardware profile
2. Compare L3 results across hardware configurations

**Pass criteria**:
- L3 passes on both hardware configurations
- Inter-hardware metric deviation < 2x claim tolerance

**Status**: Optional. Deferred to future iterations of BAMBOO.

## 3. Scoring

### 3.1 Per-Paper Score

Each paper receives an **overall level** (0-4):
- Level 0: L1 failed (cannot build)
- Level 1: L1 passed, L2 failed (builds but crashes)
- Level 2: L2 passed, L3 failed (runs but wrong results)
- Level 3: L3 passed (results match paper)
- Level 4: L4 passed (cross-hardware consistent)

And a **reproduction score** (0-1) for L3:
- `reproduction_score = (claims_within_tolerance) / (total_primary_claims)`

### 3.2 Aggregate Scores

| Metric | Formula |
|--------|---------|
| Build Rate (BR) | papers_at_L1+ / total_papers |
| Run Rate (RR) | papers_at_L2+ / total_papers |
| Reproduce Rate (ReR) | papers_at_L3+ / total_papers |
| Reproduce Rate Partial (ReR-P) | papers_at_L3_partial+ / total_papers |
| Mean Reproduction Score (MRS) | mean(reproduction_score) over L2+ papers |
| Mean Relative Deviation (MRD) | mean(|actual - expected| / |expected|) over all extracted metrics |

### 3.3 Stratified Reporting

All metrics reported as:
- **Overall**: across entire benchmark
- **Per-tier**: Tier 1/2/3/4
- **Per-venue**: NeurIPS, ICML, CVPR, etc.
- **Per-domain**: vision, NLP, robotics, general ML
- **Per-framework**: PyTorch, TensorFlow, JAX, etc.

### 3.4 Cost-Efficiency Metrics

| Metric | Definition |
|--------|------------|
| Cost per L3 pass | total_llm_cost / papers_at_L3+ |
| Tokens per paper | mean LLM tokens consumed per paper |
| Time per paper | mean wall-clock time per paper |
| Cost-adjusted ReR | ReR / mean_cost_per_paper |

## 4. Preventing Gaming

### 4.1 Anti-Cheating Measures

- Agent does NOT receive ground truth metric values
- Agent cannot access other agents' results or logs
- Claims include decoy metrics (unused in scoring) to detect value-fitting
- Random subset (10%) uses perturbed ground truth for calibration
- Output logs are archived for post-hoc audit

### 4.2 Contamination Prevention

- All papers are from 2025 (after most LLM training cutoffs)
- Commit hashes are pinned; agents cannot use newer code versions
- Evaluation harness checks that agent did not download pre-computed results

## 5. Submission Format

### 5.1 Agent Submission

An agent submission consists of:
1. **Agent code/container**: reproducible agent implementation
2. **Configuration**: model, scaffolding, prompts used
3. **Results file**: JSON array of per-paper results (schema in `schema/result.schema.json`)
4. **Cost report**: total LLM API costs, compute costs

### 5.2 Leaderboard

Public leaderboard ranks agents by:
1. Primary: **Reproduce Rate (ReR)**
2. Secondary: **Mean Reproduction Score (MRS)**
3. Tertiary: **Cost per L3 pass** (lower is better)

With stratified breakdowns for analysis.
