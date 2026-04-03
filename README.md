# BAMBOO: Benchmark for Autonomous ML Build-and-Output Observation

A large-scale benchmark for evaluating AI agents' ability to reproduce machine learning research papers using their original open-source code.

## Motivation

Existing paper reproduction benchmarks either require agents to write code from scratch (PaperBench), use pre-configured environments (CORE-Bench), or only evaluate code generation without execution (Paper2Code). None captures the real-world workflow: **clone the authors' code, set up the environment, run experiments, and verify results match the paper**.

BAMBOO fills this gap with **6,148 papers** from 9 top AI venues, evaluating the full pipeline from environment setup to numerical result verification.

## Dataset Status

| Metric | Value |
|--------|-------|
| **Papers** | 6,148 |
| **Venues** | 9 (ICML, ICLR, NeurIPS, ICCV, CVPR, ACL, EMNLP, AAAI, ICRA) |
| **Code coverage** | 100% (verified `code_url` + pinned `code_commit`) |
| **Abstracts** | 100% |
| **Difficulty scores** | 100% (6-dimension scoring, Tier 2: 6,032 / Tier 3: 116) |
| **Domains** | 12 categories (NLP 45%, Vision 15%, Generative 9%, Theory 7%, Multimodal 7%, ...) |
| **Paper PDFs** | 6,064 downloaded |
| **Paper markdowns** | 7,031 (MinerU PDF→Markdown extraction) |
| **Ground truth claims** | 6,051 paper_claims_v2 files |
| **HuggingFace** | Synced to [xln3/bamboo-papers](https://huggingface.co/datasets/xln3/bamboo-papers) |

## Key Features

- **End-to-end reproduction using original code** — not from-scratch implementation
- **Environment setup as a first-class challenge** — dependency resolution, framework compatibility, hardware adaptation
- **Four-level progressive evaluation (pass^4)** — Build → Run → Reproduce → Cross-Hardware
- **Five-layer barrier model** — structured failure attribution
- **Large scale** — 6,148 papers (vs PaperBench's 20, CORE-Bench's 90)
- **2025 papers** — minimal data contamination risk
- **MinerU-powered PDF extraction** — high-quality text/table extraction for claim identification
- **Stable paper IDs** — persistent ID mapping ensures IDs never shuffle across re-runs

## Directory Structure

```
bamboo/
├── README.md
├── CHANGELOG.md
├── configs/
│   ├── models.example.json       # Model config template (safe to commit)
│   └── models.json               # Live model configs with API keys (git-ignored)
├── docs/
│   ├── design.md                 # Benchmark design specification
│   ├── quick-start.md            # Getting started guide (English)
│   ├── usage.md                  # Comprehensive usage guide (Chinese)
│   ├── evaluation-protocol.md    # Evaluation protocol details
│   ├── related-work.md           # Related work survey
│   └── panda-architecture.md     # PANDA agent system design
├── schema/
│   ├── paper-entry.schema.json   # Paper entry metadata schema
│   └── result.schema.json        # Evaluation result schema
├── scripts/
│   ├── collect/                  # Data collection pipeline
│   │   ├── finalize_dataset.py   # Build bamboo_final.json with stable IDs
│   │   ├── pdf_extractor.py      # MinerU-based PDF extraction
│   │   ├── extract_claims.py     # LLM-based claim extraction
│   │   ├── hf_sync.py            # HuggingFace dataset sync (push/pull)
│   │   ├── compute_difficulty.py # Difficulty scoring
│   │   └── ...                   # Collection & validation scripts
│   ├── run/                      # Agent runner framework
│   │   ├── runner.py             # Comparative agent runner
│   │   ├── prompt_builder.py     # Prompt construction (bare/neutral/guided tiers)
│   │   └── agents/               # Agent adapters
│   │       ├── panda.py          # PANDA agent
│   │       ├── claude_code.py    # Claude Code
│   │       ├── opencode.py       # OpenCode
│   │       └── codex.py          # Codex
│   └── evaluate/                 # Evaluation harness
│       ├── evaluate.py           # Main evaluation script
│       ├── judge.py              # Independent claim judge (anti-leak)
│       └── metrics.py            # Aggregate metrics
└── data/
    ├── bamboo_final.json         # Full dataset (6,148 papers)
    ├── bamboo_curated/           # 62 chunked JSON files (100 papers each)
    ├── id_mapping.json           # Persistent paper_url → bamboo-XXXXX mapping
    ├── benchmark_extras.json     # Extra papers outside tracked venues
    ├── paper_pdfs/               # Downloaded PDFs (6,064) — via HuggingFace
    ├── paper_markdowns/          # MinerU PDF→Markdown (7,031) — via HuggingFace
    ├── paper_claims_v2/          # Extracted claims per paper (6,051) — via HuggingFace
    ├── curated/                  # Human-editable venue TSVs
    └── results/                  # Agent evaluation results
```

## Quick Start

See [docs/quick-start.md](docs/quick-start.md) (English) or [docs/usage.md](docs/usage.md) (Chinese) for the full guide.

### 1. Clone and download data

```bash
git clone https://github.com/xln3/bamboo.git
cd bamboo
pip install huggingface_hub

# Download large files from HuggingFace (PDFs, markdowns, claims)
python3 scripts/collect/hf_sync.py pull

# Or download only what you need:
python3 scripts/collect/hf_sync.py pull --only paper_claims_v2              # claims only (~small)
python3 scripts/collect/hf_sync.py pull --only paper_markdowns              # markdowns only
python3 scripts/collect/hf_sync.py pull --only paper_claims_v2,paper_markdowns  # both, no PDFs
```

### 2. Configure models

```bash
cp configs/models.example.json configs/models.json
# Edit configs/models.json — add your API keys
```

### 3. Run an agent on a paper

```bash
# Run on a single paper (uses bamboo_final.json by default)
python3 -m scripts.run.runner \
  --agents panda \
  --model glm-5 \
  --papers bamboo-00001 \
  --timeout 1800

# Run on a 100-paper subset
python3 -m scripts.run.runner \
  --agents panda \
  --model glm-5 \
  --dataset data/bamboo_curated/bamboo-00001_to_bamboo-00100.json \
  --timeout 1800
```

Supported agents: `panda`, `claude-code`, `opencode`, `codex`. See [docs/usage.md](docs/usage.md) for how to add your own.

### 4. Run evaluation

```bash
python3 -m scripts.evaluate.evaluate \
    --results-dir data/results/panda-glm-5/ \
    --dataset data/bamboo_final.json \
    --output data/results/panda-glm-5/report.json
```

### 5. Sync data with HuggingFace

```bash
python3 scripts/collect/hf_sync.py status   # see what's new
python3 scripts/collect/hf_sync.py pull      # download missing files
python3 scripts/collect/hf_sync.py push      # upload local files
```

## Quick Links

- [Design Document](docs/design.md)
- [Quick Start Guide (English)](docs/quick-start.md)
- [Usage Guide (Chinese)](docs/usage.md)
- [Evaluation Protocol](docs/evaluation-protocol.md)
- [PANDA Architecture](docs/panda-architecture.md)
- [Related Work](docs/related-work.md)
- [CHANGELOG](CHANGELOG.md)
- [HuggingFace Dataset](https://huggingface.co/datasets/xln3/bamboo-papers)
