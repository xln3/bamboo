# BAMBOO: Benchmark for Autonomous ML Build-and-Output Observation

A large-scale benchmark for evaluating AI agents' ability to reproduce machine learning research papers using their original open-source code.

## Motivation

Existing paper reproduction benchmarks either require agents to write code from scratch (PaperBench), use pre-configured environments (CORE-Bench), or only evaluate code generation without execution (Paper2Code). None captures the real-world workflow: **clone the authors' code, set up the environment, run experiments, and verify results match the paper**.

BAMBOO fills this gap with **6,148 papers** from 9 top AI venues, evaluating the full pipeline from environment setup to numerical result verification.

## Dataset Status

| Metric | Value |
|--------|-------|
| **Papers** | 6,148 (curated subset: 3,994) |
| **Venues** | 9 (ICML, ICLR, NeurIPS, ICCV, CVPR, ACL, EMNLP, AAAI, ICRA) |
| **Code coverage** | 100% (verified `code_url` + pinned `code_commit`) |
| **Abstracts** | 100% |
| **Difficulty scores** | 100% (6-dimension scoring, Tier 2: 6,032 / Tier 3: 116) |
| **Domains** | 12 categories (NLP 45%, Vision 15%, Generative 9%, Theory 7%, Multimodal 7%, ...) |
| **Paper PDFs** | 5,618 downloaded |
| **Paper markdowns** | 3,994 (MinerU PDF→Markdown extraction) |
| **Ground truth claims** | 2,658 papers with inline claims; 709 paper_claims_v2 files |
| **HuggingFace** | Synced to [xln3/bamboo-papers](https://huggingface.co/datasets/xln3/bamboo-papers) |

## Key Features

- **End-to-end reproduction using original code** — not from-scratch implementation
- **Environment setup as a first-class challenge** — dependency resolution, framework compatibility, hardware adaptation
- **Four-level progressive evaluation (pass^4)** — Build → Run → Reproduce → Cross-Hardware
- **Five-layer barrier model** — structured failure attribution
- **Large scale** — 6,148 papers (vs PaperBench's 20, CORE-Bench's 90)
- **2025 papers** — minimal data contamination risk
- **MinerU-powered PDF extraction** — high-quality text/table extraction for claim identification

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
│   │   ├── pdf_extractor.py      # MinerU-based PDF extraction
│   │   ├── extract_claims.py     # LLM-based claim extraction
│   │   ├── hf_sync.py            # HuggingFace dataset sync (push/pull)
│   │   ├── compute_difficulty.py # Difficulty scoring
│   │   ├── fetch_arxiv_abstracts.py  # arXiv abstract fetching
│   │   └── ...                   # Collection & validation scripts
│   ├── run/                      # Agent runner framework
│   │   ├── runner.py             # Comparative agent runner
│   │   ├── prompt_builder.py     # Prompt construction
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
    ├── bamboo_curated.json       # Curated subset (3,994 papers with markdowns + claims)
    ├── paper_pdfs/               # Downloaded PDFs (5,618)
    ├── paper_markdowns/          # MinerU PDF→Markdown (3,994)
    ├── paper_claims_v2/          # Extracted claims per paper (709)
    ├── curated/                  # Human-editable venue TSVs
    └── results/                  # Agent evaluation results
```

## Quick Start

See [docs/quick-start.md](docs/quick-start.md) (English) or [docs/usage.md](docs/usage.md) (Chinese) for the full guide.

### 1. Configure models

```bash
cp configs/models.example.json configs/models.json
# Edit configs/models.json — add your API keys
```

### 2. Run an agent on a paper

```bash
python3 -m scripts.run.runner \
  --agents panda \
  --model glm-5 \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06089 \
  --timeout 1800
```

Supported agents: `panda`, `claude-code`, `opencode`, `codex`. See [docs/usage.md](docs/usage.md) for how to add your own.

### 3. Sync data with HuggingFace

```bash
python3 scripts/collect/hf_sync.py status   # see what's new
python3 scripts/collect/hf_sync.py pull      # download missing files
python3 scripts/collect/hf_sync.py push      # upload local files
```

### 4. Run evaluation

```bash
python3 -m scripts.evaluate.evaluate \
    --results-dir data/results/panda-glm-5/ \
    --dataset data/bamboo_curated.json \
    --output data/results/panda-glm-5/report.json
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
