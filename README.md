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
| **Ground truth claims** | Pending LLM extraction (pipeline ready, needs `OPENAI_API_KEY`) |

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
├── docs/
│   ├── design.md                 # Benchmark design specification
│   ├── quick-start.md            # Getting started guide
│   ├── evaluation-protocol.md    # Evaluation protocol details
│   └── related-work.md           # Related work survey
├── schema/
│   ├── paper-entry.schema.json   # Paper entry metadata schema
│   └── result.schema.json        # Evaluation result schema
├── scripts/
│   ├── collect/                  # Data collection pipeline
│   │   ├── pdf_extractor.py      # MinerU-based PDF extraction
│   │   ├── extract_claims.py     # LLM-based claim extraction
│   │   ├── compute_difficulty.py # Difficulty scoring
│   │   ├── fetch_arxiv_abstracts.py  # arXiv abstract fetching
│   │   └── ...                   # Collection & validation scripts
│   └── evaluate/                 # Evaluation harness
│       ├── evaluate.py           # Main evaluation script
│       └── metrics.py            # Aggregate metrics
└── data/
    ├── bamboo_final.json         # Final dataset (6,148 papers)
    ├── curated/                  # Human-editable venue TSVs
    └── results/                  # Agent evaluation results
```

## Quick Start

See [docs/quick-start.md](docs/quick-start.md) for the full guide.

### Extract Ground Truth Claims (requires LLM API)

```bash
export OPENAI_API_KEY="your-key"
cd scripts/collect
python extract_claims.py --limit 100  # test on 100 papers first
python extract_claims.py              # run on all papers
```

### Run Evaluation

```bash
python -m scripts.evaluate.evaluate \
    --results-dir data/results/your-agent/ \
    --dataset data/bamboo_final.json \
    --output data/results/your-agent/report.json
```

## Quick Links

- [Design Document](docs/design.md)
- [Quick Start Guide](docs/quick-start.md)
- [Evaluation Protocol](docs/evaluation-protocol.md)
- [Related Work](docs/related-work.md)
- [CHANGELOG](CHANGELOG.md)
