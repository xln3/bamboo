# BAMBOO: Benchmark for Autonomous ML Build-and-Output Observation

A large-scale benchmark for evaluating AI agents' ability to reproduce machine learning research papers using their original open-source code.

## Motivation

Existing paper reproduction benchmarks either require agents to write code from scratch (PaperBench), use pre-configured environments (CORE-Bench), or only evaluate code generation without execution (Paper2Code). None captures the real-world workflow: **clone the authors' code, set up the environment, run experiments, and verify results match the paper**.

BAMBOO fills this gap with 1,000 papers from 10 top venues, evaluating the full pipeline from environment setup to numerical result verification.

## Key Features

- **End-to-end reproduction using original code** — not from-scratch implementation
- **Environment setup as a first-class challenge** — dependency resolution, framework compatibility, hardware adaptation
- **Four-level progressive evaluation (pass^4)** — build, run, reproduce, cross-hardware
- **Five-layer barrier model** — structured failure attribution (environment, build, framework, hardware, microarchitecture)
- **Large scale** — 1,000 papers across 10 venues (vs PaperBench's 20, CORE-Bench's 90)
- **2025 papers** — minimal data contamination risk
- **Multi-venue coverage** — NeurIPS, ICML, CVPR, ICCV, ACL, ECCV, AAAI, EMNLP, ICRA

## Directory Structure

```
bamboo/
├── README.md
├── docs/
│   ├── design.md                 # Benchmark design specification
│   ├── related-work.md           # Related work survey
│   └── evaluation-protocol.md    # Evaluation protocol details
├── schema/
│   ├── paper-entry.schema.json   # Paper entry metadata schema
│   └── result.schema.json        # Evaluation result schema
├── scripts/
│   ├── collect/                  # Paper collection & filtering
│   └── evaluate/                 # Evaluation harness
└── data/
    ├── papers/                   # Paper metadata entries
    └── results/                  # Agent evaluation results
```

## Quick Links

- [Design Document](docs/design.md)
- [Related Work](docs/related-work.md)
- [Evaluation Protocol](docs/evaluation-protocol.md)
