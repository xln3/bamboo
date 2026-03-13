# CHANGELOG

## 2026-03-14

### Dataset
- **Total papers**: 6,148 across 9 venues (ICML 1,758 / ICLR 1,318 / NeurIPS 947 / ICCV 598 / CVPR 437 / ACL 399 / EMNLP 351 / AAAI 285 / ICRA 55)
- **Domains**: 12 categories; NLP (2,307), Vision (792), Other (640), Generative (573), Multimodal (522), Theory (452), RL (237), Graph (179), Robotics (137), Systems (126), Tabular (106), Audio (77)
- **Code coverage**: 100% — all entries have verified `code_url` and pinned `code_commit`
- **arXiv IDs**: 3,949/6,148 (64%)
- **Abstracts**: 4,023/6,148 (65%)
- **Ground truth claims**: 0/6,148 — not yet extracted
- **Difficulty scores**: 0/6,148 — not yet computed

### Benchmark Framework
- Evaluation harness (`evaluate.py`, `metrics.py`) complete and tested
- JSON schemas for paper entries and agent results finalized
- pass^4 four-level evaluation model implemented (L1 Build → L2 Run → L3 Reproduce → L4 Cross-Hardware)
- 10+ aggregate metrics with stratification by venue/tier/domain
- Quick Start guide written (`docs/quick-start.md`)

### Data Collection Pipeline
- All 5 phases implemented: collection → code finding → repo validation → claim extraction → finalization
- 9 venue adapters: OpenReview (ICLR/ICML/NeurIPS), CVF (CVPR/ICCV), ACL Portal (ACL/EMNLP), AAAI OJS, Semantic Scholar (ICRA)
- Multi-source code finding: PapersWithCode, Semantic Scholar, HuggingFace, arXiv, PDF extraction

### Known Gaps
- `ground_truth_claims` empty — L3 evaluation falls back to agent self-report
- `difficulty` null — no tier-based stratification available
- 35% papers missing abstracts
- L4 Cross-Hardware deferred

---

## 2026-03-07

- Finalized dataset: 6,148 papers with verified code repos
- Claim extraction script (`extract_claims.py`) implemented but not run at scale
- Evaluation harness implemented with claim matching and metrics

## 2026-03-06

- PANDA architecture documentation added
- PDF code extraction script added
- Human curation TSV workflow established

## 2026-03-05

- Initial commit: benchmark design, paper collection pipeline
- ~35k raw papers collected across 9 venues
- Filtered to papers with open-source code
