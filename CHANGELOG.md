# CHANGELOG

## 2026-03-14 (Latest)

### Dataset
- **Total papers**: 6,148 across 9 venues (ICML 1,758 / ICLR 1,318 / NeurIPS 947 / ICCV 598 / CVPR 437 / ACL 399 / EMNLP 351 / AAAI 285 / ICRA 55)
- **Domains**: 12 categories; NLP (2,782), Vision (893), Generative (567), Theory (449), Multimodal (443), RL (220), Other (189), Graph (177), Robotics (155), Systems (110), Tabular (92), Audio (71)
- **Code coverage**: 100% — all entries have verified `code_url` and pinned `code_commit`
- **arXiv IDs**: 3,949/6,148 (64%)
- **Abstracts**: 6,148/6,148 (**100%**) — fetched from arXiv API
- **Difficulty scores**: 6,148/6,148 (**100%**) — text-based heuristic scoring (Tier 2: 6,032 / Tier 3: 116)
- **Ground truth claims**: 0/6,148 — requires LLM API for extraction (pipeline ready)

### New in this release
- **MinerU integration**: `pdf_extractor.py` wraps MinerU 2.7.6 for high-quality PDF→markdown extraction with table preservation; includes pdftotext fallback
- **Abstract completion**: `fetch_arxiv_abstracts.py` batch-fetches abstracts from arXiv API with retry logic; took coverage from 65% → 100%
- **Difficulty scoring**: `compute_difficulty.py` computes 6-dimension difficulty scores (framework complexity, dependency count, dataset requirements, hardware requirements, code quality, reproduction time) and assigns tiers 1-4; supports both GitHub API-enhanced and text-only modes
- **Domain re-classification**: With 100% abstracts available, re-ran domain classification — "Other" reduced from 640 → 189 papers (10.4% → 3.1%)
- **Claim extraction pipeline**: `extract_claims.py` updated to use MinerU for text extraction and supports direct `bamboo_final.json` processing; `extract_claims_heuristic.py` provides regex-based extraction as fallback
- **Schema update**: `paper-entry.schema.json` updated to match actual difficulty output format (0-1 dimension scores), added `abstract` and `pdf_url` fields
- **Finalization update**: `finalize_dataset.py` now carries over `ground_truth_claims` and `difficulty` into final dataset, reports data completeness stats

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
- MinerU-based PDF text extraction for improved table/formula parsing

### Remaining Work
- `ground_truth_claims` — run `extract_claims.py` with `OPENAI_API_KEY` set (or any OpenAI-compatible API via `--api-base`)
- Difficulty scoring can be improved with GitHub API data (`--token GITHUB_TOKEN` flag)
- L4 Cross-Hardware evaluation deferred

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
