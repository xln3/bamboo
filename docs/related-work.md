# Related Work

## 1. Paper Reproduction Benchmarks

### 1.1 PaperBench (OpenAI, 2025)
- **Paper**: [arXiv:2504.01848](https://arxiv.org/abs/2504.01848), ICML 2025
- **Scale**: 20 ICML 2024 Spotlight/Oral papers, 8,316 gradable sub-tasks
- **Task**: From-scratch code implementation to replicate paper results
- **Evaluation**: Hierarchical rubric tree co-developed with original authors; leaf nodes are binary pass/fail; LLM judge (SimpleJudge, o3-mini, F1=0.83)
- **Results**: Claude 3.5 Sonnet 21%, ML PhD 41.4% (48h, best of 3)
- **Variant**: PaperBench Code-Dev (lighter, no GPU required), o1 scores 43.4%
- **Limitations**: Only 20 papers; rubric construction takes weeks per paper; ICML-only; does not evaluate environment setup

### 1.2 CORE-Bench (Princeton, 2024)
- **Paper**: [arXiv:2409.11363](https://arxiv.org/abs/2409.11363), TMLR 2025
- **Scale**: 270 tasks from 90 papers (CS, social science, medicine)
- **Task**: Given code and data, agent installs dependencies, runs code, answers questions about outputs
- **Evaluation**: Three difficulty levels; accuracy on output questions
- **Data source**: CodeOcean.com verified-reproducible repositories
- **Results**: Best agent 21% on hardest level
- **Limitations**: Pre-verified environments; does not require writing code; not ML-specific

### 1.3 Paper2CodeBench (Seo et al., 2025)
- **Paper**: [arXiv:2504.17192](https://arxiv.org/abs/2504.17192)
- **Scale**: 90 papers (ICLR/ICML/NeurIPS 2024, 30 each, token < 70k)
- **Task**: Generate code repository from paper (no execution)
- **Evaluation**: Human judges rate code quality; 88% prefer PaperCoder output
- **Limitations**: Code generation only, no execution or result verification

### 1.4 ReproduceBench (Zhao et al., 2025)
- **Paper**: [arXiv:2505.20662](https://arxiv.org/abs/2505.20662)
- **Scale**: 13 cross-domain papers (knowledge distillation to PDE solving)
- **Task**: Collaborative research+code agents generate executable code from paper descriptions
- **Evaluation**: Human-curated reference implementations; multi-level metrics (code execution + final performance)
- **Limitations**: Small scale (13 papers); from-scratch code generation

### 1.5 REPRO-Bench (2025)
- **Paper**: [arXiv:2507.18901](https://arxiv.org/abs/2507.18901)
- **Scale**: 112 tasks, social science papers
- **Task**: Evaluate reproducibility **scoring** (not just reproduction) — does not assume all papers are reproducible
- **Evaluation**: Agent assesses consistency between paper claims and reproduction package
- **Results**: Best agent below random baseline (25%) on accuracy
- **Significance**: Closest to real reproducibility auditing; evaluates the evaluator

### 1.6 SciReplicate-Bench (2025)
- **Paper**: [GitHub](https://github.com/xyzCS/SciReplicate-Bench)
- **Scale**: 100 tasks from 36 NLP papers (2024)
- **Task**: Generate code from paper algorithm descriptions
- **Evaluation**: Execution accuracy + "reasoning graph accuracy" (novel metric for algorithm comprehension)
- **Agent**: Sci-Reproducer (Paper Agent + Code Agent dual framework)
- **Results**: Best LLM 39% execution accuracy
- **Limitations**: NLP-only; algorithm-level (not full paper reproduction)

## 2. ML Engineering & Research Benchmarks

### 2.1 MLE-bench (OpenAI, 2024)
- **Paper**: [arXiv:2410.07095](https://arxiv.org/abs/2410.07095)
- **Scale**: 75 Kaggle ML competitions
- **Evaluation**: Kaggle medal thresholds (bronze/silver/gold)
- **Results**: o1-preview with AIDE achieves 16.9% bronze rate
- **Difference from BAMBOO**: Tests ML engineering skill, not paper comprehension or reproduction

### 2.2 RE-Bench (METR, 2024)
- **Paper**: [arXiv:2411.15114](https://arxiv.org/abs/2411.15114), ICML 2025
- **Scale**: 7 open-ended ML research engineering tasks + 71 human expert 8h attempts
- **Results**: AI 4x human at 2h, parity at 8h, human 2x AI at 32h
- **Difference from BAMBOO**: Open-ended research tasks, not paper reproduction

### 2.3 ResearchCodeBench (2025)
- **Paper**: [arXiv:2506.02314](https://arxiv.org/abs/2506.02314), NeurIPS 2025
- **Scale**: 212 coding challenges from 20 recent ML papers
- **Task**: Fill in missing core method code given paper + context code
- **Results**: Gemini-2.5-Pro 37.3%, O3 32.3%
- **Difference from BAMBOO**: Function-level code infilling, not end-to-end reproduction

### 2.4 LMR-Bench (2025)
- **Paper**: [arXiv:2506.17335](https://arxiv.org/abs/2506.17335), EMNLP 2025
- **Scale**: 28 tasks from 23 NLP papers
- **Task**: Reproduce masked functions in existing codebase
- **Difference from BAMBOO**: Function-level masking, not full pipeline reproduction

### 2.5 MLRC-Bench (2025)
- **Paper**: [OpenReview](https://openreview.net/forum?id=t8Okk2PRWU)
- **Scale**: 7 ML research competition tasks
- **Task**: Propose and implement novel methods for open research problems
- **Results**: Best agent closes 9.3% of gap to top human
- **Difference from BAMBOO**: Novel method creation, not reproduction

### 2.6 ML-Bench (2023)
- **Paper**: [arXiv:2311.09835](https://arxiv.org/abs/2311.09835)
- **Scale**: 9,641 samples, 18 GitHub repositories
- **Task**: Repository-level ML code generation with argument/documentation handling
- **Difference from BAMBOO**: Code generation benchmark, not reproduction

## 3. Paper-to-Code Agent Frameworks (Non-benchmark)

### 3.1 PaperCoder (Seo et al., 2025)
Multi-agent framework (planning -> analysis -> generation) for ML paper-to-code. 88% rated best vs baselines, 92% judged helpful.

### 3.2 AutoReproduce (Zhao et al., 2025)
Collaborative research+code agent. Research agent extracts info via 3-stage summarization (Mineru PDF->Markdown). Code agent implements and debugs.

### 3.3 RePro (2025, arXiv:2508.16671)
"Reflective Paper-to-Code Reproduction" — fine-grained verification signals mimicking expert checklists, guiding verify-reflect-correct loops.

### 3.4 Paper2Agent (2025, arXiv:2509.06917)
Transforms papers into MCP server-based AI agents. Verified on AlphaGenome, ScanPy, TISSUE.

### 3.5 Prompt-Free Collaborative Agents (2025)
Verification agent + correction agent using only system prompts (no per-step prompt engineering). +15% on PaperBench Code-Dev, +13% on Paper2CodeBench.

### 3.6 DeepCode (HKU, 2025)
Open-source multi-agent platform for paper-to-production-code. CLI + Streamlit UI.

## 4. Platforms & Community

### 4.1 Papers with Code (Meta -> HuggingFace)
Shut down by Meta in July 2025, redirected to HuggingFace. Was the largest paper-code mapping platform.

### 4.2 ML Reproducibility Challenge (MLRC)
Annual community event (Princeton AI Lab). Peer-reviewed reproduction reports published on OpenReview/ReScience/TMLR.

### 4.3 ReproAudit (Commercial, 2025)
Commercial reproducibility auditing platform. Deterministic analysis + LLM-powered code-paper cross-referencing.

## 5. Positioning of BAMBOO

| Dimension | PaperBench | CORE-Bench | Paper2Code | ReproduceBench | **BAMBOO** |
|-----------|-----------|------------|------------|----------------|------------|
| **Goal** | Replicate from scratch | Computational repro | Code generation | From-scratch repro | **End-to-end repro with original code** |
| **Uses original code** | No | Yes | No | No | **Yes** |
| **Environment setup** | Not evaluated | Pre-configured | N/A | N/A | **Core challenge** |
| **Runs experiments** | Yes | Yes | No | Yes | **Yes** |
| **Result verification** | Rubric tree | Output matching | Human judge | Multi-level metrics | **pass^4 + tolerance** |
| **Failure diagnosis** | None | None | None | None | **Five-barrier model** |
| **Scale** | 20 | 90 | 90 | 13 | **1,000** |
| **Venues** | ICML | Multi-discipline | ICLR/ICML/NeurIPS | Cross-domain | **10 top AI venues** |
| **Paper year** | 2024 | Mixed | 2024 | Mixed | **2025** |
| **Rubric cost** | Weeks/paper + author collaboration | Low | Low | Manual | **Automated extraction** |

### Key Differentiators

1. **Real-world workflow**: BAMBOO tests the scenario researchers actually face — cloning existing code, dealing with dependency hell, running experiments, checking results. This is fundamentally different from writing code from scratch (PaperBench) or filling in masked functions (LMR-Bench).

2. **Environment setup as evaluation target**: No existing benchmark explicitly evaluates the agent's ability to resolve dependencies, handle framework version conflicts, or adapt to hardware constraints. BAMBOO's L1 (build) evaluation fills this gap.

3. **Structured failure attribution**: The five-barrier model provides actionable diagnosis, enabling systematic analysis of *why* reproduction fails — not just *whether* it fails.

4. **Scalable evaluation**: By using paper-reported numbers as ground truth and automated metric extraction (vs PaperBench's weeks-per-paper rubric construction), BAMBOO can scale to 1,000 papers.

5. **Contamination resistance**: 2025 papers from 10 venues ensure minimal overlap with LLM training data.
