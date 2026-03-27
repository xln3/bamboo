# BAMBOO 使用指南

BAMBOO (Benchmark for Autonomous ML Build-and-Output Observation) 是一个用于评估 AI Agent **复现机器学习论文**能力的基准。它测试完整的流水线：论文理解 → 环境搭建 → 实验运行 → 结果验证。

---

## 目录

1. [快速开始](#1-快速开始)
2. [模型配置（即插即用）](#2-模型配置即插即用)
3. [运行复现测试](#3-运行复现测试)
4. [评估与裁判](#4-评估与裁判)
5. [12 篇论文小数据集](#5-12-篇论文小数据集)
6. [对比不同模型](#6-对比不同模型)
7. [接入你自己的 Agent](#7-接入你自己的-agent)
8. [数据格式说明](#8-数据格式说明)
9. [常见问题](#9-常见问题)

---

## 1. 快速开始

### 1.1 环境要求

- Python 3.10+
- [Deno](https://deno.land) 2.x（用于 PANDA agent）
- PANDA agent 安装在 `~/panda2026/panda/`

### 1.2 三步跑通

```bash
cd /path/to/bamboo

# 第一步：配置模型（复制模板，填入 API key）
cp configs/models.example.json configs/models.json
# 编辑 configs/models.json，填入你的 API key

# 第二步：运行一篇论文试试
python3 -m scripts.run.runner \
  --agents panda \
  --model glm-5 \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06089 \
  --timeout 1800

# 第三步：查看结果
cat data/results/panda-glm-5/bamboo-06089.json
```

---

## 2. 模型配置（即插即用）

所有模型配置集中在 `configs/models.json`。**切换模型只需改 `--model` 参数，无需改代码。**

### 2.1 配置文件格式

```json
{
  "glm-5": {
    "provider": "openai",
    "model": "glm-5",
    "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    "api_key": "<your-zhipuai-api-key>"
  },
  "claude-sonnet": {
    "provider": "openai",
    "model": "claude-sonnet-4-20250514",
    "base_url": "https://aihubmix.com/v1",
    "api_key": "<your-aihubmix-key>",
    "no_proxy": "aihubmix.com"
  },
  "gpt-4o": {
    "provider": "openai",
    "model": "gpt-4o",
    "base_url": "https://api.openai.com/v1",
    "api_key": "<your-openai-key>"
  }
}
```

### 2.2 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `provider` | 是 | API 协议类型，目前支持 `openai`（OpenAI 兼容接口） |
| `model` | 是 | 模型名称，如 `glm-5`、`claude-sonnet-4-20250514`、`gpt-4o` |
| `base_url` | 是 | API 端点 URL |
| `api_key` | 是 | API 密钥 |
| `no_proxy` | 否 | 需要绕过代理直连的域名（用于代理环境） |

### 2.3 添加新模型

只需在 `configs/models.json` 中添加一个新条目：

```json
{
  "deepseek-v3": {
    "provider": "openai",
    "model": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "<your-deepseek-key>"
  }
}
```

然后直接使用：

```bash
python3 -m scripts.run.runner --agents panda --model deepseek-v3 --papers bamboo-06079
```

### 2.4 查看可用模型

```bash
python3 -m scripts.run.runner --list-models
```

输出示例：

```
Available model profiles (configs/models.json):

  claude-sonnet         model=claude-sonnet-4-20250514        url=https://aihubmix.com/v1
                        api_key=***
  glm-5                 model=glm-5                           url=https://open.bigmodel.cn/api/coding/paas/v4
                        api_key=***
  gpt-4o                model=gpt-4o                          url=https://aihubmix.com/v1
                        api_key=***
```

### 2.5 安全说明

`configs/models.json` 包含 API 密钥，已在 `.gitignore` 中排除。仅 `configs/models.example.json`（无真实密钥的模板）会被提交到 git。

---

## 3. 运行复现测试

### 3.1 基本用法

```bash
python3 -m scripts.run.runner \
  --agents panda \
  --model <模型名> \
  --papers <paper_id ...> \
  [--dataset <数据集路径>] \
  [--timeout <超时秒数>] \
  [--skip-judge] \
  [--dry-run]
```

### 3.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agents` | `panda` | 使用哪个 agent（可选：`panda`, `claude-code`, `opencode`, `codex`） |
| `--model` | **必填** | 模型配置名称（来自 `configs/models.json`） |
| `--papers` | 5 篇 pilot | 指定论文 ID（空格分隔多个） |
| `--sample N` | — | 随机选 N 篇有 claims 的论文 |
| `--dataset` | `data/bamboo_final.json` | 数据集文件路径 |
| `--timeout` | 1800 | 每篇论文的超时时间（秒） |
| `--skip-judge` | false | 跳过独立裁判（仅运行 agent） |
| `--judge-model` | opus | 裁判使用的模型 |
| `--dry-run` | false | 只打印 prompt，不实际运行 |
| `--list-models` | — | 列出可用模型配置后退出 |

### 3.3 运行示例

```bash
# 用 glm-5 复现一篇论文
python3 -m scripts.run.runner \
  --agents panda --model glm-5 \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06089

# 用 claude-sonnet 复现同一篇论文（对比）
python3 -m scripts.run.runner \
  --agents panda --model claude-sonnet \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06089

# 随机选 5 篇有 claims 的论文
python3 -m scripts.run.runner \
  --agents panda --model glm-5 --sample 5

# 预览 prompt（不实际运行）
python3 -m scripts.run.runner \
  --agents panda --model glm-5 \
  --papers bamboo-06079 --dry-run
```

### 3.4 输出结构

每次运行会生成以下文件：

```
data/results/panda-glm-5/           # 结果目录（agent名-模型名）
├── bamboo-06079.json                # 每篇论文的结果
├── bamboo-06080.json
├── logs/                            # 运行日志
│   ├── bamboo-06079/
│   │   ├── stdout.txt               # agent 标准输出
│   │   └── stderr.txt               # agent 标准错误
│   └── bamboo-06080/
│       ├── stdout.txt
│       └── stderr.txt
└── judge/                           # 独立裁判结果
    ├── bamboo-06079.json
    └── bamboo-06080.json
```

### 3.5 运行过程输出

```
Model: glm-5 → glm-5 @ https://open.bigmodel.cn/api/coding/paas/v4
Loaded 3994 papers from data/bamboo_curated.json
Selected 1 paper(s): ['bamboo-06089']
Agents: ['panda-glm-5']
Timeout: 1800s per paper

============================================================
[RUN] Agent=panda-glm-5  Paper=bamboo-06089  Timeout=1800s
  Title: SmoothQuant: Accurate and Efficient Post-Training Quantization for La
  Repo:  https://github.com/mit-han-lab/smoothquant
  Claims: 104
  [OK (agent wrote result)] exit=0 time=892s level=L2
  [JUDGE] Evaluating 104 claims...
  [JUDGE] 23/104 values extracted (126359ms)

============================================================
  BAMBOO Run Summary
  Model: glm-5
  Total time: 1019s
============================================================
```

---

## 4. 评估与裁判

### 4.1 pass^4 四级评估

BAMBOO 使用渐进式四级评估：

```
L1 Build  →  L2 Run  →  L3 Reproduce  →  L4 Cross-Hardware
```

| 级别 | 含义 | 通过标准 |
|------|------|----------|
| **L1** | 环境搭建成功 | 依赖安装完成，import 成功 |
| **L2** | 实验运行完成 | 进程正常退出，有输出 |
| **L3** | 结果复现 | ≥80% claims 在容差范围内 |
| **L4** | 跨硬件一致 | 在 2 种硬件上都通过 L3（未来） |

### 4.2 独立裁判（防作弊）

runner 运行完 agent 后会自动调用独立裁判（Independent Judge）：

- 裁判从 agent 的 stdout/stderr 和输出文件中**提取实际值**
- 裁判**看不到期望值**（防止 agent 作弊）
- 提取结果保存到 `results/<agent_id>/judge/<paper_id>.json`

手动运行裁判：

```bash
# 对单篇论文运行裁判
python3 -m scripts.evaluate.judge \
  --paper bamboo-06089 \
  --agent panda-glm-5 \
  --dataset data/bamboo_curated.json

# 对所有已有结果运行裁判
python3 -m scripts.evaluate.judge \
  --agent panda-glm-5 --all \
  --dataset data/bamboo_curated.json
```

### 4.3 生成评估报告

```bash
python3 -m scripts.evaluate.evaluate \
  --results-dir data/results/panda-glm-5/ \
  --dataset data/bamboo_curated.json \
  --output data/results/panda-glm-5/report.json
```

### 4.4 容差标准

| 指标类型 | 容差 | 示例 |
|----------|------|------|
| 确定性 | ±1% | 参数量、精确匹配分数 |
| 标准 | ±5% | Accuracy、F1、BLEU、mAP |
| 高方差 | ±10% | FID、IS、生成模型指标 |
| 时间类 | ±20% | 训练时间、推理速度 |

---

## 5. 策划数据集（3,994 篇论文）

`data/bamboo_curated.json` 包含 3,994 篇精选论文，每篇都有 MinerU 提取的 Markdown 全文和 ground truth claims。其中 2,658 篇有内联 claims，709 篇有 paper_claims_v2 文件。

以下 12 篇代表性论文覆盖视觉生成、模型量化、时序预测、稀疏推理等方向，适合快速验证：

### 5.1 代表论文列表

| # | paper_id | 论文 | 方向 | Claims | 代码仓库 |
|---|----------|------|------|--------|----------|
| 1 | `bamboo-06079` | VAR: Visual Autoregressive Modeling | 图像生成 | 28 | FoundationVision/VAR |
| 2 | `bamboo-06080` | Infinity: Bitwise AutoRegressive Modeling | 图像生成 | 37 | FoundationVision/Infinity |
| 3 | `bamboo-06081` | Masked Vector Quantization for AR Image Gen | 图像生成 | 28 | CrossmodalGroup/MaskedVectorQuantization |
| 4 | `bamboo-06082` | ARPC: Ultra-Low Bitrate Image Compression | 图像压缩 | 54 | (无公开代码) |
| 5 | `bamboo-06083` | Customizable-ROI Image Compression | 图像压缩 | 64 | hccavgcyv/Customizable-ROI-Based-Deep-Image-Compression |
| 6 | `bamboo-06084` | SCALAR: Scale-wise Controllable VAR | 图像生成 | 147 | AMAP-ML/SCALAR |
| 7 | `bamboo-06085` | TEAL: Training-Free Activation Sparsity | LLM 稀疏 | 250 | FasterDecoding/TEAL |
| 8 | `bamboo-06086` | CATS: Self-Attentions for Time Series | 时序预测 | 92 | dongbeank/CATS |
| 9 | `bamboo-06087` | ARCQuant: NVFP4 Quantization for LLMs | LLM 量化 | 79 | actypedef/ARCQuant |
| 10 | `bamboo-06088` | FlatQuant: LLM Quantization | LLM 量化 | 163 | ruikangliu/FlatQuant |
| 11 | `bamboo-06089` | SmoothQuant: Post-Training Quantization | LLM 量化 | 104 | mit-han-lab/smoothquant |
| 12 | `bamboo-03009` | R-Sparse: Rank-Aware Activation Sparsity | LLM 稀疏 | 100 | VITA-Group/R-Sparse |

### 5.2 运行示例（12 篇代表论文）

```bash
python3 -m scripts.run.runner \
  --agents panda \
  --model glm-5 \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06079 bamboo-06080 bamboo-06081 bamboo-06082 \
           bamboo-06083 bamboo-06084 bamboo-06085 bamboo-06086 \
           bamboo-06087 bamboo-06088 bamboo-06089 bamboo-03009 \
  --timeout 1800
```

### 5.3 预计耗时

- 每篇论文超时 30 分钟（`--timeout 1800`）
- 12 篇论文顺序运行，最长 6 小时
- 实际耗时取决于论文复杂度和模型能力
- 裁判评估每篇约 1-3 分钟

---

## 6. 对比不同模型

BAMBOO 的核心用途之一是**对比不同 Agent 或模型的复现能力**。

### 6.1 运行对比实验

```bash
# 第一组：GLM-5
python3 -m scripts.run.runner \
  --agents panda --model glm-5 \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06085 bamboo-06088 bamboo-06089

# 第二组：Claude Sonnet
python3 -m scripts.run.runner \
  --agents panda --model claude-sonnet \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06085 bamboo-06088 bamboo-06089

# 第三组：GPT-4o
python3 -m scripts.run.runner \
  --agents panda --model gpt-4o \
  --dataset data/bamboo_curated.json \
  --papers bamboo-06085 bamboo-06088 bamboo-06089
```

### 6.2 结果自动隔离

不同模型的结果存储在不同目录中，互不干扰：

```
data/results/
├── panda-glm-5/          # GLM-5 的结果
│   ├── bamboo-06085.json
│   ├── bamboo-06088.json
│   └── bamboo-06089.json
├── panda-claude-sonnet/   # Claude Sonnet 的结果
│   ├── bamboo-06085.json
│   └── ...
└── panda-gpt-4o/          # GPT-4o 的结果
    ├── bamboo-06085.json
    └── ...
```

### 6.3 生成对比报告

```bash
# 分别生成评估报告
python3 -m scripts.evaluate.evaluate \
  --results-dir data/results/panda-glm-5/ \
  --dataset data/bamboo_curated.json \
  --output data/results/panda-glm-5/report.json

python3 -m scripts.evaluate.evaluate \
  --results-dir data/results/panda-claude-sonnet/ \
  --dataset data/bamboo_curated.json \
  --output data/results/panda-claude-sonnet/report.json
```

对比报告中的关键指标：

| 指标 | 含义 |
|------|------|
| Build Rate | 环境搭建成功率 |
| Run Rate | 实验运行成功率 |
| Reproduce Rate | 结果复现率（≥80% claims 通过）|
| Mean Reproduction Score | 平均 claim 通过率 |

---

## 7. 接入你自己的 Agent

### 7.1 实现 AgentAdapter

在 `scripts/run/agents/` 下创建新文件，继承 `AgentAdapter`：

```python
# scripts/run/agents/my_agent.py
from __future__ import annotations
from pathlib import Path
from .base import AgentAdapter

class MyAgentAdapter(AgentAdapter):
    @property
    def _base_agent_id(self) -> str:
        return "my-agent"

    def build_command(
        self, prompt: str, workdir: Path, timeout_s: int
    ) -> list[str]:
        mc = self._model_config
        return [
            "my-agent-binary",
            "--prompt", prompt,
            "--model", mc.get("model", "default-model"),
        ]

    def env_overrides(self) -> dict[str, str]:
        mc = self._model_config
        return {
            "MY_AGENT_API_KEY": mc.get("api_key", ""),
            "MY_AGENT_BASE_URL": mc.get("base_url", ""),
        }
```

### 7.2 注册到 runner

编辑 `scripts/run/runner.py`，添加到 `AGENT_REGISTRY`：

```python
from .agents.my_agent import MyAgentAdapter

AGENT_REGISTRY = {
    "panda": PandaAdapter,
    "claude-code": ClaudeCodeAdapter,
    "my-agent": MyAgentAdapter,   # ← 新增
}
```

### 7.3 运行你的 Agent

```bash
python3 -m scripts.run.runner \
  --agents my-agent --model glm-5 \
  --papers bamboo-06089
```

### 7.4 Agent 需要输出什么

Agent 必须在指定路径写一个 JSON 文件（路径在 prompt 中给出），包含：

```json
{
  "paper_id": "bamboo-06089",
  "agent_id": "my-agent-glm-5",
  "timestamp": "2026-03-27T12:00:00Z",
  "pass4": {
    "l1_build": {"status": "pass|fail|timeout", "detail": "..."},
    "l2_run":   {"status": "pass|fail|timeout", "detail": "..."},
    "l3_reproduce": {"status": "skip"},
    "l4_cross": {"status": "skip"},
    "overall_level": 0
  },
  "barriers": [],
  "resource_usage": {"total_time_ms": 120000}
}
```

- `overall_level`: 0=L1 失败，1=L1 通过但 L2 失败，2=实验运行完成
- L3 由独立裁判评估，agent 设为 `"skip"`
- 如果 agent 没有输出 JSON，runner 会自动生成一个 L0 fallback 结果

---

## 8. 数据格式说明

### 8.1 数据集（bamboo_curated.json）

每篇论文的条目：

```json
{
  "paper_id": "bamboo-06089",
  "title": "SmoothQuant: ...",
  "venue": "NeurIPS",
  "year": 2022,
  "arxiv_id": "2211.10438",
  "code_url": "https://github.com/mit-han-lab/smoothquant",
  "paper_url": "https://arxiv.org/pdf/2211.10438",
  "ground_truth_claims": [
    {
      "claim_id": "c1",
      "description": "Table 3: SmoothQuant-O1 accuracy on LAMBADA (OPT-175B)",
      "metric_name": "accuracy",
      "metric_value": 74.7,
      "source_location": "Table 3",
      "tolerance": 0.05,
      "higher_is_better": true,
      "dataset": "LAMBADA",
      "category": "main"
    }
  ]
}
```

### 8.2 Claims 文件（paper_claims_v2/）

每篇论文的 claims 也单独存储在 `data/paper_claims_v2/<paper_id>.json`，runner 会自动合并。

### 8.3 裁判结果

```json
{
  "paper_id": "bamboo-06089",
  "agent_id": "panda-glm-5",
  "claim_results": [
    {
      "claim_id": "c1",
      "actual_value": 74.5,
      "evidence_text": "From stdout: LAMBADA acc: 74.5%",
      "confidence": "high"
    }
  ]
}
```

---

## 9. 常见问题

### Q: 如何用非 PANDA 的 agent（如 Claude Code）？

```bash
python3 -m scripts.run.runner --agents claude-code --model claude-sonnet --papers bamboo-06089
```

Claude Code adapter 不需要 model_config 中的所有字段，但仍需提供 `--model` 参数。

### Q: 论文没有 code_url 怎么办？

对于 `code_url` 为 null 的论文（如 bamboo-06082），prompt 会提示 agent 自行从论文或 arxiv 页面查找代码仓库。

### Q: 如何只运行 agent 不跑裁判？

```bash
python3 -m scripts.run.runner --agents panda --model glm-5 --papers bamboo-06089 --skip-judge
```

### Q: 超时了怎么办？

调大 `--timeout`（单位秒），默认 1800s（30 分钟）：

```bash
python3 -m scripts.run.runner --agents panda --model glm-5 --papers bamboo-06079 --timeout 3600
```

### Q: 在哪看 agent 的完整输出？

```bash
cat data/results/panda-glm-5/logs/bamboo-06089/stdout.txt
cat data/results/panda-glm-5/logs/bamboo-06089/stderr.txt
```

### Q: 如何添加新的论文到数据集？

编辑 `data/bamboo_curated.json`，添加论文条目（至少需要 `paper_id`、`title`、`code_url`）。然后用 `scripts/evaluate/fill_claims.py` 提取 claims：

```bash
python3 -m scripts.evaluate.fill_claims --workers 1
```

### Q: configs/models.json 中的 provider 有哪些选项？

目前支持 `openai`（OpenAI 兼容接口），覆盖智谱 GLM、OpenAI GPT、Anthropic Claude（通过 aihubmix 转发）等。
