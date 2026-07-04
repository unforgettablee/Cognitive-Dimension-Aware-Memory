# Memory Mechanism for Code Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Windows](https://img.shields.io/badge/platform-windows%20%7C%20linux-lightgrey.svg)]()

**基于认知维度记忆的代码代理增强框架。** 从已解决的任务中提取结构化经验记忆，通过双通道嵌入检索 + LLM 认知重排 + Synergy 协同选择，将最相关的经验注入代码代理，在 SWE-bench Verified 上实现 **+20 个百分点**的通过率提升（56% → 76%）。

---

## 核心亮点

- **4×4 认知维度记忆矩阵**：4 种认知维度（算法/工具/语法/逻辑）× 多层次抽象程度，精准捕捉修复经验
- **双通道嵌入检索**：任务级 Embedding（语义匹配）+ 认知级 Embedding（维度感知重排），alpha 权重调节
- **LLM 认知重排 & Synergy 检测**：检索后由 LLM 对候选记忆重排序，并通过协同得分选择互补的多维度记忆组合
- **三段式混合文本表示**：问题锚点 + 修复方案 + 底层结果，解决语义空间不匹配和记忆过度抽象问题
- **增量记忆池（Sequential Pipeline）**：逐任务串行执行，每轮积累的记忆即时用于后续任务，模拟真实持续学习场景
- **动态阈值评分**：基于记忆库分数分布的动态阈值，自适应过滤低质量记忆

---

## 实验结果

在 SWE-bench Verified Django 子集（100 个任务）上，使用 DeepSeek-V4-Flash 模型：

| 指标 | Baseline（无记忆） | MTL（记忆增强） | 变化 |
|------|---------------------|-------------------|------|
| 通过率 | 56.0% | 76.0% | **+20.0 pp** |
| 通过任务数 | 56 / 100 | 76 / 100 | +20 |
| 改进任务（失败→通过） | — | 22 | — |
| 回归任务（通过→失败） | — | 2 | — |

> 实验设计：前 100 个任务提取记忆建立经验库，后 100 个任务加载经验库进行评估。

---

## 环境要求

- Python 3.12+
- Docker（harbor 沙箱执行环境）
- DeepSeek API Key
- Harbor >= 0.1.18
- 支持 Windows / Linux

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/unforgettablee/memory-mechanism-for-code-agent.git
cd memory-mechanism-for-code-agent

# 创建虚拟环境并安装
uv venv --python 3.12
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

uv pip install -e ".[harbor]"

# 设置 API 密钥
# Linux/macOS: export DEEPSEEK_API_KEY="sk-xxx"
# Windows: set DEEPSEEK_API_KEY=sk-xxx

# 下载任务
python -m harbor.cli.main tasks download --benchmark swebench-verified --output harbor-tasks/swebench-verified

# 验证安装
mtl --help
```

## 工作流程

```
[1] 运行任务 → 生成轨迹（trajectory + workflow）
[2] 提取记忆 → 4×4 认知维度矩阵 + 具体锚点（函数级定位）
[3] 建立索引 → 三段式混合 Embedding（锚点 + 方案 + 结果）+ 双通道存储
[4] 检索记忆 → 语义检索 → 动态阈值过滤 → LLM 认知重排 → Synergy 协同选择
[5] 注入代理 → Top-K 多维度记忆注入 prompt
```

### 记忆认知维度

| 维度 | 说明 | 抽象层次 |
|------|------|----------|
| 算法策略 (Algorithmic) | 解决方案的整体算法或策略设计 | 4 级抽象 |
| 工具使用 (Tool/API) | 库调用、API 用法、工具链选择 | 4 级抽象 |
| 语法结构 (Syntactic) | 代码语法、AST 结构、模式匹配 | 4 级抽象 |
| 逻辑推理 (Logical) | 错误定位、根因分析、逻辑推导 | 4 级抽象 |

每条记忆包含具体代码锚点（`file:line`），指向问题所在的精确函数和方法，避免经验过于抽象无法落地。

---

## CLI 参考

### `mtl extract` — 从轨迹中提取记忆

```bash
mtl extract \
  --jobs-dir jobs/baseline \
  --memory-dir memories/swebench-verified \
  --start 0 --limit 100 \
  --only-passed
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-j, --jobs-dir` | *(必填)* | Harbor 任务输出目录 |
| `-m, --memory-dir` | `memories/swebench-verified` | 记忆输出目录 |
| `-s, --start` | `0` | 起始任务索引 |
| `-l, --limit` | `100` | 最大处理数量 |
| `--only-passed` | *(关闭)* | 仅提取通过任务的经验 |

### `mtl retrieve` — 查询记忆库

```bash
mtl retrieve -m memories/swebench-verified -q "修复 Django 模型导入错误" -k 3
echo "修复 KeyError..." | mtl retrieve -m memories/swebench-verified --json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-m, --memory-dir` | `memories/swebench-verified` | 记忆库路径 |
| `-q, --query` | *(stdin)* | 查询文本 |
| `-k, --top-k` | `3` | 返回记忆数 |
| `--json` | *(关闭)* | JSON 格式输出 |

### `mtl run` — 批量运行任务

```bash
mtl run -s 100 -e 150 -n 4 --memory-path memories/swebench-verified
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-p, --tasks-dir` | `harbor-tasks/swebench-verified` | 任务定义目录 |
| `-s, --start` | `0` | 起始索引 |
| `-e, --end` | *(全部)* | 结束索引（不包含） |
| `-a, --agent` | `mini-swe-agent` | 代理名称 |
| `-m, --model` | `deepseek/deepseek-chat` | 模型名称 |
| `--memory-path` | *(无)* | 启用记忆检索 |
| `-n, --concurrent` | `2` | 并行数 |

### `mtl experiment` — 批量实验流水线

```bash
mtl experiment -c harbor/configs/experiments/e1_full.yaml --dry-run
mtl experiment --name my-exp --source-start 0 --source-end 51 --target-start 100 --target-end 200
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-c, --config` | *(无)* | YAML 配置文件 |
| `--name` | `mtl-experiment` | 实验名称 |
| `--source-start` | `0` | 源任务起始索引 |
| `--source-end` | `100` | 源任务结束索引 |
| `--target-start` | `100` | 目标任务起始索引 |
| `--target-end` | `200` | 目标任务结束索引 |
| `--memory-dir` | `memories/swebench-verified` | 记忆目录 |
| `--only-passed` | *(开启)* | 仅使用通过任务记忆 |
| `-n, --concurrent` | `2` | 并行数 |
| `--rerank/--no-rerank` | `--rerank` | LLM 重排序开关 |
| `--synergy/--no-synergy` | `--synergy` | Synergy 选择开关 |

### `mtl sequential` — 顺序实验流水线（增量记忆池）

```bash
# 使用配置文件
mtl sequential -c harbor/configs/experiments/e31_seq_full.yaml

# 使用命令行参数
mtl sequential --start 0 --end 500 --pool-dir memories/sequential-pool --jobs-dir jobs/sequential

# 试运行预览
mtl sequential -c harbor/configs/experiments/e33_seq_no_memory.yaml --dry-run

# 从中断处恢复
mtl sequential -c harbor/configs/experiments/e31_seq_full.yaml --resume-from 150
```

逐任务串行执行：第 1 轮（任务 0，无记忆）→ 提取 → 第 2 轮（任务 1，参考任务 0 的记忆）→ 提取 → ... → 第 N 轮（任务 N，参考前 N-1 个任务的记忆）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-c, --config` | *(无)* | YAML 配置文件 |
| `--name` | `sequential-experiment` | 实验名称 |
| `-p, --tasks-dir` | `harbor-tasks/swebench-verified` | 任务定义目录 |
| `-s, --start` | `0` | 起始索引 |
| `-e, --end` | `500` | 结束索引（不包含） |
| `-a, --agent` | `mini-swe-agent` | 代理名称 |
| `-m, --model` | `deepseek/deepseek-chat` | 代理模型 |
| `--jobs-dir` | `jobs/sequential` | 任务输出目录 |
| `--pool-dir` | `memories/sequential-pool` | 增量记忆池目录 |
| `--only-passed` | *(开启)* | 仅提取通过任务 |
| `--resume-from` | `0` | 从第 N 轮恢复 |
| `--rerank/--no-rerank` | `--rerank` | LLM 重排序开关 |
| `--synergy/--no-synergy` | `--synergy` | Synergy 选择开关 |
| `--alpha-dual` | `0.70` | 任务 vs 认知权重 |
| `--top-k` | `3` | 每次检索记忆数 |
| `--api-key` | *(环境变量)* | LLM API Key |
| `--api-base` | *(环境变量)* | LLM API Base URL |
| `--llm-model` | *(环境变量)* | LLM 模型名称 |

### `mtl info` — 记忆库统计

```bash
mtl info -m memories/swebench-verified
mtl info -m memories/swebench-verified --json
```

---

## 实验配置

预置消融实验配置文件位于 `harbor/configs/experiments/`：

**批量实验**（源任务提取 → 目标任务评估）：

| 配置 | 说明 | 关键差异 |
|------|------|----------|
| `e1_full.yaml` | 完整系统 | 所有功能开启 |
| `e2_no_rerank.yaml` | 无 LLM 重排序 | `use_cognitive_rerank: false` |
| `e3_no_synergy.yaml` | 无 Synergy | `use_llm_synergy: false` |
| `e23_pure_embedding.yaml` | 纯 Embedding Top-K | 无 LLM，单通道 |
| `e24_no_memory.yaml` | 无记忆基线 | 记忆禁用 |

**顺序实验**（增量池，逐任务）：

| 配置 | 说明 | 关键差异 |
|------|------|----------|
| `e31_seq_full.yaml` | 完整系统（顺序） | 所有功能开启，递增池 |
| `e32_seq_embedding.yaml` | 纯 Embedding（顺序） | 无 LLM，单通道 |
| `e33_seq_no_memory.yaml` | **无记忆基线** | 记忆禁用，独立运行 |
| `e34_seq_mtl_original.yaml` | 原始方案 | 三级抽象，无维度重排 |

自定义配置：

```yaml
# my_experiment.yaml
name: "my-experiment"
description: "自定义实验"

tasks_dir: "harbor-tasks/swebench-verified"
source_start: 0
source_end: 51
target_start: 100
target_end: 200

agent: "mini-swe-agent"
model: "deepseek/deepseek-chat"

jobs_dir: "jobs/my-exp"
memory_dir: "memories/swebench-verified"
only_passed: true
concurrent: 4

retrieval_config:
  features:
    use_cognitive_rerank: true
    use_llm_synergy: true
  weights:
    alpha_dual_task: 0.70
  retrieval:
    top_n_candidates: 20
    top_k: 3
    min_memories: 1
  threshold:
    score_threshold_floor: 0.45
    score_threshold_std: 0.5
```

---

## Docker

```bash
# 构建
docker build -t memory-agent .

# 运行命令
docker run -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY memory-agent info -m /app/memories/swebench-verified

# 挂载本地目录
docker run \
  -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
  -v $(pwd)/memories:/app/memories \
  -v $(pwd)/jobs:/app/jobs \
  memory-agent sequential -c harbor/configs/experiments/e31_seq_full.yaml

# Docker Compose
docker compose run mtl experiment -c harbor/configs/experiments/e1_full.yaml
```

---

## Python API

```python
from mtl import CognitiveRetriever, MemoryExtractor
from mtl.pipeline import ExperimentPipeline, ExperimentConfig
from mtl.pipeline.sequential import SequentialPipeline, SequentialConfig

# 检索
retriever = CognitiveRetriever("memories/swebench-verified")
results = retriever.retrieve("修复数据管道中的 KeyError", top_k=3)
for r in results:
    print(r["level"], r["type"], r["combined_score"])

# 提取
extractor = MemoryExtractor(
    jobs_dir="jobs/baseline",
    memory_dir="memories/swebench-verified",
    only_passed=True,
)
extractor.extract_batch(start=0, limit=100)

# 批量实验
config = ExperimentConfig(
    name="my-exp",
    source_start=0, source_end=51,
    target_start=100, target_end=200,
    jobs_dir="jobs/my-exp",
)
pipeline = ExperimentPipeline(config)
results = pipeline.run()
print(f"通过率: {results['pass_rate']:.1f}%")

# 顺序实验（增量记忆池）
config = SequentialConfig(
    name="my-seq-exp",
    start_index=0, end_index=500,
    pool_dir="memories/sequential-pool",
    jobs_dir="jobs/sequential",
)
pipeline = SequentialPipeline(config)
summary = pipeline.run()
print(f"最终通过率: {summary['pass_rate']:.1f}% "
      f"({summary['passed']}/{summary['total_tasks']})")
```
