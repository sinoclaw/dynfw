# 面壁智能语料配方 —— 调研与下载计划（2026-09-11）

来源：MiniCPM 论文 arXiv:2404.06395 · Ultra-FineWeb 论文 arXiv:2505.05427 · UltraData L0-L4 框架 arXiv:2602.09003 · 各数据集 HF README（实测体积）

---

## 一、面壁的数据治理体系：UltraData L0 → L4 分层框架

| 层 | 数据集 | 做什么 | 实测体积 | token 量 |
|---|---|---|---|---|
| **L0** | Common Crawl 原始快照 | 原始网页 | — | — |
| **L1** | **Ultra-FineWeb-L1** | 正文抽取 → 语言过滤(fastText) → 启发式过滤 → 敏感字段替换 → MinHash 去重（**按 CC dump 内去重，不跨全集**） | **3.31 TB** | — |
| **L2** | **Ultra-FineWeb** | **Ultra-FineWeb classifier**（fastText 轻量分类器）筛出的高质量子集 | **10.21 TB** | **EN 1T + ZH 120B** |
| **L3** | **Ultra-FineWeb-L3** | **Q&A 对生成** + **多风格改写**（百科/教科书/博客/摘要） | **1.90 TB** | **EN 400B+ + ZH 200B+** |
| L4 | 未发布 | （推测：课程/配比层） | — | — |

**⭐ 关键事实：Ultra-FineWeb-L3 是 MiniCPM5-1B 训练【衰减阶段(decay)】的关键数据**（README 原话）。

### 附带资产
| 数据集 | 体积 | 用途 |
|---|---|---|
| **UltraX-Preview** | **487 GB** | 5 个英文语料各 ~20B token，被 UltraX 精炼（**数据效率更优**） |
| UltraData-Math | 635 GB | 数学 |
| UltraData-Code | 1.22 TB | 代码 |
| UltraData-SFT-2605 | 361 GB | SFT |
| DCAD-2000 | 2.76 TB | 2282 语言多语种（README 称全量 46.72TB / 86.3 亿文档） |
| UltraChat | 21 GB | SFT |
| InfLLM-V2-data-5B | 16 GB | 长上下文 |
| FormalVerse | 4.2 GB | 形式化 |
| UltraSafety | — | 安全 |

---

## 二、MiniCPM 的训练配方（论文原文数字）

### 模型与 token 量（arXiv:2404.06395 Table 2）
| 模型 | non-emb 参数 | 层数 L | d_model | 词表 | **训练 token** | batch |
|---|---|---|---|---|---|---|
| MiniCPM-1.2B | 1,247,442,432 | 52 | 1536 | 73,440 | **1.1T** | 2M→4M |
| MiniCPM-2.4B | 2,442,057,984 | 40 | 2304 | 122,753 | **1.1T** | 4M |

→ **token/参数 = 458（2.4B）/ 917（1.2B）**

### WSD (Warmup-Stable-Decay) 学习率调度 —— 面壁的核心方法学
- **衰减用指数退火**：`f(s−T) = 0.5^((s−S)/T)`，**T = 5000 step = 20B tokens**
- **⭐ 衰减阶段只需总 token 的 10%** 就能达到最优：`WSD(D, 0.1D)`；2.5% 不够（实证）
- **衰减阶段的数据配方是关键**：混入「高质量数据 + SFT 数据」，含 UltraChat、SlimOrca 等
- 论文原话：衰减阶段数据混合「包含更多样化的数据和**专有数据**」

### SFT
- MiniCPM-2.4B：**4B token SFT**；MiniCPM-1.2B：6B token SFT
- MiniCPM5-2B：**400B token 深度思考 SFT**（官方披露的唯一数字）

### 数据缩放实验设定
- D = 10N（0.009B 模型用 0.09B token）
- 12 个模型 0.04B~2B，每档 6 个 decay 模型

---

## 三、Ultra-FineWeb 的筛选与验证方法（arXiv:2505.05427）

### 分类器
- **fastText 轻量分类器**（不是 LLM-based）
- **成本对比**：LLM 分类器处理 15T token 需 **6,000 H100 小时**；fastText 只需 **80 CPU × 1,000 小时**；优化后 **1,200 → 110 H100 GPU 小时**（32 卡 <3.5 小时）

### 验证策略（"efficient verification"）
1. 训 **MiniCPM-1.2B 架构 + MiniCPM3-4B tokenizer**，每配置 **100B token**
2. 再基于该模型做 **两阶段退火 10B token**，**30% 权重给验证数据，70% 给默认混合比例**
3. Lighteval 评测

### 实测效果
| 中文对比 | C-Eval | CMMLU | 平均 |
|---|---|---|---|
| Chinese-FineWeb | 33.95 | 32.41 | 33.18 |
| Chinese-FineWeb-edu-v2 | 34.17 | 34.93 | 34.55 |
| **Ultra-FineWeb-zh** | **34.26** | **36.06** | **35.16** |

- Ultra-FineWeb-en 早期即超 FineWeb 和 FineWeb-edu
- Ultra-FineWeb-zh 在 **40B token 后**中文平均分明显提升

### 预处理细节
去重复空行与多余空格、去变音符号、**英文全转小写**

---

## 四、Ultra-FineWeb-L3 的精炼方法（decay 阶段用）

1. **Q&A Pair Generation**：把陈述性网页文档转成「**原文 + 多个问答对**」结构；**训练时原文拼在 Q&A 对前面**，让模型学显式知识组织
2. **Multi-style Rewriting**：单一来源改写成多种表达风格（百科/教科书/博客/摘要），多视角表达同一知识
3. token 计数基于 **MiniCPM5 tokenizer**
4. **实测：Ultra-FineWeb-en-L3 后期训练平均分最高；zh-L3 的优势随训练推进而扩大**

---

## 五、UltraX 的数据效率证据

- 5 个英文语料各 ~20B token 被 UltraX 精炼
- **在 FineWeb 上，UltraX 用 16B token (45.49) 就超过 Raw 和 ProX-C 用 20B token (45.08 / 45.05)**
- 评测：1B MiniCPM 从零训 20B token，10 个基准（ARC-C/E, CSQA, HellaSwag, MMLU, OBQA, PIQA, SIQA, WinoGrande, SciQ），LightEval 零样本

---

## 六、我们的下载计划（A800 节点 /data，943 GB 可用）

**下载实测速度**：curl 单连接 1.9 MB/s → **aria2c 16 连接 51 MB/s（+27×）**

### 已启动 ✅
| 数据集 | 体积 | token | 说明 |
|---|---|---|---|
| **UltraX-Preview（全量）** | **487 GB** | **~100B** | 5 语料 × 20B，UltraX 精炼版；通用底座主力，密度 4.9 B/token |

细目：UltraX-FineWeb-ProX-Doc 109.3GB / UltraX-Ultra-FineWeb 108.9GB / UltraX-FineWeb 108.8GB / UltraX-RedPajama-V2 91.4GB / UltraX-AICC 68.6GB

### 待定（剩余 ~456 GB）
| 候选 | 体积 | 密度 | 用途 | 建议 |
|---|---|---|---|---|
| Ultra-FineWeb-L3 (部分) | 1.90 TB 全量 | **3.2 B/tok（最密）** | **decay 阶段**（面壁自家就这么用） | ⭐ 强烈建议抽 ~250GB |
| UltraData-Math | 635 GB | ? | 数学能力 | 抽 ~100GB |
| UltraData-Code | 1.22 TB | ? | 代码能力 | 抽 ~100GB |
| Ultra-FineWeb (L2 全量) | 10.2 TB | 9.1 B/tok | 与教师最同源 | 放不下，用 UltraX 替代 |

---

## 七、对我们规划的影响（PLAN-DISTILL-THEN-TRAIN.md）

1. **语料不再是瓶颈** —— 面壁把教师吃的语料全开源了，且国内镜像可下（aria2c 51MB/s）
2. **数据配方有官方蓝本**：L2 做 stable + **L3 做 decay**（WSD 的 0.1D 阶段）
3. **WSD 的具体参数可抄**：指数退火、T=20B token 等价步数、**decay 只需 10% token**
4. **验证方法可抄**：100B token / 1.2B 架构 / 两阶段退火 10B token（30% 验证数据）
5. **密度差异要记账**：L2 9.1 B/tok vs L3 3.2 B/tok vs UltraX 4.9 B/tok —— 同样 1TB 盘，装 L3 能多拿 3 倍 token
