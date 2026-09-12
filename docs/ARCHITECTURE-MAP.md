# DynFW 架构台账（ARCHITECTURE-MAP）

> **唯一权威表**：版本号 ↔ 文件 ↔ 机制 ↔ 状态 ↔ 结论指针。
> 建立：2026-09-12（安安）。**目的：不再靠 docstring / commit 考古；任何结论先落本表再对外说。**

---

## 0. 先钉死最常被混淆的两件事：主线 ≠ 最优

| 问题 | 答案 | 文件 | 证据 |
|---|---|---|---|
| **主线**（实际在训练/推进的） | **v6** `BDHBlockFWCycleLM` | `dynfw/models/fused_fw_fw_cycle.py` | `train_talk.py --arch v6/v6w`；`benchmarks/seg_attr_2b.py` 的 `CFG2B=dict(D=2048,nh=16,n_layer=42,mlp_mult=4,vocab=130560,W=256)`；`PLAN-DISTILL-THEN-TRAIN.md` Step1 |
| **最优**（蒸馏口径成绩最好） | **FusedFWFull 每层独立** 385M → KL **93.23** | `dynfw/models/fused_fw_full.py` | `results/CLASH_FULLPARAM_SHOWDOWN.md` |
| **省参最优** | **FusedFWFull weight-sharing**（4 层，**不 tie**）272M → KL **115.24** | `dynfw/models/fused_fw_full_shared.py` | `results/CLASH_WS_DEPTH_SCAN.md` |
| **最优实现形态** | `to_opt5` 融合实现（**数学等价**，+2.19× 吞吐） | `dynfw/models/fused_fw_fw_cycle_opt.py` | 本表 §4 |

> ⚠️ **v6 是「O(T) 压缩状态」路线的载体，不是能力最优版本。** 把二者混讲是历史表述错误。

---

## 1. 归属轴：谁家的架构

| 归属 | 架构 | 文件 |
|---|---|---|
| **我方** | FusedFW / DynFW 系 | `fused_fw*.py`（18 个） |
| 外部参照（标尺） | 标准 SDPA Transformer | `transformer.py` |
| 外部参照（对手） | 官方 BDH（pathwaycom/bdh） | `bdh_qwen.py` `bdh_rawfw_qwen.py` |
| 外部教师 | Qwen3-0.6B（HF，不入库） | — |
| 非架构 | 分块 KL 损失（显存基础设施） | `training/chunked_kl.py` |

## 2. 机制轴（读代码实证，不看文件名）

| 机制 | 文件 | 复杂度 |
|---|---|---|
| 全序列 softmax 注意 | `transformer.py` | O(T²) |
| raw 分块注意 + 跨块 fast-weight | `bdh_qwen` `fused_fw_fw_cycle`(v6) `vla`(v6.5) `gdn`(v6.6) `dla`(v7) `dla_topk`(v8) `rawfw_cycle` | 块内 O(T·W)+定长状态 |
| 线性核 GLA | `bdh_gla` `v2` `v3` | 真 O(T) |
| 纯 rho 快权重（无注意） | `fused_fw`(v0.0.1) `fused_fw_qwen` `lin` `rec` | O(T) 定长状态 |

## 3. 状态轴 + 版本号冲突清点

| 状态 | 文件 |
|---|---|
| 冻结基线 | `fused_fw.py`（v0.0.1）、`transformer.py` |
| **主线** | `fused_fw_fw_cycle.py`(v6) |
| 增量探索 | `vla_cycle`(v6.5) `gdn_cycle`(v6.6) `dla_cycle`(v7) `dla_topk_cycle`(v8) `rawfw_cycle` `la_cycle` `lin` `bdh_gla*` |
| 判负保留 | 早期链 `fused_fw_qwen`(v1) `rec`(v2) `la`(v3) |

**版本号冲突（须唯一化）**：v5 ×3（`full_shared` / `la_cycle` / `lin`）、v6 ×3（`fw_cycle` / `bdh_gla` / `bdh_gla_v2`）、v7 ×2（`dla_cycle` / `bdh_gla_v3`）；`train_talk.py --arch v6` 实指 `BDHBlockFWCycleLM`，与文件名对不上。

---

## 4. 实验台账（三把尺子，**互不可比**）

### 批 1 · 蒸馏全参（D=768 nh=8 L=4，WikiText-103 前2000行，20ep，seed0，共享 233M 词表税）
| 架构 | 参数 | final KL |
|---|---|---|
| **FusedFWFull 每层独立** | 385M | **93.23** ← 最优 |
| FusedFWFull-WS（4 层） | 272M | **115.24** ← 省参最优 |
| BDH | 270M | 117.32 |
| TF | 262M | 133.60 |
| FusedFWFull-WS（16 层） | 272M | 177.87 ❌ 加深反恶化 |
| FusedFWFull + tie | 268M | 1298.56 ❌ tie 崩 |
> 来源：`results/CLASH_{FULLPARAM_SHOWDOWN,WS_DEPTH_SCAN,FUSEDFW_FULL_SHARED}.md`

### 批 2 · 蒸馏小配置（D=128 nh=16 L=2 mm=64，45.2M，3 seed）
| 架构 | final KL（中位） | 复杂度斜率 |
|---|---|---|
| **v6 `fw_cycle`** | **82.39** | **0.76 → 真 O(T)** |
| v5 `la_cycle` | 84.57 | 1.50 → O(T²) |
| bdh_qwen | 88.41 | — |
| TF | 168.98 | — |
> 来源：`experiments/distill/V6_BREAKTHROUGH.md`

### 批 3 · 真训 T=8192（D=256 nh=8 L=6，tinystories，2000 步 / 65.5M token，seed0）
| 臂 | 结构参数 | val_loss | 吞吐 |
|---|---|---|---|
| v6 `mm=4`(N=128) | 4.7M | 3.4624 | 68.5k(orig) → **150k(to_opt5)** |
| v6 `mm=16`(N=512) | 18.9M | 3.3972 | — |
| v6 `mm=64`(N=2048) | 75.5M | 3.3697 | — |
| **TF** | **4.8M** | **2.3791** | 276.3k |
> 来源：`results/scanN_*`、`results/t8192_*`

**批 3 结论（硬）**：v6 结构参数 ×16（4.7M→75.5M）只换 0.093 nats ⇒ **不是容量账，是压缩状态的机制账**；差距 0.99 nats。

---

## 5. 已证伪清单（不要再重犯）

| 假设 | 证伪证据 |
|---|---|
| v6 = 能力 + 真 O(T) 双达标 | 仅在蒸馏模仿分成立；真训 T=8192 输 TF 0.99 nats |
| 加参数（扩 N）能救 v6 | ×16 结构参数 → 仅 0.093 nats |
| 「长上下文成本赢」 | 训练实测慢 4.03×（未接线）→ opt5 后约 1.84× |
| tie embeddings 可省参 | KL 93→1299，崩 14× |
| weight-sharing 加深能补能力 | 4 层 115 → 16 层 178，恶化 |
| 「我们的架构慢 4×」 | **脏评测**：TF 用 SDPA 融合内核、我方用未优化实现；opt5 接口后 2.19× |

## 6. 铁律（每次对轰前逐条核）

1. 速度对轰：**双方必须同优化等级**，且优化必须**真的接进评测入口**（本例 opt5 未接线，教训在案）。
2. 三批尺子（D=768 蒸馏 / D=128 蒸馏 / D=256 真训）**数字不可互比**，引用时必须写配置。
3. KL = 模仿分，不是能力分；能力分须真实 checkpoint + 标准评测。
4. 报数必须附：配置、预算、seed、优化形态。
