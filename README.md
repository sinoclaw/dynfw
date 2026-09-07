# DynFW

**DynFW — Dynamic Fast-Weight Neural Networks** · v0.0.1（Reproducible Research Snapshot）

`Dyn` = Dynamic · `FW` = Fast Weight。动态快权重神经网络——一个**去掉标准自注意力**、用**稀疏激活 + fast-weight rho 记忆**换取长上下文线性复杂度（O(T)）的语言模型架构，作为 Transformer 的结构性替代路线进行公平评估。

> **v0.0.1 是冻结的可复现研究基线（Baseline / Reproducible Research Snapshot），不是"模型版本"。**
> 冻结自 `github.com/sinoclaw/fdn` commit `97f3eaf`（FAIR_BENCH_V2 公平基准）。**任何后续优化不得偷偷修改 v0.0.1 的 benchmark 定义**（见 `docs/fairness.md`）。

## ⚠️ Current limitations（诚实声明，不吹）

- ✗ GPU benchmark（本仓库基线仅在 **CPU** 上跑，无 GPU 列）
- ✗ Large-scale model（仅在 200K–600K 参数级的小型 char-LM 上验证）
- ✗ Real-world downstream tasks（仅字符级语言建模）
- ✗ Long-context quality evaluation（仅测到 T=4096 的 prefill/性能，未做长上下文质量评估）
- ✗ Large-scale scaling（未做 10M+ 规模 scaling）

**这些限制让"能力相当/略优 + 结构降本"的结论只锚定小型 char-LM。"能否替代 Transformer"需真实模型蒸馏 benchmark 才能定论。**

## v0.0.1 基线组成

- `dynfw/models/fused_fw.py` — FusedFW：稀疏激活 + fast-weight rho 记忆 + 非线性读回（+FFN）
- `dynfw/models/transformer.py` — SDPA-Transformer **参考基线**（现代最佳实现，非 naive attention）
- `experiments/v0_0_1_baseline/fair_bench_v2.py` — 严格公平基准（参数匹配 / 5 seed / 固定 val / prefill+decode 分开测）
- `results/v0.0.1/fair_bench_v2.json` — v0.0.1 冻结结果

## v0.0.1 结果摘要（CPU，SDPA-TF 基线，5 真实独立 seed，参数 ≤±5% 匹配）

| 参数 | TF val | FusedFW+FFN val | Δ(FW−TF) | 训练加速 | decode(FW/TF ms/tok) |
|---|---|---|---|---|---|
| 200K | 3.401±.021 | **3.235±.014** | **−0.165** | 1.40x | 0.36 / 0.71 |
| 400K | 3.122±.015 | **3.052±.017** | **−0.070** | 1.20x | 0.32 / 0.74 |
| 600K | 3.039±.010 | **2.980±.009** | **−0.059** | 1.25x | 0.35 / 0.81 |

**prefill 渐近**（~400K，batch 8）：TF 每加倍 2.46-2.91x（超线性 → O(T²)），FusedFW 每加倍 1.75-2.39x（近线性）；FW/TF 从 T=256 的 1.33（慢）→ T=4096 的 0.57（1.75x 快）。

**结论（诚实版）**：同参下 FusedFW+FFN **能力相当/略优** + **训练快 1.2-1.4x** + **decode 每 token 快 ~2.2x** + prefill 长上下文优势（随 T 放大）。**真实架构级优势 = O(T) vs O(T²) 缩放 + decode 便宜，幅度 1.3-2.2x** ——不是早先夸大的"32×"（那是 naive attention 基线伪影，详见 `docs/FAIR_BENCH_V2_REPORT.md`）。

## 结构

```
dynfw/
├── dynfw/
│   ├── models/{fused_fw.py, transformer.py}
│   ├── memory/  capability/  training/
├── experiments/
│   ├── v0_0_1_baseline/  scaling/  long_context/  ablations/
├── benchmarks/{prefill/ decode/ throughput/}
├── results/v0.0.1/
└── docs/{architecture.md, fairness.md, limitations.md, roadmap.md}
```

## 核心原则

> **No benchmark regression, no hidden changes.**

每个版本必须同时保存：旧模型 + 旧 benchmark + 新模型 + 新 benchmark。**不能为了证明"新模型更强"而偷偷更换 benchmark。**（见 `docs/fairness.md`）

## 运行 v0.0.1 基准

```bash
pip install -e .
cd experiments/v0_0_1_baseline && python fair_bench_v2.py   # 需 /data/bdh/input.txt 字符级数据
```

## 文档

- [`docs/architecture.md`](docs/architecture.md) — 架构
- [`docs/fairness.md`](docs/fairness.md) — 公平基准声明（审计固定标准）
- [`docs/limitations.md`](docs/limitations.md) — 已知限制
- [`docs/roadmap.md`](docs/roadmap.md) — 路线图
- [`docs/FAIR_BENCH_V2_REPORT.md`](docs/FAIR_BENCH_V2_REPORT.md) — v0.0.1 公平基准报告
