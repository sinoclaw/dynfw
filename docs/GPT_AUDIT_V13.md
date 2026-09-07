# GPT 审计 V13 —— 主线开新仓 DynFW v0.0.1（2026-09-07）

> 来源：ChatGPT 分享「深夜问候聊天」（share t_6a9e466e…，第三方后端抓取全文）。
> 审计对象：FusedFW 主线（FAIR_BENCH_V2 结果，commit 97f3eaf）。
> 核心建议：**开新仓库，改名 DynFW，冻结 97f3eaf 为 v0.0.1 可复现基线，把"探索期"与"正式路线"切开。**

## 一、为什么开新仓库（不改名继续用 fdn）
- **FDN 背着历史包袱**（dynamic node / DCA / BDH 一堆 No-Go/partial 探索），新仓库应从 **FusedFW 这条已经过公平审计的主线**重新开始。
- **项目名不该是 FusedFW**（太"模块名"——以后 FusedFW-v2/新 state mechanism 会绑死）；**不能 BDH-FW/BDH++**（让人误以为"BDH 变体"，而我们想发展自己路线）；**不能 DNN**（太泛）。
- 目的：把 `fdn` 仓库（探索期、含大量 negative result）与**正式路线**（DynFW）切开，干净。

## 二、命名投票
- 🥇 **DynFW — Dynamic Fast-Weight Neural Networks**：`Dyn`=Dynamic / `FW`=Fast Weight；不蹭 BDH；是**路线名非模块名**；天然可扩展 DynFW-Base/LM/Code/MoE/DCA；覆盖已有层次（Fused Fast-Weight computation / Persistent associative state / Sparse activation / Nonlinear readback / Dynamic capability(未来)）。
- 🥈 **FWA — Fast-Weight Architecture**：干净但太泛、辨识度低。
- 🥉 **NFW — Neural Fast Weights**：像基础架构名，易与已有 Fast Weight 文献混。

## 三、版本号 = Baseline / Reproducible Research Snapshot（不是"模型版本"）
`DynFW v0.0.1` 定义成：
- FusedFW + FFN
- Transformer-SDPA baseline
- parameter-matched benchmark
- 5-seed evaluation
- fixed validation set
- prefill benchmark
- KV-cache decode benchmark
- current known limitations

**把 `97f3eaf`（FAIR_BENCH_V2）当冻结基线。** 铁律：**以后任何优化不能偷偷修改 v0.0.1 的 benchmark 定义**——否则几个月后 `v0.0.1 TF3.04/FW2.98` vs `v0.0.2 TF3.21/FW2.80` 没人知道是模型进步还是 benchmark 变了。

## 四、仓库结构建议
```
dynfw/
├── README.md / LICENSE / CITATION.cff
├── dynfw/
│   ├── models/{fused_fw.py, transformer.py}
│   ├── memory/  capability/  training/
├── experiments/
│   ├── v0_0_1_baseline/  scaling/  long_context/  ablations/
├── benchmarks/{prefill/ decode/ throughput/}
├── results/v0.0.1/
└── docs/{architecture.md, fairness.md, limitations.md, roadmap.md}
```

## 五、`fairness.md`（特别重要）
把审计得出的公平标准写死：Transformer 用 SDPA、KV cache、固定 validation、seed-before-init、parameter matching、5 seeds、prefill/decode 分开。**这样以后 Hermes 不能自己偷偷改变实验标准。**

## 六、README 开头不要吹（诚实边界反而涨可信度）
```
Current limitations:
✗ GPU benchmark
✗ large-scale model
✗ real-world downstream tasks
✗ long-context quality evaluation
✗ large-scale scaling
```

## 七、第一条原则
> **No benchmark regression, no hidden changes.**
每个版本同时保存：旧模型+旧 benchmark + 新模型+新 benchmark。**不能"为证明新模型更强，把 benchmark 换了"。**

## 八、可做的事（审计提出）
1. 写 DynFW v0.0.1 README 开头
2. 按方案写仓库目录结构 README 描述
3. 写版本号管理 + benchmark 不可变原则说明
