# Growth-inheritance 报告 —— warmstart 扩宽能否复用已学权重？（v0.0.1）

> 脚本：`experiments/scaling/growth_inheritance.py`（3 路径判据）+ `experiments/scaling/growth_inheritance_diag.py`（机理诊断，P0 seed 修正）
> 数据：`results/v0.0.1/growth_inheritance.json`、`results/v0.0.1/growth_inheritance_diag.json`｜CPU，3 seed，固定 val 集，同数据/同预算。

## 问题（爸爸问）
DynFW v0.0.1 是固定结构，100K → 更大要重头训练。**"能否从小模型 warmstart 扩宽到更大模型，复用已学权重"**——即"增量长大 + 复用"是否有戏。

## 设计（判据先行，一次一组变量）
- **A fresh-big**：D=96 从零随机，总预算 300 iter
- **B grow**：D=64 先训 150，**零填充扩宽到 D=96**（旧块精确继承、新块清零中性起跑），再训 150（总计 300）
- **C fresh-big-150**：D=96 从零随机只训 150（隔离"大模型短训"效应）

N=4*D，k=16，use_ffn=True，同数据/同预算/同 seed。

## 三路径结果（3 seed，mean±std；`growth_inheritance.py` P0 修复版，权威判定）
| 路径 | val | 说明 |
|---|---|---|
| **A fresh-big（300 it）** | **3.150 ± 0.010** | 最优 |
| **B grow（150小+150大）** | 3.314 ± 0.008 | 比 A 差 **+0.165** |
| C fresh-big（150 it） | 3.521 ± 0.024 | 预算不足 |

> 注：P0 seed 修正与否，A<B<C 顺序稳（故初版 unseeded A=3.145/B=3.314/C=3.521 → 修正 A=3.150/B=3.314/C=3.521）。**结论（B<A、warmstart 不敌重头训满）鲁棒。**

## 判据结论
**同总预算下，warmstart 扩宽（B）劣于目标规模重头训练（A），B − A = +0.165**（`growth_inheritance.py` 权威判定；机理诊断 seed 方案下为 +0.126，二者方向一致）。3 seed std 0.008-0.024 稳定。
但 **B > C**（B 比"同 150 iters 但随机起点的 fresh 大模型"好 0.207）→ **warmstart 非全无用，只是打不过"全程重头训满"**。

## 🔬 为什么 B 会降（机理诊断，逐迭代 val 曲线）
### (一) grow 瞬间无害（推翻此前"LayerNorm 重归一化破坏"猜测）
`grow0 − small_end = −0.087`：零填充扩宽 + LN 重归一化到 96 维后，**val 反而略降（变好）**。→ 增长瞬间的机制扰动**不是** B 下降的原因。

### (二) warmstart 给的是"大头"，但衰减极快
同迭代数 K 下 grown(扩宽) vs fresh(随机) 的 val 对比（3 seed 平均）：
| K(大模型迭代) | grown val | fresh val | grown 领先 | grown 改进/30it | fresh 改进/30it |
|---|---|---|---|---|---|
| 30 | 3.638 | 4.716 | **−1.078** | — | — |
| 60 | 3.522 | 4.182 | −0.660 | −0.116 | −0.534 |
| 90 | 3.431 | 3.877 | −0.445 | −0.091 | −0.305 |
| 120 | 3.356 | 3.680 | −0.324 | −0.076 | −0.197 |
| 150 | 3.293 | 3.535 | **−0.243** | −0.063 | −0.145 |

- **平均改进率（30→150）**：grown **−0.00288/iter**，fresh **−0.00984/iter** → **fresh 快 3.4x**。
- **机理**：warmstart 把 grown 直接放进一个**近收敛盆地**（继承了 D64 已训好的特征），梯度大多是"微调已近最优的解"，**收益递减**；而 fresh 从随机起点出发有**巨大余量**，早期改进凶猛。所以 grown 起初领先 1.078，却以 1/3.4 的速度爬坡，被 fresh 迅速追上。

### (三) 同预算下 B 输给 A = 预算机会成本 > warmstart 收益（决定性分解）
> 注：以下分解来自 `growth_inheritance_diag.json`（独立 seed 方案）；它的 B−C / A−C 与 base 实验（B−C=0.207 / A−C=0.371）数值略异，但**"额外预算价值 > warmstart 收益"的排序在两种 seed 方案下都成立**。
| 量 | 值 | 含义 |
|---|---|---|
| warmstart 优势 @150 (B−C) | **−0.243** | B 比"150 iters 随机起点 fresh"好 0.243 |
| 额外 150 big-iter 价值 (A−C) | **−0.369** | A 比 C 多训 150 iters 带来的改进 |
| **B − A** | **+0.126** | = 0.369 − 0.243 |

**∴ B 输给 A 的根因 = 预算机会成本**：B 把 150 iters 花在小模型（D64）上，最终规模只训到 150 iters；A 全程把 300 iters 花在 D96 上。**warmstart 带回 0.243 的头，但 B 牺牲的 150 个 big-iter 值 0.369 更大** → B 净输 0.126。且 warmstart 的优势因"grown 改进慢 3.4x"而**不随训练复利**，只会被 fresh 反超。

## 结论（诚实、决定性）
1. **"增量长大 + 复用"在 DynFW v0.0.1 上不成立**：先把预算花在小模型、再扩宽继续训，**不如直接把大模型从零训满**（B < A）。**静态架构往上扩 = 重头训练**（TF 亦然，非独有坑）。
2. **warmstart 非全无用**：它确实让"同 150 iters 预算"的 B 比随机起点的 C 好 0.243——只是这份"头"**不随训练复利**（grown 改进率仅 fresh 的 1/3.4 倍），**打不过"把预算直接花在目标规模"**。
3. **与 GSLM No-Go 同源**：增长/复用无独立的、会随训练放大的表征红利；容量与能力须在目标规模直接学。

## 边界
- 只在单层 FusedFW + 一个扩宽步长（D64→D96）测；未测更深/更大步长/插值继承/LR 重置。但**方向结论（warmstart 收益递减、重头训更优）已清晰且稳健**。
- 本实验**不支持**"FusedFW 能越长越大复用"；未来做 growth 的诚实边界。

## P0 修复注记
初版 `growth_inheritance.py` 的 `run()` 存在 P0 seed bug（`FusedFW(...)` 在 `manual_seed` 前创建 → seed 不控初始化）。已在本仓库 `growth_inheritance.py` 修正（模型一律 seed-先建）；`growth_inheritance_diag.py` 为 P0 修正后的权威复跑（3 路径 + 逐迭代曲线 + 机理分解）。
