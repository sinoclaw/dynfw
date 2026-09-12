# DynFW 架构台账（ARCHITECTURE-MAP）

> **唯一权威表**：版本号 ↔ 文件 ↔ 机制 ↔ 状态 ↔ 结论指针。
> 建立 2026-09-12（安安）。**任何结论先落本表再对外说，不靠 docstring / commit 考古。**
> 最近更新：**2026-09-12 17:40 —— 泄漏事件 + 全量重测**。

---

## ⚠️ 当前口径声明（最重要，先读这条）

**2026-09-12 发现两处因果性缺陷，本表此前引用的全部实验读数整体作废：**

| 缺陷 | 内容 | 影响 |
|---|---|---|
| **D1 · 跨 block memory 泄漏** | FWAttention 返回整段序列的 k⊗v 累积并传给下一 block → 位置 t 检索时读到 t 之后的累积 = 偷看未来 | 6 个架构：v6/v6.5 vla/v6.6 gdn/v7 dla/v8 dla_topk/rawfw；**历史 n_layer≥2 读数全部作废** |
| **D2 · softmax 路径泄漏** | `fused_fw_full` 的 `scores.tril(-1)` 后接 `F.softmax` → 上三角被置 0 后被 softmax 变成 exp(0)=1 的非零权重 | 仅 `use_softmax=True` 消融变体；raw 路径不受影响（已回归验证逐位一致） |

- 两处均已修复（D1: `d44e189`；D2: `eca1110`），探针实测 **10/10 + 4/4 全部 `CAUSAL OK`（maxdiff = 0）**。
- **旧结论文档已全部删除**（`results/CLASH_*.md` ×8、`experiments/distill/V6_BREAKTHROUGH.md`、`CLASH_RESULT.md`），旧运行产物目录亦已清理。
- **唯一有效口径 = §4 重测矩阵**（泄漏修复后、同尺子、3 seed）。任何对外引用必须来自 §4。

---

## 0. 主线 ≠ 最优（结构判定仍成立；成绩读数以 §4 为准）

| 问题 | 答案 | 文件 | 状态 |
|---|---|---|---|
| **主线**（实际推进的） | **v6** `BDHBlockFWCycleLM` | `dynfw/models/fused_fw_fw_cycle.py` | ⚠️ 「能力」依据随 D1 作废，待 §4 复测 |
| **蒸馏最优** | FusedFWFull 每层独立 385M | `dynfw/models/fused_fw_full.py` | 因果 ✅（D2 已修）；旧 KL 93.23 作废待重跑 |
| **省参最优** | FusedFWFullShared（4 层不 tie）272M | `dynfw/models/fused_fw_full_shared.py` | 因果 ✅；旧 KL 115.24 作废待重跑 |
| **最优实现形态** | `to_opt5` 融合实现（数学等价，+2.19× 吞吐） | `dynfw/models/fused_fw_fw_cycle_opt.py` | ✅ 有效（属速度账，与泄漏无关） |

> 主线判定依据：`train_talk.py --arch v6/v6w`；`benchmarks/seg_attr_2b.py` 的 `CFG2B`；`PLAN-DISTILL-THEN-TRAIN.md` Step1。
>
> **v6 当前形态（2026-09-12 起）**：块内读侧默认 = **`raw`**（对齐 BDH-CQ 官方配方），实测中位 KL **84.57**；
> 原 softmax 形态（93.05）保留为 `--fw-read softmax` 消融项。详见 §4.2。

---

## 1. 归属轴：谁家的架构

| 归属 | 架构 | 文件 |
|---|---|---|
| **我方** | FusedFW / DynFW 系 | `fused_fw*.py` |
| 外部参照（标尺） | 标准 SDPA Transformer | `transformer.py` |
| 外部参照（对手） | 官方 BDH（pathwaycom/bdh） | `bdh_qwen.py` `bdh_rawfw_qwen.py` |
| 外部教师 | Qwen3-0.6B（HF，不入库） | — |
| 非架构 | 分块 KL 损失（显存基础设施） | `training/chunked_kl.py` |

## 2. 机制轴（读代码实证，不看文件名）

| 机制 | 文件 | 复杂度 |
|---|---|---|
| 全序列 softmax 注意 | `transformer.py`, `fused_fw_full*`（raw/softmax） | O(T²) |
| raw 分块注意 + 跨块 fast-weight | `bdh_qwen` `fused_fw_fw_cycle`(v6) `vla`(v6.5) `gdn`(v6.6) `dla`(v7) `dla_topk`(v8) `rawfw_cycle` | 块内 O(T·W) + 定长状态 |
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

## 4. 实验台账（唯一有效口径 = 泄漏修复后重测矩阵）

**尺子**：批 2 配方 —— D=128 / nh=16 / n_layer=2 / mlp_mult=64 / steps=1，45.2M 学生，Qwen3-0.6B 教师，
corpus_en，block 256 / batch 4 / max_batches 40 / epochs 20，共享教师 logits，**3 seed（0/1/2）取中位**。
**脚本**：`experiments/distill/run_retest_leakfix.sh` | **汇总**：`experiments/distill/collect_retest.py`
**产物**：`results/retest_<arch>_s<seed>/distill_result.json`

### 4.1 重测矩阵结果（2026-09-12，24+15 run 全完成）

**A · W=block(256) 批**（`results/retest_*`，脚本 `run_retest_leakfix.sh`）
| 架构 | 中位 KL | 逐 seed |
|---|---|---|
| `la_cycle`(v5) | 78.56 | 77.55 / 78.56 / 80.73 |
| `dla_cycle`(v7) | 93.05 | 99.00 / 87.66 / 93.05 |
| `bdh` | 93.66 | 93.66 / 88.94 / 97.64 |
| `tf` | 94.58 | 167.43 / 94.33 / 94.58 |
| `rawfw_cycle` | 96.45 | 98.45 / 79.64 / 96.45 |
| `gdn_cycle`(v6.6) | 354.52 | 354.37 / 354.52 / 354.71 |
| `fw_cycle`(v6) | 354.69 | 354.69 / 354.84 / 354.44 |
| `slot_topk`(v8) | 354.69 | 354.69 / 354.84 / 354.44 |

**B · W=64 批（公平口径，`results/w64_*`，脚本 `run_retest_w64.sh`）**
| 排名 | 架构 | 中位 KL | 逐 seed |
|---|---|---|---|
| 1 | `rawfw_cycle` | **84.57** | 81.61 / 84.57 / 87.42 |
| 2 | `gdn_cycle`(v6.6) | **88.99** | 86.87 / 88.99 / 97.49 |
| 3 | `fw_cycle`(v6) | 93.05 | 99.00 / 87.66 / 93.05 |
| 4 | `bdh`（锚） | 93.66 | 88.94 / 93.66 / 97.64 |
| 5 | `slot_topk`(v8) | 97.47 | 96.86 / 97.47 / 112.23 |
| — | `la_cycle`(v5，锚) | **78.56** | 77.55 / 78.56 / 80.73 |

**C · W=block → W=64 对照（本表最重要的方法论发现）**
| 架构 | W=block | W=64 | Δ |
|---|---|---|---|
| `fw_cycle`(v6) | 354.69 | 93.05 | **-261.6** |
| `gdn_cycle` | 354.52 | 88.99 | **-265.5** |
| `slot_topk` | 354.69 | 97.47 | -257.2 |
| `rawfw_cycle` | 96.45 | 84.57 | -11.9 |
| `dla_cycle` | 93.05 | 93.05 | 0（一致性 PASS） |

> ⚠️ **铁律新增（第 7 条）**：凡"块内窗口 + 跨块/跨 chunk 记忆"架构，**报分前必须验证 chunk 数 > 1**
> （chunk 数 = T / W）。W = block 时整层只有 1 个 chunk，**跨块记忆被静默关闭**，架构退化为「仅块内自注意」——
> 此时读数与架构能力无关（v6 的 354 全由此产生，不是崩塌）。

### 4.2 v6 块内读侧单变量对比（2026-09-12，W=64 同口径 3 seed）
| read_mode | 中位 KL | 逐 seed | 判定 |
|---|---|---|---|
| `softmax`（原默认） | 93.05 | 99.00 / 87.66 / 93.05 | ❌ 负担 |
| **`raw`（新默认）** | **84.57** | 84.57 / 81.61 / 87.42 | ✅ 三 seed 全改善 |

**结论**：v6 在 v5→v6 改造时**多改了一个变量**（v5 本是 raw，v6 写成 softmax）→ 白丢 8.5 分。
改回 raw 后 v6 = 84.57（对齐 BDH-CQ 官方配方）。

**等价性判定（决定性）**：`v6(read_mode='raw')` 与 `fused_fw_rawfw_cycle.py` 的
**参数量相同（45,187,072）、参数逐位相同 8/8、logits maxdiff = 0.000000e+00（T=64/256/1024）**
⇒ **两者是同一个模型** ⇒ `rawfw_cycle` 已删除并合并进 v6。
v6 默认 `read_mode='raw'`（`distill_qwen.py --fw-read` 默认同为 raw；softmax 保留为可选消融）。

### 4.3 机制触发核查（2026-09-12 新增，必查项）
| 架构 | 机制 | 触发判据 | 实测 | 判定 |
|---|---|---|---|---|
| `fw_cycle`(v6) | 跨 chunk fast-weight 累积 | chunk 数 > 1 | W=64 → 4 chunks | ✅ 触发 |
| `gdn_cycle`(v6.6) | 线性核定长状态 | chunk 数 > 1 | W=64 → 4 chunks | ✅ 触发 |
| `slot_topk`(v8) | 选择性聚合（topk 读） | 读侧与 sum 有差异 | T=256: maxdiff 0.944 | ✅ 真分化 |
| **`dla_cycle`(v7)** | 信息感知槽合并（DLA Alg.2） | K < chunk 数 时输出应分化 | **T=1024 K=2/4/8 全部 ≤8.3e-07** | ❌ **未分化** |

**v7 判定详情**：`fused_fw_dla_cycle.py` 的合并实现自带"简化/近似"注释
（`new_S = S2[:, :, :-1, :]` 直接去尾，未做 Alg.2 的紧凑化；每 batch 独立选最低密度用 loop 近似）。
→ **矩阵里 `dla_cycle` 的读数实为 `fw_cycle` 的成绩**；DLA 的差异化从未被真正评测。
→ 若要宣称 v7 机制有效，**必须先完整实现 DLA Algorithm 2 再重测**。

### 4.3 口径锚点验证（确认尺子未漂移）
| 架构 | 旧值（泄漏版） | 本次实测 | 判定 |
|---|---|---|---|
| `la_cycle`(v5，不中招) | 84.57 | 78.56 | ✅ 同量级 |
| `bdh`（不中招） | 88.41 | 93.66 | ✅ 同量级 |
| `tf`（不中招） | 168.98 | 94.58（s0=167.43 为离群） | ✅ 同量级 |
| `fw_cycle`(v6，中招) | 82.39 | 354.69（W=block）/ **93.05（W=64）** | ❌ 旧值作废 |

### 4.4 因果性验收（每架构必过）
`benchmarks/probe_causality_all.py`（10 架构）+ `probe_causality_full_variants.py`（批1 冠军/省参冠军）
→ **全部 maxdiff = 0**。脚本：`verify_full_softmax_fix.py`（含 raw 不变量回归）。

---

## 5. 作废清单（不要再引用）

| 原结论 | 作废原因 |
|---|---|
| 「v6 能力更强（KL 82.39 < v5 84.57）」 | **D1 泄漏 artifact**：修复后 v6 = 354.69，实为大幅输 v5 |
| 「v6 能力 + 真 O(T) 双达标」 | 能力那半随 D1 作废；O(T) 那半（斜率 0.76）待复验 |
| 批 2 全部读数（D=128 蒸馏） | 含中招架构，整批作废 |
| 批 1 全部读数（D=768 蒸馏） | 架构本身不中招，但**旧口径已弃用**，须按 §4 尺子重跑后才可引用 |
| 批 3 真训读数（T=8192 / N 扫描） | 同上，旧口径弃用 |
| `CLASH_FUSEDFW_FULL_SOFTMAX.md` 的 softmax 消融 | **D2**：比的是"有泄漏版 vs 无泄漏版"，不公平（原始文档已删，须重跑） |
| 「我们的架构慢 4×」 | 脏评测：对手用 SDPA 融合内核、我方用未优化实现（opt5 接线后 2.19×） |
| **`dla_cycle`(v7) 的矩阵读数 93.05** | **实为 `fw_cycle` 的成绩**：DLA 合并是简化实现，T=1024 / K=2,4,8 实测全部 ≤8.3e-07 **未分化** |
| 「v6 退化到 354 = 架构崩塌」（我的初判） | **误判**：实为 W=block 时 chunk 数=1、跨块记忆被静默关闭；W=64 → 93.05 |
| 「V6_BREAKTHROUGH：v6 能力 + O(T) 双达标」 | 能力那半是 D1 泄漏 artifact（见上） |

## 6. 铁律（每次对轰前逐条核）

0. **分块架构报分前必须验证 chunk 数 = T/W > 1**：W = block 时整层只有 1 个 chunk，跨块记忆被
   **静默关闭**，架构退化成「仅块内自注意」，读数与架构能力无关（v6 的 354 全由此产生）。
   各架构对比必须用**同一个 W**，否则是脏评测。

1. 速度对轰：**双方必须同优化等级**，且优化必须**真的接进评测入口**。
2. 不同尺子（配置/口径）的数字**不可互比**，引用时必须写配置。
3. KL = 模仿分，不是能力分；能力分须真实 checkpoint + 标准评测。
4. 报数必须附：配置、预算、seed、优化形态、**因果性验收状态**。
5. **缺陷无存量豁免**：底层版本带缺陷，其历史读数一律作废重测。
6. 修 bug 必须做**不变量回归**（如 D2 修复后 raw 路径 logits 逐位不变），证明未波及有效读数。

---

## 7. 已删除版本（2026-09-12，v7 / v8）

| 版本 | 原文件 | 删除理由（实测证据） |
|---|---|---|
| **v7 DLA** | `fused_fw_dla_cycle.py` | 「信息感知合并」是 **null operation**：读侧 `Σ_i S_i`（无差别求和）+ 合并 `S_i+S_{i+1}`（也是求和）⇒ **ΣS 恒定 ⇒ 输出恒等**。<br>实测（T=1024/W=64，16 chunks，合并必被触发）：合并策略 `argmin`(原版) vs `argmax`(相反) vs `rand`(随机) → maxdiff **0.000e+00**；K=2（强制合并）vs K=64（不合并）→ maxdiff **0.000e+00**。<br>⇒ v7 与 v6 **数学等价**，不是精度机制。**DLA 的合并真实作用 = 容量控制（显存有界），非保信息** —— 我方文件头「尽量保信息」的动机系误读。 |
| **`rawfw_cycle`** | `fused_fw_rawfw_cycle.py` | 与 `v6(read_mode='raw')` **逐位等价**（参数量相同 45,187,072、参数 8/8 张量逐位相同、logits maxdiff = 0.000000e+00 @T=64/256/1024）→ 纯重复，**合并进 v6** |
| **v8 slot_topk** | `fused_fw_dla_topk_cycle.py` | 唯一真正改读侧的版本（**真分化**：T=256 时与 v6 maxdiff 0.944），但同口径 KL **97.47 vs v6 的 93.05 = 差 4.4 分** → 4-chunk 场景下 Top-K 选择性读**反而丢信息**（K=8/topk=2）。 |

**删除范围**：2 个模型本体 + 蒸馏入口(`distill_qwen.py`)分支 + 因果探针条目 +
10 个专用诊断/探针脚本 + 3 个证伪脚本（方法与数字已完整记录于本表，需复核时按上表方法重跑即可）。

**结论（对研究路线）**：
- 「压缩状态丢信息」**不能靠"整理/合并状态"解决** —— 只要读侧是无差别聚合，槽结构就被抹平。
- 必须先改**读侧**；而唯一试过的读侧改动（v8 Top-K）实测更差。
- 下一步方向：**混合架构**（少量精确层 + 多数压缩层，Jamba/Griffin 路线）或读侧的其它设计。
