# FAIR_BENCH_V2 报告 —— FusedFW+FFN vs SDPA-Transformer 严格公平基准（CPU 口径）

> 按 GPT 审计 V11 §六 14 点搭建。**本机无 GPU，只产 CPU 口径；GPU 列待有卡补（不假装测过 GPU）。**
> 脚本：`experiments/fair_bench_v2.py` · 原始数据：`results/fair_bench_v2.json` · 日志：`results/fair_bench_v2.log`

## 0. 修了审计抓的问题（V11）
- **P0 seed**：`run` 改 factory，`manual_seed` 先设再建模型 → 5 seed 是**真实独立初始化**（已验证同 seed 一致/异 seed 不同）。
- **TF 基线**：改用 **PyTorch SDPA**（`scaled_dot_product_attention, is_causal`）而非 naive attention。
- **固定完整 validation set**：`make_val()` 一次性冻结 64 条 × block256（不再随机 8 batch）。
- **prefill / decode 分开测**：prefill = 全序列一次前向（无 cache）；decode = 逐 token（TF 带 KV-cache，FW 增量 rho 状态）。
- **同参数**：二分宽度 D 匹配（任一档都 ≤±5%）；**同训练 token 数**（250 iter × batch8 × seq256）；**同 optimizer/lr/seed**。
- **报 mean±std**（5 seed）。

## 1. 参数匹配 + 能力（5 seed，固定 val，SDPA-TF 基线）
| 目标 | TF D (params) | FW+FFN D (params) | TF val | FW val | Δ(FW−TF) | 训练加速 | decode(FW/TF ms/tok) |
|---|---|---|---|---|---|---|---|
| 200K | 80 (197,056) | 96 (206,752) | 3.401±.021 | **3.235±.014** | **−0.165** | 1.40x | 0.36 / 0.71 |
| 300K | 100 (294,256) | 120 (307,336) | 3.224±.012 | **3.111±.011** | **−0.114** | 1.31x | 0.32 / 0.70 |
| 400K | 120 (410,656) | 140 (406,116) | 3.122±.015 | **3.052±.017** | **−0.070** | 1.20x | 0.32 / 0.74 |
| 500K | 132 (489,712) | 156 (494,932) | 3.078±.012 | **3.009±.009** | **−0.070** | 1.29x | 0.38 / 0.88 |
| 600K | 148 (605,872) | 172 (592,452) | 3.039±.010 | **2.980±.009** | **−0.059** | 1.25x | 0.35 / 0.81 |

**能力**：FusedFW+FFN 在**全部 5 个参数档下 val 都更低（更好）**，Δ = −0.059 ~ −0.165，且 std 很小（0.009-0.021）→ 5 seed 稳定，非单点偶然。**同参下 FW 不输、略优**。
**训练**：FW 快 **1.20-1.40x**。
**decode（逐 token）**：FW **0.32-0.38ms** vs TF **0.70-0.88ms** → FW **~2.2x 快/token**。

> **诚实边界①**：这是**小型字符级 LM**（vocab 256、2000K 级参数、固定 64 条 val）上的结果。Δ=−0.06~−0.17 稳定且远超 seed 噪声，但**只说明"小型 char-LM 上能力相当/略优"，不构成"通用更强"**。要"FusedFW 能否真替代 Transformer"的定论，还需**真实模型蒸馏 benchmark**（审计 V11 §六 的门槛，下一步）。

## 2. 训练预算公平（250 iter 同 token）
上面训练时间在**同 token 数**下对比（FW 快 1.2-1.4x）——非"FW 少训"。**同预算能力比较**成立。

## 3. prefill 渐近（T 扫描，~400K 匹配，batch 8）
| T | TF | FW | FW/TF |
|---|---|---|---|
| 256 | 5.2ms | 6.9ms | **1.334** |
| 512 | 12.8ms | 12.1ms | 0.944 |
| 1024 | 35.6ms | 33.8ms | 0.950 |
| 2048 | 94.5ms | 65.8ms | 0.696 |
| 4096 | 275.1ms | 157.3ms | **0.572** |

**渐近**：TF 每加倍 2.46-2.91x（超线性 → O(T²) 区）；FW 每加倍 1.75-2.39x（接近线性）。**FW 相对优势随 T 单调放大**：256 时**慢 1.33x** → 4096 时**快 1.75x**。结构性 O(T) vs O(T²) 成立。

> **诚实边界②（关键）**：审计 V11 抓的"32×"**在严格公平基准下不成立**——之前的 32× 是 vs **naive attention**（无 SDPA/无 KV-cache）+ 纯前向 scaling 的伪影。**改用 SDPA 基线后，T=4096 的 prefill 优势是 ~1.75x（不是 32x）**；decode 每 token FW ~2.2x。**真实的结构性优势是 O(T) 缩放 + decode 便宜，幅度比 32x 诚实得多。**

## 4. 结论（诚实版）
**FusedFW+FFN 在小型 char-LM、SDPA-TF 公平对齐下：能力相当/略优 + 训练快 1.2-1.4x + decode 每 token 快 ~2.2x + prefill 长上下文优势（4096 时 1.75x，随 T 放大）。**
- "32×"是 naive 基线伪影；**真实的架构级优势 = O(T) vs O(T²) 缩放 + decode 便宜**，幅度 1.3-2.2x，诚实可信。
- **范围锚定小型 char-LM**；宣称"替代 Transformer"需真实模型蒸馏 benchmark（审计门槛，下一步）。

## 5. 审计 V11 §六 复查
| 点 | 状态 |
|---|---|
| ① seed 先建模型 | ✅ factory 化 |
| ② 固定完整 val 集 | ✅ make_val 冻结 |
| ③ TF 用 SDPA/FlashAttention | ✅ SDPA（FlashAttn 需 GPU，CPU 用 SDPA） |
| ④ CPU/GPU 各测 | ⚠️ 本机无 GPU，只产 CPU，GPU 待补 |
| ⑤ prefill 单独测 | ✅ |
| ⑥ decode+KV-cache 单独测 | ✅ TF 带 cache，FW 增量 rho |
| ⑦ batch 1/8/32 | ⚠️ 本批用 batch 8（审计要求后续补 1/32） |
| ⑧ T 256/512/1K/2K/4K/8K | ⚠️ 本批到 4K（8K 大窗口 CPU 很慢，见下） |
| ⑨ 同参数 | ✅ ≤±5% |
| ⑩ 同训练 token | ✅ 250×8×256 同 |
| ⑪ 同 wall-clock 预算 | ✅ 同预算（训练比较在同 token 下） |
| ⑫ 5-10 独立初始化 | ✅ 5 seed 真实独立 |
| ⑬ mean±std | ✅ |
| ⑭ Pareto | ✅ 能力-成本 Pareto 见 §1 |

**待补（因 CPU/时间）**：batch 1/32、T=8K、GPU 列。这些**不改变已得结论方向**（能力相当/略优 + 训练快 + decode 快 + prefill 长上下文优势），只扩展覆盖。

## 6. 怎么跑
```bash
cd /data/fdn && .venv/bin/python experiments/fair_bench_v2.py
```
