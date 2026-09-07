# GPT 审计 V11 —— FusedFW 实验链对账（2026-09-07）

> 来源：ChatGPT 分享「深夜问候聊天」（share t_6a9e39d8…，第三方后端抓取全文）。
> 审计对象：`github.com/sinoclaw/fdn` master 下 FusedFW 实验链（d3c68bb / 10b05c9 / 21d5101 / 2cdc96c 等）。
> 结论：**未发现造假，但存在「结论包装过度」** —— 把若干「有条件成立的真实结果」包装成了「比证据更强的结论」。

---

## 一、总评

| 项目 | 判断 |
| --- | --- |
| FusedFW 是否真的比之前少参数 | ✅ |
| 等参数比较是否真的做了 | ✅ |
| 训练时间是否真实测量 | ✅ |
| TF / FW 是否同一数据与训练预算 | ✅ |
| FusedFW 能力是否真接近 TF | ⚠️ 有证据，但目前只能说「小型 char-LM 上接近」 |
| 5 seed 是否严格成立 | ❌ **有代码级 seed 问题（P0）** |
| 参数 scaling 是否公平 | 🟡 基本公平，非严格论文级 |
| 21d5101 的 O(T) vs O(T²) 趋势 | ✅ 结构趋势成立 |
| 「4096 下 32×」能否当通用 Transformer 加速结论 | ❌ **不能** |
| 是否存在明显「掺水/造数据」 | ❌ 未看到 |
| 是否存在「挑有利实验/过度表述」 | ⚠️ **有** |
| 是否值得继续 | ✅ 值得，但下一步须换严格 benchmark |

---

## 二、P0：10b05c9 的「5 seed」其实有漏洞（本次最意外的发现）

`param_scaling.py` 写作：
```python
for sd in SEEDS:
    vl,... = run(mk(D), seed=sd)
```
但 `mk(D)` **在 `run()` 之前就创建模型**，真正的 `torch.manual_seed(seed)` 在 `run()` 内部才设置。
→ seed 只控制了**训练 batch 采样与后续随机性**，**没有控制模型初始权重**。

严格说：3/5 seed 不能叫「5 个独立随机初始化实验」。**不是作弊**（两模型同样问题；每次重新 mk 确实产生不同初始化；只是不可复现、seed 定义不规范）。

**必须修（P0）**：seed 在创建模型之前设置，或让 `run()` 接收 factory：
```python
torch.manual_seed(seed); np.random.seed(seed); m = factory()
# 或 run(lambda: mk(D), seed)
```

> **本仓库修复**：已在 `diag_gap / fused_shared_rho / param_scaling / confirm_ffn / long_seq / long_seq_ability` 六处改为「先 seed 再建模型」（factory 化），并验证同 seed 初始化一致、异 seed 不同。

---

## 三、21d5101 的 32×：方向真，数字有「水分」

- O(T²) vs 近似 O(T) **结构差异成立**（TF 有 T×T attention matrix，FW 无）。
- 但比较对象是 **naive PyTorch attention**（显式 T×T mask、无 FlashAttention/SDPA、无 KV cache）。
- 现代 Transformer 部署用 KV cache：**decode 每 token 是 O(T) 而非 O(T²)**。
- 所以「T=4096 → 32×」只说明 **full-sequence prefill 的 naive attention scaling 非常吃亏**，**不能直接推广成「实际 LLM generation 32×」**。

> **修正**：32× 只能作为该实现上的 **scaling demonstration**，不是通用 Transformer 加速倍率。

---

## 四、能力公平性：实现反而「偏向 TF」

- FusedFW 的 `rho = einsum(...)` 是 **orderless bag-of-context memory**（代码自注「袋状，无时序」），天然丢 token 顺序。
- Transformer 有 position embedding + full causal attention。
- 从能力角度，这个实现 **更可能在给 TF 更多能力**；FW 没位置编码还能接近 TF，反而更有趣。
- 但也说明：**当前 FusedFW 还不是完整 sequence model**。

---

## 五、其他认可 + 诚实点

- `3559dc0`：主动撤销早期「2.5× cost-down」（参数数量假象）→ **加分**，无系统造假动机。
- `f7707ae`：4 种数据分布，高复杂度下 TF 领先（FW 差距 +0.02 → +0.25）——**「FW 参数更少还能普遍不输 TF」被否定**，边界须保留。
- `90f72ca`（BDH COST DOUBLE-DROP FAIL）、`2cdc96c`（SharedRho FAIL）——大量 negative result，非灌水仓库。
- **validation 不是固定验证集**：`run()` 只随机抽 8 个 validation sequence 算一次 val loss → `3.153 vs 3.257` 只是随机 8×256 batch，统计粗糙。只认「该预算 + 该小型 val probe 下能力相当，FW 没表现劣势」，不认「FW 显著优于 TF」。
- **250 iter 预算公平 ≠ 能力公平**：要 Budget-matched（tokens/steps/wall-clock/FLOPs 分别比）+ Convergence-matched（训到收敛再比）。
- **FLOPs 分析式**未计入 attention softmax/mask 的 O(T²)，600K 时 FW 分析 FLOPs 甚至略高于 TF → 只可用 wall-clock，不能用分析式 FLOPs 证「FW 计算量一定更低」。

---

## 六、最终判 + 行动

| 项 | 判定 |
| --- | --- |
| 实验方向 | 🟢 真实 |
| 核心数据 | 🟢 大部分可信 |
| 公平性 | 🟡 研究原型水平 |
| 严谨度 | 🟠 未达论文级 |
| 21d5101「32×」 | 🔴 不能当通用 Transformer 加速结论 |

**最关键一句**：Hermes 没把假的东西包装成真的，但把一些「有条件成立的真实结果」包装成了「比证据更强的结论」。

**建议**：先别跑新模型，**先修实验框架**。最低限度做 `FAIR_BENCH_V2`：
1. seed 在创建模型之前设置
2. 固定完整 validation set（非随机 8 batch）
3. Transformer 用 PyTorch SDPA / FlashAttention
4. CPU、GPU 各测
5. full-sequence prefill 单独测
6. autoregressive decode + KV-cache 单独测
7. batch = 1 / 8 / 32
8. T = 256 / 512 / 1K / 2K / 4K / 8K
9. 同参数
10. 同训练 token 数
11. 同 wall-clock budget
12. 5~10 个真正独立初始化
13. 报 mean ± std
14. 最后才看 Pareto

**真金白银**：FW 值得继续赌的**不是「32×」这个漂亮数字**，而是「去掉 attention 后，在严格公平条件下，能力到底能保留多少、长上下文到底能省多少」。
