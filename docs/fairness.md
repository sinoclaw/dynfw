# Fairness 声明 —— DynFW 公平基准标准（冻结，不可更改）

> 本文件固化外部审计（GPT 审计 V11/V13）、以及团队方法论确认的**公平基准标准**。
> **目的：防止 DynFW 后续版本为了"证明新模型更强"而偷偷更改 benchmark 定义。**
> 任何改动必须升级版本号并显式记录（见 v0.0.1 冻结规则）。

## 一、对比基线（硬性）

1. **Transformer 必须用 SDPA**（`torch.nn.functional.scaled_dot_product_attention, is_causal`）——**禁止 naive attention**（显式 QKᵀ + tril mask + softmax）。改用 SDPA 后早先"32×"会被证伪为 naive 基线伪影。
2. **decode 必须带 KV-cache**（TF 逐 token 只用已缓存 k,v；FW 用增量 rho 状态）。prefill 与 decode **分开测**，二者复杂度不同，不得混报。
3. 位置编码：TF 用**正弦固定编码**（0 参数字典），不得用可学习 maxT×D 参数池（会污染"同参数"对齐）。

## 二、数据与口径

4. **固定完整 validation set**（一次性冻结），禁止每次随机抽 8 batch 当 val。
5. **同数据、同 vocab、同 seq、同 optimizer/lr、同训练 token 数**、同 wall-clock 预算。
6. **参数匹配**：两边参数量匹配（任一侧目标参数差 ≤±5%），用宽度搜索对齐。**禁止"参数少的 FW vs 大 TF"不匹配对比**（那是"参数少"的功劳，非架构）。

## 三、统计与复现

7. **seed 先设再建模型**（factory 化）——严禁"模型先建、seed 后设"，否则"5 seed"实为同初始化 + 5 次训练随机化。
8. **≥5 true 独立初始化**，报 **mean ± std**。
9. **预计算判据**：跑前锁死达标值，跑后如实对照，不补指标、不挑 seed、不删失败结果。
10. 分析式 FLOPs **低估 attention softmax/mask 的 O(T²) 开销**——**报成本以 wall-clock 为准**，FLOPs 只作硬件无关参考。

## 四、结论边界

11. 能力"略优"只说**小型 char-LM 范围**，**不说"显著更强"**；**"替代 Transformer"须真实模型蒸馏 benchmark 后才可提**。
12. GPU 基准（FlashAttention）**必须标注"待补"**，不得假装测过 GPU。
13. **No benchmark regression, no hidden changes**：每个版本同时保存旧模型+旧 benchmark + 新模型+新 benchmark。

## 五、v0.0.1 冻结

`97f3eaf`（fair_bench_v2.py 脚本 + fair_bench_v2.json 结果）为 **v0.0.1 冻结基线**。任何后续（v0.0.2+）若改动上述任一标准，必须：
- 升级版本号；
- 在 `docs/roadmap.md` 记录"改了哪条标准、为什么"；
- 同时保留 v0.0.1 的旧模型与旧 benchmark，供对照。
