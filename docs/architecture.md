# Architecture —— DynFW v0.0.1

## 一、设计动机

Transformer 的**自注意力随序列长度 O(T²)**，且 KV-cache 随 T 线性增长。DynFW 的假设是：**去掉标准自注意力，用稀疏激活 + fast-weight rho 记忆**，可在**长上下文**换取 **O(T) 线性复杂度 + decode 便宜**，同时保持能力不输。

## 二、核心计算

```
输入 → 嵌入 (D)
   → enc:  D → N            (线性投影到 latent N)
   → top-k 稀疏激活         (只保留 k 个最大通道, 其余置 0 → 真稀疏)
   → rho = Σ_t act ⊗ ln_et  (Hebbian 累积快权重 [N,D], 袋状无时序)
   → mem_ctx = act ⊗ rho    (act 加权读回)
   → h = 嵌入 + mem_ctx → LayerNorm → out(D→D)
   → +FFN (非线性读回)       (补局部字符映射容量, 已证是补能力最便宜杠杆)
   → head → vocab 分布
```

关键：`rho` 是**状态 >> 参数**的 fast-weight（`rho` 形状 [N,D]，随序列被 Hebbian 累积但不进反传参数）。**单层 O(T)**。

## 三、与 Transformer 的对比

| 维度 | FusedFW | SDPA-Transformer |
|---|---|---|
| 前向复杂度 | O(T)（线性，无 attention matrix） | O(T²)（SDPA，T×T） |
| decode/token | 常数（增量 rho 状态） | O(T)（KV-cache） |
| 时序建模 | 袋状无时序（rho 是共现池） | 完整 causal attention + 位置编码 |
| 稀疏 | top-k 稀疏激活（真稀疏，落在 latent） | 无 |

> **⚠️ 诚实**：FusedFW 的 rho 是**无时序**共现池，丢失字符级顺序/n-gram。补 FFN 仅补局部字符映射。**长上下文能力是否真的不输——这是 FAIR_BENCH_V2 要验证的问题**（v0.0.1 结果：小型 char-LM 上相当/略优，范围受限）。

## 四、覆盖的层次（后续扩展）

```
DynFW
├── Fused Fast-Weight computation   ✅ v0.0.1
├── Persistent associative state    ✅ v0.0.1 (rho)
├── Sparse activation               ✅ v0.0.1 (top-k)
├── Nonlinear readback              ✅ v0.0.1 (FFN)
└── Dynamic capability (未来)        ── DynFW-DCA / 自生长（路线图）
```

## 五、参考基线

`dynfw/models/transformer.py`：SDPA-Transformer + 正弦固定位置编码。**公平对比必须用它**（见 `fairness.md`）。
