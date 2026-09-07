# Roadmap —— DynFW 路线图

> 原则：**No benchmark regression, no hidden changes.** 每个版本同时保存旧模型+旧 benchmark + 新模型+新 benchmark。

## v0.0.1（当前，冻结）
✅ FusedFW + FFN · SDPA-Transformer 基线 · 参数匹配基准 · 5 seed · 固定 val · prefill/KV-cache decode 分开测 · CPU 口径 · 诚实限制（`limitations.md`）
- 结果：小型 char-LM 上能力相当/略优 + 训练 1.2-1.4x + decode ~2.2x + prefill 长上下文 1.75x@4K。

## 待补（不动结论方向的覆盖扩展）
- [x] **batch 1 / 8 / 32**（`docs/FAIR_BENCH_V2_EXTENDED.md`，2026-09-07）：O(T) 优势跨 batch 成立；FW 长序快 2.8-5.6x，短序(~256)略慢~1.0-1.1x；decode 仍 FW ~2.1x。
- [x] **T = 8K**（同上）：T 到 8192，FW/TF 单调降（batch1 到 0.18）；大窗口 FW 优势随 T 放大。
- [ ] **GPU 列**（FlashAttention 基线）——需有卡，标"待补"不假装测过。

## 下一阶段候选（需先锁判据，一次一组变量）
1. **真实模型蒸馏 benchmark**：审计 V11 §六 的门槛——"跑完 FAIR_BENCH_V2 才有资格决定进不进成熟开源模型蒸馏"。这是回答"能否替代 Transformer"的关键一役。
2. **长上下文质量评估**：给定 T=512/1024/4096 的真实困惑度/下游任务，验证"长上下文能力真的不输"而非只看成本。
3. **DynFW-DCA**：把演进线接回 DCA 动态能力（能力池 + 免训 Interference Predictor 管理 REUSE/SPAWN）——但 DCA 目前是独立线程（审计 V12：DCA 可作辅助，主线优先）。
4. **自生长 / 结构可训练**：（探索期遗留）验证结构能否按需生长，还是保持固定结构高效。

## 版本演进规则
- 每次改基准标准 → 升版本号 + 在 `fairness.md` 记录改了哪条、为什么。
- 禁止"为证新更强而换 benchmark"。
- 所有 negative / partial 结果如实归档，不删。
