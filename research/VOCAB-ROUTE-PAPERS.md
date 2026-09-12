# 词表路线论文调研（2026-09-11）

调研对象：词表大小 / 有无词表 / 换词表 / 省词表税 / 词表评测

## A. 词表缩放律 —— 决定「词表该多大」

| arXiv | 标题 | 会场 | 核心结论 |
|---|---|---|---|
| **2407.13623** | Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies | NeurIPS 2024 | 存在 compute-optimal 词表；N_nv=0.08·C^0.50、N_v=0.20·C^0.42；**N_v_opt ∝ N_nv^0.83（γ<1，词表参数应比非词表参数慢增长）**；loss 的词表修正项 f(V)=0.0064·log²V−0.1581·logV+1.2047 |
| 2501.16975 | Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling | ICML 2025 | **输入词表 vs loss 是 log-linear**；词表×128 → 400M 模型追平 1B baseline loss，零额外算力；**必须解耦输入/输出词表**（输入侧扩大几乎零算力开销，输出侧扩大显著增加小模型开销） |

### 2407.13623 关键数字（原文）
- 训练规模：33M–3B 参数 / 最多 500B 字符；词表拟合范围 4K–96K
- 预测 Llama2-70B 最优词表 ≥ **216K**（实际 32K，差 7×）
- 32K → 43K：ARC-Challenge **29.1 → 32.0**（同 2.3e21 FLOPs）
- 自校验（我用论文公式复算 Llama2-70B）：261K，量级吻合 ✓

### 代入我们的规模（我用论文公式外推）
| 场景 | N_nv | d_model | V_opt | 实际用 | 超标 |
|---|---|---|---|---|---|
| 我们 v6/v7 对轰 | 25.7M | 128 | **21.9K** | 151936 | **6.9×** |
| 我们 600M 档 | 457.6M | 1024 | **30.7K** | 151936 | **5.0×** |
| MiniCPM5-2B（面壁） | 1.98B | 2048 | **52.6K** | 130560 | **2.5×** |
| Llama2-70B | 69.5B | 8192 | 261K | 32000 | 0.12×（论文说太小） |

⚠️ 外推限制：论文拟合范围 33M–3B 参数 / 词表 4K–96K，我们的 25.7M 低于其参数下界；ws 架构不满足「每层独立参数」假设，该公式对 ws 不适用。

## B. 有无词表 / byte-level —— 决定「要不要词表」

| arXiv | 标题 | 会场 | 核心结论 |
|---|---|---|---|
| **2412.09871** | Byte Latent Transformer: Patches Scale Better Than Tokens | Meta, 2024-12 | **按下一字节熵动态分段成 patch**；首次 FLOP 受控 byte-level 缩放研究（到 8B 参数 / 4T 字节）；**固定推理成本下缩放显著优于 token-based**；首次证明无固定词表能 scale |
| 2605.08044 | Fast Byte Latent Transformer | 2026-05 | BLT 加速版 |
| 2105.13626 | ByT5: Towards a token-free future | TACL 2022 | byte-level 起点，当时打不过 BPE |
| 2401.13660 | MambaByte | COLM 2024 | 无分词 + 线性 |
| 2506.14123 | Sampling from Your Language Model One Byte at a Time | 2025-06 | 逐字节采样 |
| 2608.28151 | Nested Byte-Level Vocabularies Are Cheap to Deploy and Expensive to Share | 2026-08 | **预注册负面结果** |

## C. 换词表 / 词表手术 —— ★ 最关键（打破「词表单向门」）

| arXiv | 标题 | 会场 | 核心结论 |
|---|---|---|---|
| **2506.06607** | Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit | 2025-06 | **无训练**移植词表：OMP 用 anchor token 稀疏重建 OOV embedding，再转回 base 空间；击败 zero-init/mean-init/WECHSEL/FOCUS/ZETT；**配套工具 mergekit-tokensurgeon**；明说可用于跨词表蒸馏/投机解码/集成/域适配 |
| **2503.20083** | Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching | **NeurIPS 2025** | **首个跨完全不同词表的有效蒸馏**；把词表移植当自蒸馏；**支持 subword → byte-level 快速迁移**；数学专用大模型 → 不同词表的小通用模型成功 |
| 2405.07883 | Zero-Shot Tokenizer Transfer (ZeTT) | NeurIPS 2024 | 超网络：输入 tokenizer → 预测 embedding；接近原模型性能；剩余差距 **<1B token 续训**可补 |
| 2408.04303 | Trans-Tokenization and Cross-lingual Vocabulary Transfers | COLM 2024 | 跨语言词表迁移 |
| 2402.09977 | Fast Vocabulary Transfer for LM Compression | EMNLP 2022 | 压模型用词表迁移 |
| 2506.01535 | Dictionaries to the Rescue | ACL 2025 | 双语词典做跨语言词表迁移 |
| 2402.14714 | Efficient Vocabulary Expansion Towards Multilingual LLMs | 2024-02 | 多语扩词表 |
| 2508.15807 | Vocabulary Expansion via KL-Based Self-Distillation | 硕士论文 2025-08 | KL 自蒸馏扩词表 |

## D. 省词表税 / 省 token —— 直接可用

| arXiv | 标题 | 会场 | 核心结论 |
|---|---|---|---|
| **2605.29459** | Kronecker Embeddings: Byte-Level Structured Token Representations for Parameter-Efficient LMs | 2026-05（单作者） | 确定性字节级因子分解替代 |V|×d 表，**兼容标准 BPE**；**消除 91–94% 输入侧参数**；V=131072 时 **4.5MB 缓冲 vs 2.15GB 表**，步时开销 0.01–0.24%；124M 三 seed 对照 val loss 低 2.5±0.2%（0.083±0.007 nats，~9% 困惑度），收敛快 1.43×；typo 鲁棒性 +8.2pp。**代价**：字节相似语义远的对（compute/commute）会聚类，需早期 attention 消歧 |
| **2511.20849** | Length-MAX Tokenizer | TMLR 2025 | 直接优化「平均 token 长度」（图分割 + 贪心）；**比 BPE 少 14–18% token（10K–50K 词表），64K 时少 13.0%**；GPT-2 124M/355M/1.3B 从零训 5 seed：**步数少 18.5%/17.2%/18.5%**，推理延迟低 13.7%/12.7%/13.7%，124M 吞吐 +16%；LAMBADA ppl −11.7%、HellaSwag +4.3%；**embedding+KV-cache 内存 −18%** |
| 2603.02597 | GPUTOK: GPU Accelerated Byte Level BPE | 2026-03 | tokenizer 加速 |

## E. 词表评测

| arXiv | 标题 | 会场 | 核心结论 |
|---|---|---|---|
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite | COLM 2026 | 超越 fertility/压缩率；加 UTF-8 边界完整性、数字位值边界对齐；**信息论指标预测语言建模能力（ρ≤0.80），结构敏感指标关联任务准确率** |
| 2506.03101 | Beyond Text Compression: Evaluating Tokenizers Across Scales | ACL 2025 | **小模型能预测大模型上的词表差异**（省算力）；**英文任务词表选择几乎无影响，多语种有持续差异**；提 Zipf 律内在指标比压缩率更相关 |
| 2310.08754 | Tokenizer Choice For LLM Training: Negligible or Crucial? | 2023-10 | 已在团队知识库 |
| 2012.15613 | How Good is Your Tokenizer? | ACL 2021 | 多语模型单语性能 vs 词表 |
| 2606.15044 | Equity with Efficiency: Tokenizers for Multilingual LLMs | 2026-06 | 多语词表实证 |

## 决策修正（本轮）

1. **「词表单向门」不成立** —— ALM (NeurIPS 2025) 已实现跨词表蒸馏，OMP 已实现无训练移植 → 可以「自训小词表 + 吃面壁软标签」
2. **词表大小应由缩放律定，不是拍脑袋** —— 我们 25.7M 模型 V_opt≈22K、600M 档≈31K；用 151936 超 5–7×
3. **输入/输出词表应解耦** —— 输入侧可放大（算力近免费），输出侧必须小（每 token 2VD 算力）
4. **省输入侧参数有现成方法** —— Kronecker 干掉 91–94% 输入侧参数且保 BPE 兼容
5. **同词表大小还能再省 13–18% token** —— Length-MAX（直接换算法，不改架构）
6. **byte-level 的正解是 BLT 式动态 patch，不是纯 256 字节**
