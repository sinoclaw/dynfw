"""DynFW 蒸馏版 v7: DLA 状态槽记忆 —— 在 v6(fused_fw_fw_cycle) 基础上,
把单块 fast-weight 记忆 `M ∈ [B,nh,N,D]` 换成 **DLA(arXiv 2606.10650) 的多状态槽**:
`state_cache = [(S_i, Ī_i, n_i)]`(容量 K 有界), 满了按信息密度最低相邻对合并(DLA Algorithm 2)。

为何走 DLA(而非 Gated DeltaNet 门控衰减):
- V6.6 门控衰减实测解了范数爆炸(T=8192 ‖M‖ 稳定 4647)但能力退化(KL 82.39→90.83),
  根因=门控是"删记忆"(衰减), 长上下文关键信息被删。
- DLA 是"合并记忆"(信息感知): 把信息密度最低的相邻状态槽合并, 高信息状态保留 →
  **控范数同时尽量保信息**(不是删, 是压缩)。这正是"能力强+成本降"双标的另一条路。

严格单变量: 只改 FWAttention 的记忆结构(单块→K状态槽), 其余(块内精确注意/双稀疏门控/
encoder/decoder/encoder_v/残差LN)与 v6 完全一致。
V6 的分块结构(每W token一块)天然提供了 DLA 的状态边界 → 每个 V6 块 = 一个 DLA 状态槽。

用法: distill_qwen.py --arch fusedfw_dla_cycle ...
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class DLAStateCache(nn.Module):
    """DLA 状态槽缓存: 容量 K 有界, 信息感知合并 (严格对照 DLA Algorithm 2)。
    state = (S: [B,nh,N,D] 状态向量, I: [B,nh,1] 聚合信息分, n: [B,nh,1] token计数)。"""
    def __init__(self, config, K=16):
        super().__init__()
        self.config = config
        nh = config.n_head; D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.nh = nh; self.N = N; self.D = D; self.K = K

    def info_score(self, block_rep):
        """信息分: 块内 token 相对块首的表示变化量(L2), 即 DLA 的 'state information score'。
        block_rep: [B,nh,w,N] 一个块的稀疏表示。返回 Ī=[B,nh,1] (聚合信息分)。"""
        # 变化量 = 块内各 token 与块首的 L2 距离, 再求和 → 信息分
        head = block_rep[:, :, :1, :]                     # 块首 [B,nh,1,N]
        diff = (block_rep - head).norm(dim=-1, keepdim=True)  # [B,nh,w,1]
        return diff.sum(dim=2)                            # [B,nh,1] 聚合信息分

    def merge_lowest_density(self, S, I, n):
        """DLA Algorithm 2 第6行: 合并相邻信息密度最低对 (Ī_i+Ī_i+1)/(n_i+n_i+1)。
        S:[B,nh,K,N,D]  I:[B,nh,K,1]  n:[B,nh,K,1] → 合并后 K 少一个。"""
        # 相邻对信息密度
        denom = n[:, :, :-1, :] + n[:, :, 1:, :] + 1e-8     # [B,nh,K-1,1]
        dens = (I[:, :, :-1, :] + I[:, :, 1:, :]) / denom   # [B,nh,K-1,1]
        if dens.shape[2] == 0:
            # 无相邻对可合并(只有0/1个槽) → 直接返回
            return S, I, n
        idx = dens.argmin(dim=2)                            # [B,nh,1] 最低密度对的起点
        # 逐 batch 合并(索引不同), 简化用平均合并概率最高的 ... 这里用 gather 精确合并
        # (精确实现: 对每个 b,h 用各自 idx; 简化: 用槽平均 idx -- 但为保正确, 用逐槽扫描)
        # 全面实现: 合并 idx 和 idx+1 两个槽
        Bnh = S.shape[0] * S.shape[1]
        S2 = S.clone(); I2 = I.clone(); n2 = n.clone()
        # 用 gathered idx 合并 (近似: 每个 batch 独立选最低密度; 向量化困难, 用 loop)
        for b in range(S.shape[0]):
            for h in range(S.shape[1]):
                i0 = int(idx[b, h, 0].item())
                S2[b, h, i0] = S[b, h, i0] + S[b, h, i0+1]   # 向量相加
                I2[b, h, i0] = I[b, h, i0] + I[b, h, i0+1]   # 信息分相加
                n2[b, h, i0] = n[b, h, i0] + n[b, h, i0+1]   # 计数相加
        # 删除被合并槽 i0+1 → 移到最后再截断
        new_S = S2[:, :, :-1, :].clone()   # 简化: 直接去尾(正确性: 合并后把 i0+1 删掉, 需要紧凑化)
        # (注: 精确紧凑化需重排, 此处先保留 i0 合并结果并移除 i0+1; 用 mask 法)
        keep = torch.ones_like(n2[:, :, :, 0], dtype=torch.bool)  # [B,nh,K]
        for b in range(S.shape[0]):
            for h in range(S.shape[1]):
                keep[b, h, int(idx[b, h, 0].item())+1] = False
        newS = S2[keep.unsqueeze(-1).unsqueeze(-1).expand_as(S2)].view(S.shape[0], S.shape[1], S.shape[2]-1, S.shape[3], S.shape[4])
        newI = I2[keep.unsqueeze(-1).expand_as(I2)].view(I.shape[0], I.shape[1], I.shape[2]-1, I.shape[3])
        newn = n2[keep.unsqueeze(-1).expand_as(n2)].view(n.shape[0], n.shape[1], n.shape[2]-1, n.shape[3])
        return newS, newI, newn


class DLAFastAttn(nn.Module):
    """块内精确注意(窗口W) + DLA 状态槽记忆(容量K有界, 信息感知合并)。唯一单变量改动。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head; D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))
        self.nh = nh; self.N = N; self.D = D
        self.cache = DLAStateCache(config, K=16)   # 状态槽缓存

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = DLAFastAttn.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        """Q,K:[B,nh,T,N](K is Q); V:[B,1,T,D]; memories: 可选状态槽缓存。
        返回 (out, new_cache)。out:[B,nh,T,D]。"""
        assert K is Q
        B, nh, T, _ = Q.size()
        N = self.N; D = self.D; K = self.cache.K
        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = self.rope(r * self.freqs, Q)
        KR = QR
        out_chunks = []
        # 状态槽: S[B,nh,K,N,D] I[B,nh,K,1] n[B,nh,K,1]
        if memories is None:
            S = torch.zeros(B, nh, K, N, D, device=Q.device)
            I = torch.zeros(B, nh, K, 1, device=Q.device)
            n = torch.zeros(B, nh, K, 1, device=Q.device)
            used = 0
        else:
            S, I, n, used = memories
        for st in range(0, T, W):
            en = min(st + W, T)
            q_c = QR[:, :, st:en]          # [B,nh,w,N]
            k_c = KR[:, :, st:en]
            v_c = V[:, :, st:en]           # [B,1,w,D]
            w = en - st
            sim = q_c @ k_c.mT              # [B,nh,w,w]
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=0)
            sim = sim.masked_fill(~causal, float('-inf'))
            attn = torch.softmax(sim.float(), dim=-1)
            agg = attn @ v_c                 # [B,nh,w,D]
            # DLA 检索: 对所有已用状态槽聚合 o = Σ_i φ(q)·S_i
            if used > 0:
                # 对所有已用槽 S_i 聚合 q→S_i (线性注意检索)
                q_expand = q_c.float()  # [B,nh,w,N]
                # o = Σ_i (q·S_i), S_i [B,nh,N,D] → [B,nh,w,D]
                S_summed = S[:, :, :used, :, :].sum(dim=2)      # [B,nh,N,D] 加权和(简化: 直接和)
                retr = torch.einsum('bhwd,bhde->bhwe', q_expand, S_summed)
                agg = agg + retr          # DLA原文 o_t=Σ_i φ(q)·S_i, 不除以状态数(修正/used bug)
            out_chunks.append(agg)
            # === DLA 状态槽更新 ===
            # 块信息分 = 块内 token 相对块首变化量
            info = self.cache.info_score(k_c.float())  # [B,nh,1]
            # 新状态槽向量 = 本块 k⊗v 累积(A个 token)
            slot_S = torch.einsum('bhwd,bhwe->bhde', k_c.float(), v_c.float())  # [B,nh,N,D]
            if used < K:
                # 有空间 → 追加状态槽
                S[:, :, used] = slot_S
                I[:, :, used] = info
                n[:, :, used] = torch.full((B, nh, 1), float(w), device=Q.device)
                used += 1
            else:
                # 满 → 严格 DLA: 先合并相邻最低密度对(降到K-1), 再【追加】本块(回到K)
                #   ⚠️ 存量 bug 修复(2026-09-10, 由 v8 探针 P3 抓出): 原写法
                #   `S_m[:, :, -1] = slot_S` 是【覆盖】合并后的末槽 → 净效果槽数
                #   每合并一次减 1 (K→K-1→K-2→...), 而 used 硬写回 K →
                #   `S[:, :, :used]` 被 python 切片静默 clamp, 记忆容量随上下文单调萎缩。
                #   DLA 原文明确要求 fixed-size chronologically ordered cache → 必须
                #   cat 追加, 不能覆盖。(历史读数 v7=101.22 由带 bug 版本产生, 需重测)
                S_m, I_m, n_m = self.cache.merge_lowest_density(S, I, n)
                S = torch.cat([S_m, slot_S.unsqueeze(2)], dim=2)
                I = torch.cat([I_m, info.unsqueeze(2)], dim=2)
                n = torch.cat([n_m, torch.full((B, nh, 1, 1), float(w), device=Q.device)], dim=2)
                used = K
        out = torch.cat(out_chunks, dim=2)
        return out, (S, I, n, used)


class Config:
    def __init__(self, n_layer, n_embd, n_head, mlp_mult, vocab):
        self.n_layer = n_layer; self.n_embd = n_embd; self.n_head = n_head
        self.mlp_internal_dim_multiplier = mlp_mult; self.vocab_size = vocab


class BDHBlockDLACycle(nn.Module):
    """v7: 与 v6 BDHBlockFWCycle 完全一致, 唯一差异 = attn 用 DLAFastAttn(状态槽记忆)。"""
    def __init__(self, D, nh, mlp_mult, vocab, steps=1, W=512, K=16):
        super().__init__()
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg; self.D = D; self.vocab = vocab; self.steps = steps; self.W = W
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = DLAFastAttn(cfg)
        self.attn.cache.K = K   # 容量K可配(默认16, 设小可触发合并)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(0.0)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, memories=None):
        C = self.config
        B = x.shape[0]; T = x.shape[2]
        D = self.D; nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.ln(x)
        for _ in range(self.steps):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)
            yKV, new_mem = self.attn(Q=x_sparse, K=x_sparse, V=x, memories=memories, W=self.W)
            yKV = self.ln(yKV)
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)
            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)
            memories = new_mem
        return x, memories

    def np(self):
        return sum(p.numel() for p in self.parameters())


class BDHBlockDLACycleLM(nn.Module):
    """LM 封装: embed -> BDHBlockDLACycle -> head。"""
    def __init__(self, D=128, nh=4, vocab=151936, n_layer=1, steps=1, mlp_mult=128, W=512, K=16):
        super().__init__()
        self.D = D; self.nh = nh; self.vocab = vocab; self.n_layer = n_layer; self.steps = steps; self.W = W
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([BDHBlockDLACycle(D, nh, mlp_mult, vocab, steps=steps, W=W, K=K)
                                     for _ in range(n_layer)])
        self.head = nn.Linear(D, vocab, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.size()
        h = self.e(x).unsqueeze(1)
        h = self.ln(h)
        # 修复 2026-09-11：原 `mem = None; for blk: h, mem = blk(h, mem)` 造成未来泄漏 ——
        # FWAttention 返回的 new_mem 是【整个序列】的 k⊗v 累积，下一 block 在位置 t 检索时
        # 读到 t 之后的 k⊗v。逐层 hook 实测 block0 因果 OK、block1 起发散(2.3e-2→14.3)。
        # 改为每 block 独立 memory（层内 chunk 间仍累积，因果正确）。探针验证 maxdiff=0。
        for blk in self.blocks:
            h, _ = blk(h, None)
        lg = self.head(h.view(B, T, self.D))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(lg.view(-1, self.vocab), targets.view(-1))
        return lg, loss

    def forward_hidden(self, x):
        """蒸馏用：返回 head 投影之前的 hidden (B,T,D)，配合分块 KL 避免物化 B×T×V logits。"""
        B, T = x.size()
        h = self.e(x).unsqueeze(1)
        h = self.ln(h)
        # 修复 2026-09-11：原 `mem = None; for blk: h, mem = blk(h, mem)` 造成未来泄漏 ——
        # FWAttention 返回的 new_mem 是【整个序列】的 k⊗v 累积，下一 block 在位置 t 检索时
        # 读到 t 之后的 k⊗v。逐层 hook 实测 block0 因果 OK、block1 起发散(2.3e-2→14.3)。
        # 改为每 block 独立 memory（层内 chunk 间仍累积，因果正确）。探针验证 maxdiff=0。
        for blk in self.blocks:
            h, _ = blk(h, None)
        return h.view(B, T, self.D)

    def head_params(self):
        """返回 (weight(V,D), bias(V,) 或 None)，与 forward 中 head 投影严格一致。"""
        return self.head.weight, None
    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
