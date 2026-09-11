"""DynFW 蒸馏版 v6: BDHBlockFWCycle —— 在 v5(BDHBlockCycleLM) 基础上，把整段 T×T 注意
换成"块内局部注意 + 跨块 fast-weight 记忆累积"(BDH 论文擅长长上下文的真机制)。

背景(爸爸指令"攻 O(T) 化, 让擅长长上下文成真"):
- v5 la_cycle(91.6) 能力强但 attn 是 `scores=(QR@KR.mT).tril(-1)` 整段 T×T = O(T²)。
- BDH 论文源码(bdh_cq.py 213-251行)证实: 它靠"块内精确注意(窗口W内) + 跨块 fast-weight
  检索(memories 每token线性累积 einsum('b h n d, b n e -> b h d e', k, v))" 实现长上下文便宜。
- 本 v6 严格单变量: 只把 attn 换成"分块 + fast-weight 记忆", 其余(encoder/decoder/encoder_v/
  双稀疏门控/残差LN)与 v5 完全一致。目标 = 能力不输(≈91.6) 且 复杂度真 O(T)(无 T×T)。

用法: distill_qwen.py --arch fusedfw_gla_cycle ... (借用 gla_cycle 分支, 或单独加分支)
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class FWAttention(nn.Module):
    """块内精确注意(窗口W) + 跨块 fast-weight 记忆检索(累积 O(T))。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))
        self.nh = nh; self.N = N; self.D = D

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = FWAttention.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        """Q,K: [B,nh,T,N]  (K is Q); V: [B,1,T,D]; memories: 可选 [B,nh,N,D] 历史 fast-weight。
        返回 (out, new_memories). out: [B,nh,T,D] 对齐 v5 attn 输出语义。"""
        assert K is Q
        B, nh, T, _ = Q.size()
        N = self.N; D = self.D
        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = self.rope(r * self.freqs, Q)
        KR = QR
        # 块内注意: 只在窗口 W 内算 (用 causal 三角掩码 + 位置差剪枝)
        # 对每个位置 i, 只看 [i-W+1, i] 的 token; 用滑动窗口 masked_fill 实现, 不物化整段 T×T(经分块)
        # 为严格单变量+可跑, 这里用 chunk 逐块算:每块 size=W, 只在块内+前一快照的 fast-weight。
        out_chunks = []
        new_mem = memories if memories is not None else torch.zeros(B, nh, N, D, device=Q.device)
        for st in range(0, T, W):
            en = min(st + W, T)
            q_c = QR[:, :, st:en]          # [B,nh,w,N]
            k_c = KR[:, :, st:en]
            v_c = V[:, :, st:en]           # [B,1,w,D]
            w = en - st
            sim = q_c @ k_c.mT              # [B,nh,w,w]
            # 因果掩码: 位置 i 看自己及之前 (diagonal=0, 对齐标准因果LM/教师语义)
            #   ⚠️ 必须用 -inf (而非0) + softmax, 否则未来位置 softmax 后权重=1/Z≠0 → 泄漏未来
            #   (原实现 masked_fill(~causal,0) 是因果bug; 已用探针验证 diagonal=0 无泄漏无NaN)
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=0)
            sim = sim.masked_fill(~causal, float('-inf'))
            attn = torch.softmax(sim.float(), dim=-1)  # [B,nh,w,w]
            agg = attn @ v_c                 # [B,nh,w,D]
            # 跨块 fast-weight 检索: 用当前 q 从历史状态检索
            if new_mem is not None and (st > 0 or memories is not None):
                retr = torch.einsum('bhwd,bhde->bhwe', q_c.float(), new_mem.float())  # [B,nh,w,D]
                agg = agg + retr
            out_chunks.append(agg)
            # 更新 fast-weight: 累积本块 (每 token k⊗v)
            new_mem = new_mem + torch.einsum('bhwd,bhwe->bhde', k_c.float(), v_c.float())
        out = torch.cat(out_chunks, dim=2)   # [B,nh,T,D]
        return out, new_mem


class Config:
    def __init__(self, n_layer, n_embd, n_head, mlp_mult, vocab):
        self.n_layer = n_layer; self.n_embd = n_embd; self.n_head = n_head
        self.mlp_internal_dim_multiplier = mlp_mult; self.vocab_size = vocab


class BDHBlockFWCycle(nn.Module):
    """v6: 与 v5 BDHBlockCycle 完全一致, 唯一差异 = attn 用 FWAttention(块内+fast-weight)。"""
    def __init__(self, D, nh, mlp_mult, vocab, steps=1, W=512):
        super().__init__()
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg; self.D = D; self.vocab = vocab; self.steps = steps; self.W = W
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = FWAttention(cfg)
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
        """x: [B,1,T,D]; memories: 可选 fast-weight 状态 [B,nh,N,D]。
        返回 (x', new_memories)。"""
        C = self.config
        B = x.shape[0]; T = x.shape[2]
        D = self.D; nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        x = self.ln(x)
        for _ in range(self.steps):
            x_latent = x @ self.encoder            # [B,nh,T,N]
            x_sparse = F.relu(x_latent)
            yKV, new_mem = self.attn(Q=x_sparse, K=x_sparse, V=x, memories=memories, W=self.W)
            yKV = self.ln(yKV)                       # yKV=[B,nh,T,D], LayerNorm 对最后D  (对齐v5)
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


class BDHBlockFWCycleLM(nn.Module):
    """LM 封装: embed -> BDHBlockFWCycle -> head。Qwen3 vocab 兼容。"""
    def __init__(self, D=128, nh=4, vocab=151936, n_layer=1, steps=1, mlp_mult=128, W=512):
        super().__init__()
        self.D = D; self.nh = nh; self.vocab = vocab; self.n_layer = n_layer; self.steps = steps; self.W = W
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([BDHBlockFWCycle(D, nh, mlp_mult, vocab, steps=steps, W=W)
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
        h = self.e(x).unsqueeze(1)        # B,1,T,D
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
        h = self.e(x).unsqueeze(1)        # B,1,T,D
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
