"""DynFW 蒸馏版 raw 分块：BDH-CQ 官方 raw attention（无 softmax）+ 块内注意 + 跨块 fast-weight。

单向变量 vs v6(fused_fw_fw_cycle.py)：结构(完整读回/encoder/decoder/双稀疏门控/残差LN)全同，
唯一差异 = FWAttention 用 **raw（无 softmax）+ tril(-1)+mask=0**（BDH-CQ 官方 bh_cq.py 机制），
而非 v6 的 softmax + diagonal=0 + mask=-inf。

为什么 raw 是正解（2026-09-10 转向依据）：
- v6 的 softmax 分块在严格因果下 = 104.50，输给 BDH(raw+tril) 88.41 —— softmax 是 BDH 系承重件禁忌。
- BDH-CQ 官方 bh_cq.py 用 raw(无softmax) + masked_fill(~causal, 0.)：raw 下 mask=0 是算术真0(非 softmax(0)=1/Z) → 严格因果。
- 目标是「能力达标(≈la_cycle 84.57/bdh 88.41) + 真 O(T)(跨块 fast-weight) + 严格因果」。
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class FWAttentionRaw(nn.Module):
    """BDH-CQ 官方 raw 分块注意(无 softmax) + 跨块 fast-weight 记忆累积(O(T))。
    raw 下 masked_fill(~causal, 0) 是真0(非 softmax(0)=1/Z)，严格因果，无 NaN。"""
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
        pc, ps = FWAttentionRaw.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        """Q,K:[B,nh,T,N] (K is Q); V:[B,1,T,D]; memories: 可选[B,nh,N,D]历史fast-weight。
        返回 (out[B,nh,T,D], new_mem[B,nh,N,D])。"""
        assert K is Q
        B, nh, T, _ = Q.size()
        N = self.N; D = self.D
        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = self.rope(r * self.freqs, Q)
        KR = QR
        out_chunks = []
        new_mem = memories if memories is not None else torch.zeros(B, nh, N, D, device=Q.device)
        for st in range(0, T, W):
            en = min(st + W, T)
            q_c = QR[:, :, st:en]
            k_c = KR[:, :, st:en]
            v_c = V[:, :, st:en]
            w = en - st
            sim = q_c @ k_c.mT              # [B,nh,w,w]
            # BDH-CQ 官方: tril(-1) omit self + masked_fill(~causal, 0.)  RAW 无 softmax
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=-1)
            attn = sim.masked_fill(~causal, 0.0)   # raw: 未来=真0, 严格因果
            agg = attn @ v_c                # [B,nh,w,D]
            # 跨块 fast-weight 检索
            if new_mem is not None and (st > 0 or memories is not None):
                retr = torch.einsum('bhwd,bhde->bhwe', q_c, new_mem)
                agg = agg + retr
            out_chunks.append(agg)
            # 累积 fast-weight O(T)
            new_mem = new_mem + torch.einsum('bhwd,bhwe->bhde', k_c, v_c)
        out = torch.cat(out_chunks, dim=2)
        return out, new_mem


class Config:
    def __init__(self, n_layer, n_embd, n_head, mlp_mult, vocab):
        self.n_layer = n_layer; self.n_embd = n_embd; self.n_head = n_head
        self.mlp_internal_dim_multiplier = mlp_mult; self.vocab_size = vocab


class BDHBlockRawFWCycle(nn.Module):
    """与 v6 BDHBlockFWCycle 完全一致，唯一差异 = attn 用 FWAttentionRaw(raw)。"""
    def __init__(self, D, nh, mlp_mult, vocab, steps=1, W=512):
        super().__init__()
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg; self.D = D; self.vocab = vocab; self.steps = steps; self.W = W
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = FWAttentionRaw(cfg)
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
        """x:[B,1,T,D]; 返回 (x', new_memories)。"""
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


class BDHBlockRawFWCycleLM(nn.Module):
    """LM 封装: embed -> BDHBlockRawFWCycle -> head。Qwen3 vocab 兼容。"""
    def __init__(self, D=128, nh=4, vocab=151936, n_layer=1, steps=1, mlp_mult=128, W=512):
        super().__init__()
        self.D = D; self.nh = nh; self.vocab = vocab; self.n_layer = n_layer; self.steps = steps; self.W = W
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([BDHBlockRawFWCycle(D, nh, mlp_mult, vocab, steps=steps, W=W)
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

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
