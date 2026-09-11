"""DynFW 蒸馏版 v6.6: Gated DeltaNet 门控衰减 —— 在 v6(fused_fw_fw_cycle) 基础上，
把 fast-weight 的无脑累积 `M += k⊗v` 改成 Gated DeltaNet 的门控衰减 `M = α·M + k⊗v`
(α=可学习数据依赖门控)。严格单变量: 只改记忆更新规则, 其余(块内注意/双稀疏门控/encoder/
decoder/encoder_v/残差LN)与 v6 完全一致。

为何选门控(而非 VLA 残差修正):
- v6.5(VLA 残差修正 `M += β·k⊗(v-M·k)`)在我们载体实测爆 inf —— 根因=VLA 前提 d_k≈d_v 量级匹配,
  但我们 k_c=x_sparse(0.089) v_c=原始x(0.79) 量级差 8.8 倍 + 维度不同(N vs D), resi 巨大→爆。
- 门控衰减 `M=α·M+k⊗v` 不要求 k/v 量级匹配(不需要预测残差), 只加一个数据依赖遗忘门 α,
  直接压在 k/v 量级不匹配的载体上仍成立 → 载体真正兼容(军规: 先判载体再嫁接)。

用法: distill_qwen.py --arch fusedfw_gdn_cycle ...
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class GDNFastAttn(nn.Module):
    """块内精确注意(窗口W) + Gated DeltaNet 门控衰减 fast-weight 记忆。唯一单变量改动。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))
        self.nh = nh; self.N = N; self.D = D
        # Gated DeltaNet: 数据依赖门控 α = sigmoid(linear(x_sparse))  —— 用 query 侧算 gate
        self.gate = nn.Linear(N, 1)   # 从 k_c(N维) 映射到单门控标量
        # 关键: 初始化 gate.bias=+4 → sigmoid(4)≈0.98, 初始几乎不衰减(等价 v6 无门控),
        # 让训练自己学会何时衰减(而非一开始就砍一半). 严格单变量: 只改初始化.
        nn.init.constant_(self.gate.bias, 4.0)

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = GDNFastAttn.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        """Q,K:[B,nh,T,N](K is Q); V:[B,1,T,D]; memories: 可选 [B,nh,N,D]。
        返回 (out, new_mem)。out:[B,nh,T,D] 对齐 v6。"""
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
            q_c = QR[:, :, st:en]          # [B,nh,w,N]
            k_c = KR[:, :, st:en]          # [B,nh,w,N]
            v_c = V[:, :, st:en]           # [B,1,w,D]
            w = en - st
            sim = q_c @ k_c.mT              # [B,nh,w,w]
            causal = torch.tril(torch.ones(w, w, device=Q.device, dtype=torch.bool), diagonal=0)
            sim = sim.masked_fill(~causal, float('-inf'))
            attn = torch.softmax(sim.float(), dim=-1)
            agg = attn @ v_c                 # [B,nh,w,D]
            if new_mem is not None and (st > 0 or memories is not None):
                retr = torch.einsum('bhwd,bhde->bhwe', q_c.float(), new_mem.float())
                agg = agg + retr
            out_chunks.append(agg)
            # === Gated DeltaNet 门控衰减记忆更新 (唯一单变量改动) ===
            # 数据依赖门控 α = sigmoid(gate(k_c)), [B,nh,w,1]
            alpha = torch.sigmoid(self.gate(k_c.float()))      # [B,nh,w,1]
            # 衰减: M <- α·M (广播到 [B,nh,N,D] 的最后一维时间轴, 每 token gate)
            # 块级近似: 用块内 α 的平均作为该块衰减 (简化, 保持 O(T))
            alpha_blk = alpha.mean(dim=2, keepdim=True)        # [B,nh,1,1]
            new_mem = alpha_blk * new_mem
            # 写入: M <- M + k⊗v (原始累积, 门控已衰减)
            new_mem = new_mem + torch.einsum('bhwd,bhwe->bhde', k_c.float(), v_c.float())
        out = torch.cat(out_chunks, dim=2)
        return out, new_mem


class Config:
    def __init__(self, n_layer, n_embd, n_head, mlp_mult, vocab):
        self.n_layer = n_layer; self.n_embd = n_embd; self.n_head = n_head
        self.mlp_internal_dim_multiplier = mlp_mult; self.vocab_size = vocab


class BDHBlockGDNCycle(nn.Module):
    """v6.6: 与 v6 BDHBlockFWCycle 完全一致, 唯一差异 = attn 用 GDNFastAttn(门控衰减)。"""
    def __init__(self, D, nh, mlp_mult, vocab, steps=1, W=512):
        super().__init__()
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg; self.D = D; self.vocab = vocab; self.steps = steps; self.W = W
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = GDNFastAttn(cfg)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(0.0)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.apply(self._init_weights)
        # 门控 bias 初始化为 +4 → sigmoid≈0.98, 初始几乎不衰减(等价 v6 无门控), 训练自学何时衰减
        # (在 apply 之后显式重设, 避免 _init_weights 的 Linear 分支把它归零)
        nn.init.constant_(self.attn.gate.bias, 4.0)

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


class BDHBlockGDNCycleLM(nn.Module):
    """LM 封装: embed -> BDHBlockGDNCycle -> head。"""
    def __init__(self, D=128, nh=4, vocab=151936, n_layer=1, steps=1, mlp_mult=128, W=512):
        super().__init__()
        self.D = D; self.nh = nh; self.vocab = vocab; self.n_layer = n_layer; self.steps = steps; self.W = W
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([BDHBlockGDNCycle(D, nh, mlp_mult, vocab, steps=steps, W=W)
                                     for _ in range(n_layer)])
        self.head = nn.Linear(D, vocab, bias=False)
        self.apply(self._init_weights)
        # LM 层 apply 会递归覆盖 block 里的 gate.bias=4, 此处重新设定(初始不衰减)
        for b_ in self.blocks:
            nn.init.constant_(b_.attn.gate.bias, 4.0)

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
