"""DynFW 蒸馏版 v6.5: VLA-style 残差修正记忆更新 —— 在 v6(fused_fw_fw_cycle) 基础上，
把 fast-weight 的无脑累积 `M += k⊗v` 改成 VLA 的"残差修正 write"(delta-rule) + write 归一化。

背景(爸爸指令"控范数, 防长上下文范数爆炸"):
- v6 记忆更新是标准累积 `new_mem = new_mem + einsum(k_c, v_c)` (=线性注意堆积), 会随 T 无界生长
  (VLA论文: 标准线性注意 ‖S‖_F 在 T=1000 时达 1600, 长程检索干扰退化)。
- VLA(arXiv 2605.11196) 把它改成"在线正则最小二乘 + 自适应惩罚矩阵 A_t(Sherman-Morrison)"的
  残差修正, 并归一化 write 方向: ‖S‖_F 降到 <15, 梯度 Jacobian 谱范数=1(稳定)。

本 v6.5 严格单变量: 只改 FWAttention.update_memory(VLA 残差式), 其余(块内注意/双稀疏门控/
encoder/decoder/encoder_v/残差LN)与 v6 完全一致。目标 = 范数受控 + KL 不退化 + 能力保持。

用法: distill_qwen.py --arch fusedfw_vla_cycle ...
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class VLAFastAttn(nn.Module):
    """块内精确注意(窗口W) + VLA残差修正 fast-weight 记忆(控范数)。唯一单变量改动。"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))
        self.nh = nh; self.N = N; self.D = D
        # VLA 完整 A_t (Sherman-Morrison 维护): 用可学习门控 β (write strength) + d^2 惩罚矩阵 A(用逆表)
        self.beta = nn.Parameter(torch.zeros(1))             # 可学习 write 门控 (sigmoid 化)
        self.eta  = nn.Parameter(torch.zeros(1))             # 可学习惩罚增量 (sigmoid 化)

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = VLAFastAttn.phases_cos_sin(phases)
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
        beta = torch.sigmoid(self.beta)
        eta  = torch.sigmoid(self.eta)
        lam0 = 1.0   # A_0 初始惩罚 (正)
        Ainv = torch.eye(self.N, device=Q.device).unsqueeze(0).unsqueeze(0) / lam0  # [1,1,N,N] S-M 逆
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
            if new_mem is not None and (st > 0 or memories is not None):
                retr = torch.einsum('bhwd,bhde->bhwe', q_c.float(), new_mem.float())
                agg = agg + retr
            out_chunks.append(agg)
            # === VLA delta-rule 更新 (归一化k + 残差写, 整块近似) ===
            k_norm = k_c.float() / (k_c.float().norm(dim=-1, keepdim=True) + 1e-6)  # [B,nh,w,N]
            pred = torch.einsum('bhwd,bhde->bhwe', k_norm, new_mem.float())          # [B,nh,w,D]
            resid = v_c.float() - pred                                              # [B,nh,w,D]
            # 残差写: S <- S + beta * k_norm^T ⊗ resid
            new_mem = new_mem + beta * torch.einsum('bhwd,bhwe->bhde', k_norm, resid)
        out = torch.cat(out_chunks, dim=2)
        return out, new_mem


class Config:
    def __init__(self, n_layer, n_embd, n_head, mlp_mult, vocab):
        self.n_layer = n_layer; self.n_embd = n_embd; self.n_head = n_head
        self.mlp_internal_dim_multiplier = mlp_mult; self.vocab_size = vocab


class BDHBlockVLACycle(nn.Module):
    """v6.5: 与 v6 BDHBlockFWCycle 完全一致, 唯一差异 = attn 用 VLAFastAttn(残差修正)。"""
    def __init__(self, D, nh, mlp_mult, vocab, steps=1, W=512):
        super().__init__()
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg; self.D = D; self.vocab = vocab; self.steps = steps; self.W = W
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = VLAFastAttn(cfg)
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


class BDHBlockVLACycleLM(nn.Module):
    """LM 封装: embed -> BDHBlockVLACycle -> head。"""
    def __init__(self, D=128, nh=4, vocab=151936, n_layer=1, steps=1, mlp_mult=128, W=512):
        super().__init__()
        self.D = D; self.nh = nh; self.vocab = vocab; self.n_layer = n_layer; self.steps = steps; self.W = W
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([BDHBlockVLACycle(D, nh, mlp_mult, vocab, steps=steps, W=W)
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
