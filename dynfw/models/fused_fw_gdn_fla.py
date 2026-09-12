"""C 方案：GDNFastAttnFLA —— v6.6 的块内 raw 注意 + 跨块门控记忆，跨块段换 FLA 内核。

数学依据（本文件的核心事实）：
  v6.6 的 agg_t = Σ_{s∈block, s<t} (q_t·k_s)·v_s  +  q_t @ M_prev
                = 带衰减的全局线性注意 = GLA
  ⟹ 与 fla 的 fused_chunk_simple_gla 算的是同一件事，只差两处：
     ① FLA 的 o 含 self 项 (q_t·k_t)·v_t（S 用"写之后"），我们 diagonal=-1 不含
        ⇒ 精确减掉即可（数值上无损，不是近似）
     ② FLA 的门控是 per-token/head-wise 标量，我们退化成"块级均值"（省算力的妥协）
        ⇒ 用 gate_mode='block' 复现等价（先验证），'token' 是本方案免费解锁的升级

只替换"跨块记忆段"这个算力瓶颈；块内 raw 读出与 RoPE、以及模型其余部分完全不动。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.simple_gla.fused_chunk import fused_chunk_simple_gla
    _FLA_OK = True
except Exception:
    _FLA_OK = False


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi))


class GDNFastAttnFLA(nn.Module):
    """v6.6(GDNFastAttn) 的 FLA 版：跨块记忆段用 fused_chunk_simple_gla 并行化。

    参数与 GDNFastAttn 完全一致（gate: Linear(N,1)），保证权重可互转、可同 seed 对照。
    """

    def __init__(self, config, read_mode='raw', gate_mode='token', chunk=64):
        super().__init__()
        if not _FLA_OK:
            raise RuntimeError('flash-linear-attention 不可用（需要 fla.ops.simple_gla.fused_chunk）')
        self.read_mode = read_mode
        self.gate_mode = gate_mode          # 'block' = 与 v6.6 等价；'token' = 解锁的升级
        self.chunk = chunk                   # FLA 内部分块粒度（建议 64；影响速度不影响语义）
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = nn.Buffer(get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N))
        self.nh = nh; self.N = N; self.D = D
        self.gate = nn.Linear(N, 1)
        nn.init.constant_(self.gate.bias, 4.0)

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = GDNFastAttnFLA.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, K, V, memories=None, W=512):
        """Q,K:[B,nh,T,N] (K is Q); V:[B,1,T,D]; memories:[B,nh,N,D]。
        返回 (out, new_mem)，形状与 GDNFastAttn 完全一致。"""
        assert K is Q
        B, nh, T, _ = Q.size()
        N, D = self.N, self.D

        r = torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype).view(1, 1, -1, 1)
        QR = self.rope(r * self.freqs, Q)          # [B,nh,T,N]（与 v6.6 同）
        KR = QR

        # ---- 门控 g（log 空间，<=0）----
        # ⚠️ 门控语义说明：v6.6 是"块级门控"（块内写入不衰减、块间整体衰减一个标量），
        # GLA 原生是"逐 token 衰减"。但两者可以精确互转 —— 只要令
        #     g_t = 0        （块内非首 token：不衰减，写入原样累加）
        #     g_首 = log α_blk（块首：把整个旧状态衰减 α_blk）
        # 则 S_块末 = α_blk·S_块初 + Σ_{j∈block} k_j⊗v_j，与 v6.6 逐字一致。
        # gate_mode='token' 用 GLA 原生逐 token 门控（v6.7 默认）；
        # gate_mode='block' 用上式精确复现 v6.6，用于单变量归因（不改变数值语义）。
        alpha_logit = self.gate(QR)                                  # [B,nh,T,1]
        if self.gate_mode == 'block':
            nblk = (T + W - 1) // W
            pad = nblk * W - T
            a = alpha_logit
            if pad:
                a = F.pad(a, (0, 0, 0, pad), value=0.0)
            a_blk = F.logsigmoid(a.float()).view(B, nh, nblk, W).mean(dim=3)   # [B,nh,nblk] 块内均值(log域)
            g = torch.zeros(B, nh, T, device=QR.device, dtype=torch.float32)
            # 每块第一个 token 承担整块衰减；块内其余 token 不衰减
            g[:, :, 0::W] = a_blk[:, :, :g[:, :, 0::W].shape[2]]
        else:
            g = F.logsigmoid(alpha_logit.float())                    # 逐 token 门控（GLA 原生）
        g = g.squeeze(-1) if g.dim() == 4 else g                      # [B,nh,T]

        # ---- 组 FLA 所需的 head-second 布局 ----
        # ⚠️ 关键：FLA 的 triton kernel 是为 bf16 写的，fp32 下【反向慢 11×】（实测：
        #    同形状 fp32 fwd+bwd 162.38ms vs bf16 14.70ms；前向仅差 2.24×）。
        #    而我们的对照形态 opt5 本身就是 bf16(strict_bf16=True) ⇒ 让 FLA 也走 bf16
        #    才是真正的同口径。代价是 FLA 段引入 bf16 数值误差，须重跑长 T 验证能力不损。
        q_f = QR.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)                 # [B,T,nh,N]
        k_f = KR.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        v_f = V.expand(-1, nh, -1, -1).permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)  # [B,T,nh,D]
        g_f = g.permute(0, 2, 1).contiguous().to(torch.bfloat16)                     # [B,T,nh]

        st0 = memories
        if st0 is not None:
            st0 = st0.to(torch.bfloat16)

        o, new_mem = fused_chunk_simple_gla(
            q_f, k_f, v_f, g_f,
            initial_state=st0,
            output_final_state=True,
            scale=1.0,                    # 我们的块内 raw 无 1/sqrt(N) 缩放，FLA 默认会乘
        )
        o = o.float()
        new_mem = new_mem.float()

        # ---- 精确减掉 self 项 (q_t·k_t)·v_t，对齐 v6.6 的 diagonal=-1 ----
        self_dot = (QR.float() * KR.float()).sum(dim=-1, keepdim=True)     # [B,nh,T,1]
        self_term = (self_dot * V.float())                                 # [B,nh,T,D]（V 广播到 nh）
        out = o.permute(0, 2, 1, 3).float() - self_term                    # [B,nh,T,D]

        return out.to(Q.dtype), new_mem


class BDHBlockFLA(nn.Module):
    """与 BDHBlockGDNCycle 完全一致，唯一差异 = attn 用 GDNFastAttnFLA。"""

    def __init__(self, D, nh, mlp_mult, vocab, steps=1, W=512, read_mode='raw',
                 gate_mode='token', chunk=64):
        super().__init__()
        from dynfw.models.fused_fw_gdn_cycle import Config
        cfg = Config(1, D, nh, mlp_mult, vocab)
        self.config = cfg; self.D = D; self.vocab = vocab; self.steps = steps; self.W = W
        N = mlp_mult * D // nh
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = GDNFastAttnFLA(cfg, read_mode=read_mode, gate_mode=gate_mode, chunk=chunk)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(0.0)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.apply(self._init_weights)
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


class BDHBlockFLALM(nn.Module):
    """LM 封装：embed -> BDHBlockFLA -> head。接口与 BDHBlockGDNCycleLM 一致。"""

    def __init__(self, D=128, nh=4, vocab=151936, n_layer=1, steps=1, mlp_mult=128, W=512,
                 read_mode='raw', gate_mode='token', chunk=64):
        super().__init__()
        self.D = D; self.nh = nh; self.vocab = vocab; self.n_layer = n_layer
        self.steps = steps; self.W = W
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.blocks = nn.ModuleList([BDHBlockFLA(D, nh, mlp_mult, vocab, steps=steps, W=W,
                                                 read_mode=read_mode, gate_mode=gate_mode, chunk=chunk)
                                     for _ in range(n_layer)])
        self.head = nn.Linear(D, vocab, bias=False)
        self.apply(self._init_weights)
        for b_ in self.blocks:
            nn.init.constant_(b_.attn.gate.bias, 4.0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, x):
        B, T = x.size()
        h = self.e(x).unsqueeze(1)
        h = self.ln(h)
        for blk in self.blocks:
            h, _ = blk(h, None)
        return h.view(B, T, self.D)

    def forward(self, x, targets=None):
        lg = self.head(self.forward_hidden(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(lg.view(-1, self.vocab), targets.view(-1))
        return lg, loss

    def head_params(self):
        return self.head.weight, None

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
