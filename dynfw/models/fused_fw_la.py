"""DynFW 蒸馏版 v3：FusedFW + BDH 机制融合学生模型（Qwen3 vocab 兼容）。

方案"吸收机制"（2026-09-08）。从 BDH 吸收的钥匙：
  BDH 保顺序的本质不是"累加"（cumsum），而是【内容寻址的因果读取】——
  每个位置 t 用它的当前隐状态（sparse 激活）去【检索】历史每个位置的贡献，
  再因果加权。这是 attention 的本质，既保顺序又能区分"谁重要"。

对比纯 FusedFW 的 rho 读回（einsum 整段求和->无差别累加，丢顺序）：
  - FusedFW       : mem_ctx = einsum('btn,bnd->btd', act, rho)         # 求和，丢顺序
  - FusedFW_rec   : cumsum 累加（方向A）                               # 保顺序但仍无差别累加
  - FusedFW_la    : K-is-Q 内容寻址因果注意（本文件）                  # 保顺序 + 内容寻址

结构（继承 FusedFW 骨架，只换读回算子）：
  嵌入 → enc(D→N) → ReLU/top-k 稀疏激活 → [BDH式 K-is-Q 因果注意读取历史]
     → +FFN(非线性读回) → head(→vocab)
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class CausalContentAttention(nn.Module):
    """BDH 式 K-is-Q 内容寻址因果注意（保顺序 + 内容寻址）。"""
    def __init__(self, nh, D, N):
        super().__init__()
        self.nh = nh
        self.freqs = nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        pc, ps = CausalContentAttention.phases_cos_sin(phases)
        return (v * pc).to(v.dtype) + (v_rot * ps).to(v.dtype)

    def forward(self, Q, V):
        """Q: [B,nh,T,N] 稀疏激活(作为 query&key), V: [B,1,T,D] 值。因果读取历史。"""
        _, _, T, _ = Q.size()
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)          # K=Q
        scores = (QR @ QR.mT).tril(diagonal=-1)   # 因果 [B,nh,T,T]
        # V 广播到 nh 头（V 是 D 维, Q 是 N 维 -> 用投影后的值）
        return scores @ V                     # [B,nh,T,D]


class FusedFWLa(nn.Module):
    """FusedFW 骨架 + BDH 内容寻址因果读取（Qwen3 vocab 兼容）。"""
    def __init__(self, D=128, N=512, k=16, nh=4, vocab=151936, use_ffn=True, n_layer=1):
        super().__init__()
        self.D = D; self.N = N; self.k = k; self.nh = nh; self.vocab = vocab
        self.use_ffn = use_ffn; self.n_layer = n_layer
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, D, bias=False)
        self.head = nn.Linear(D, vocab, bias=False)
        self.encs = nn.ModuleList([nn.Linear(D, N, bias=False) for _ in range(n_layer)])
        self.decs = nn.ModuleList([nn.Linear(N, D, bias=False) for _ in range(n_layer)])
        # BDH 内容寻址因果读取
        self.attns = nn.ModuleList(
            [CausalContentAttention(nh, D, N) for _ in range(n_layer)])
        if use_ffn:
            self.ffn = nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))

    def forward(self, x, t=None):
        B, T = x.size(); h = self.e(x)
        for i in range(self.n_layer):
            h, _, _ = content_rho_forward(self, h, self.encs[i], self.decs[i],
                                          self.out, self.attns[i])
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def forward_hidden(self, x):
        """蒸馏用：返回 head 投影之前的 hidden (B,T,D)，配合分块 KL 避免物化 B×T×V logits。"""
        B, T = x.size(); h = self.e(x)
        for i in range(self.n_layer):
            h, _, _ = content_rho_forward(self, h, self.encs[i], self.decs[i],
                                          self.out, self.attns[i])
            if self.use_ffn:
                h = h + self.ffn(self.ln(h))
        return self.ln(h)

    def head_params(self):
        """返回 (weight(V,D), bias(V,) 或 None)，与 forward 中 head 投影严格一致。"""
        return self.head.weight, None
    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())


def content_rho_forward(self, et, enc, dec, out, attn):
    """单层：稀疏激活 + BDH 内容寻址因果读取历史。返回 h, act, attn_ctx。"""
    B, T, D = et.size(); N = self.N; k = self.k
    lat = enc(et)                                    # [B,T,N]
    topv, topi = torch.topk(lat, k, dim=-1)
    act = torch.relu(torch.zeros_like(lat).scatter(-1, topi, topv))   # [B,T,N] 稀疏
    ln_et = self.ln(et)                              # [B,T,D]
    # BDH 式：sparse 激活作 Q/K，隐状态作 V，因果内容寻址读取
    x_sparse = act.transpose(1, 2).unsqueeze(1)      # [B,1,T,N] -> 广播到 nh? 需 [B,nh,T,N]
    # 把 sparse 激活当作单头（nh=1 复用 attn 频率即可），值 V 用 [B,1,T,D]
    Q = act.unsqueeze(1)                             # [B,1,T,N]
    V = ln_et.unsqueeze(1)                           # [B,1,T,D]
    attn_ctx = attn(Q, V)                            # [B,1,T,D]
    mem_ctx = attn_ctx.squeeze(1)                    # [B,T,D]
    h = et + mem_ctx; h = self.ln(h); h = out(h)
    return h, act, mem_ctx
