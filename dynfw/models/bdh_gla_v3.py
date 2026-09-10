"""DynFW 蒸馏版 v7：BDH 稀疏激活 + GLA 线性核（BDH_GLA_v3）。

v3 关键修正（对齐 BDH 真实运作，非 v2 的 h-投影）：
  BDH 的核心 = x_sparse（ReLU 稀疏激活）既是 Q 又是 K 又是 V 的内容（K-is-Q 自相似）。
  v2 错在：GLA 的 q/k/v 从 h（D维隐藏）投影，脱离稀疏激活 → 能力只有 355。

  v3 正确：
    1. x -> encoder -> ReLU -> x_sparse [B,T,N]（超大稀疏激活，表达力来源）
    2. x_sparse 当作 q=k=v（K-is-Q 自相似），reshape 到 [B,nh,T,dk]，N=nh*dk
    3. 用 fla chunk_gla（真 O(T) 线性核）替代 O(T²) 的 QR@QR.mT 因果注意
    4. 保留 通道门控 x_sparse*y_sparse + encoder_v + decoder + LN 残差

  = 守住 BDH 的稀疏+门控表达力，但把 O(T²) 注意换成 O(T) 线性核。
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from fla.ops.gla import chunk_gla


class BDHGLAv3(nn.Module):
    def __init__(self, D=128, nh=4, N=512, vocab=151936, n_layer=2, use_ffn=True):
        super().__init__()
        self.D = D; self.nh = nh; self.N = N; self.vocab = vocab
        self.n_layer = n_layer; self.use_ffn = use_ffn
        assert N % nh == 0, "N 必须能被 nh 整除（N=nh*dk）"
        self.dk = N // nh
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.head = nn.Linear(D, vocab, bias=False)
        # BDH 表达组件（每层）
        self.encs = nn.ModuleList([nn.Linear(D, N, bias=False) for _ in range(n_layer)])
        self.enc_vs = nn.ModuleList([nn.Linear(N, N, bias=False) for _ in range(n_layer)])
        self.decs = nn.ModuleList([nn.Linear(N, D, bias=False) for _ in range(n_layer)])
        # GLA 可学习门控（Mamba 式选择性）：从 x_sparse 得出，log-space
        self.gla_gate = nn.ModuleList([nn.Linear(N, N, bias=False) for _ in range(n_layer)])
        if use_ffn:
            self.ffns = nn.ModuleList([
                nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
                for _ in range(n_layer)])

    def forward(self, x, t=None):
        B, T = x.size(); h = self.e(x)
        for i in range(self.n_layer):
            h = self.gla_bdh_layer(h, self.encs[i], self.enc_vs[i], self.decs[i],
                                   self.gla_gate[i], self.ffns[i] if self.use_ffn else None)
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def gla_bdh_layer(self, h, enc, enc_v, dec, gg, ffn):
        """BDH 稀疏激活 当 Q=K=V，过 GLA 线性核（真 O(T)），保留门控。"""
        B, T, D = h.size(); nh = self.nh; dk = self.dk; N = self.N
        # BDH 稀疏激活（表达力核心）
        lat = enc(h)                       # [B,T,N]
        x_sparse = F.relu(lat)             # 稀疏正激活
        # x_sparse 当 Q=K=V（K-is-Q 自相似），reshape 到 head
        q = x_sparse.view(B, T, nh, dk).transpose(1, 2)   # [B,nh,T,dk]
        k = q                                          # K is Q
        v = q
        # 可学习 gate（Mamba 式，log-space）
        g = torch.sigmoid(gg(x_sparse).view(B, T, nh, dk).transpose(1, 2))
        g = torch.log(g.clamp(min=1e-4, max=1.0))        # log-gate
        # GLA 线性核（真 O(T)）
        res = chunk_gla(q, k, v, g, 1.0)
        yKV = res[0] if isinstance(res, tuple) else res  # [B,nh,T,dk]
        yKV = yKV.transpose(1, 2).reshape(B, T, N)        # [B,T,N]
        # encoder_v 二阶段 + y_sparse
        y_latent = enc_v(yKV)                             # [B,T,N]
        y_sparse = F.relu(y_latent)
        # 通道门控（BDH 核心）
        xy = x_sparse * y_sparse
        out = dec(xy)                                     # [B,T,D]
        h = h + out
        h = F.layer_norm(h, (D,))
        if ffn is not None:
            h = h + ffn(h)
        return h

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
