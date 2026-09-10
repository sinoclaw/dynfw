"""DynFW 蒸馏版 v6：BDH 稀疏记忆 + GLA 线性核混合（BDH_GLA v2）。

方案"BDH + Mamba式线性核"，用户拍板。v2 修正：完整保留 BDH 表达力组件。
v1 错在：只把注意换成 GLA，丢了 BDH 的 稀疏+通道门控+encoder_v，且 gate 用固定0.1 → 能力弱(357)。

v2 正确融合（一次做对）：
  保留 BDH 全套表达组件（ReLU 稀疏激活 x_sparse + encoder_v 二阶段 + 通道门控 x*y + decoder）
  只把 O(T²) 的 K-is-Q 因果注意（attn.forward: QR@QR.mT，scores@V）替换为 GLA 真线性核
  gate 用可学习 per-head 投影（Mamba 式选择性门控），非固定常量

结构：
  x → enc → ReLU → x_sparse [B,T,N]（稀疏激活，表达力）
  → GLA(q,k,v,g) 线性读历史（真 O(T)，替代 O(T²) 注意）
  → encoder_v 二阶段 → y_sparse = ReLU(yKV@enc_v)
  → 通道门控 xy = x_sparse * y_sparse（BDH 核心动态性）
  → decoder → LN 残差 → FFN → head
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from fla.ops.gla import chunk_gla


class BDHGLAv2(nn.Module):
    def __init__(self, D=128, nh=4, dk=64, N=512, vocab=151936, n_layer=2, use_ffn=True):
        super().__init__()
        self.D = D; self.nh = nh; self.dk = dk; self.N = N
        self.vocab = vocab; self.n_layer = n_layer; self.use_ffn = use_ffn
        self.e = nn.Embedding(vocab, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.head = nn.Linear(D, vocab, bias=False)
        # BDH 表达组件（每层）
        self.encs = nn.ModuleList([nn.Linear(D, N, bias=False) for _ in range(n_layer)])
        self.enc_vs = nn.ModuleList([nn.Linear(nh*dk, N, bias=False) for _ in range(n_layer)])
        self.decs = nn.ModuleList([nn.Linear(N, D, bias=False) for _ in range(n_layer)])
        # GLA：q/k/v 投影 + 可学习 gate（Mamba 式选择性）
        self.gla_q = nn.ModuleList([nn.Linear(D, nh*dk, bias=False) for _ in range(n_layer)])
        self.gla_k = nn.ModuleList([nn.Linear(D, nh*dk, bias=False) for _ in range(n_layer)])
        self.gla_v = nn.ModuleList([nn.Linear(D, nh*dk, bias=False) for _ in range(n_layer)])
        self.gla_gate = nn.ModuleList([nn.Linear(D, nh*dk, bias=False) for _ in range(n_layer)])
        if use_ffn:
            self.ffns = nn.ModuleList([
                nn.Sequential(nn.Linear(D, 4*D), nn.GELU(), nn.Linear(4*D, D))
                for _ in range(n_layer)])

    def forward(self, x, t=None):
        B, T = x.size(); h = self.e(x)
        for i in range(self.n_layer):
            h = self.gla_bdh_layer(h, self.encs[i], self.enc_vs[i], self.decs[i],
                                   self.gla_q[i], self.gla_k[i], self.gla_v[i],
                                   self.gla_gate[i], self.ffns[i] if self.use_ffn else None)
        lg = self.head(self.ln(h))
        loss = None if t is None else F.cross_entropy(lg.view(-1, self.vocab), t.view(-1))
        return lg, loss

    def gla_bdh_layer(self, h, enc, enc_v, dec, gq, gk, gv, gg, ffn):
        """BDH 稀疏×GLA 线性：保 BDH 表达力，用 GLA 真线性读历史。"""
        B, T, D = h.size(); nh = self.nh; dk = self.dk
        # BDH 稀疏激活（表达力）
        lat = enc(h)                                   # [B,T,N]
        x_sparse = F.relu(lat)                          # 稀疏正激活
        # GLA 线性读历史（真 O(T)，替代 O(T²) K-is-Q 注意）
        q = gq(h).view(B, T, nh, dk).transpose(1, 2)    # [B,nh,T,dk]
        k = gk(h).view(B, T, nh, dk).transpose(1, 2)
        v = gv(h).view(B, T, nh, dk).transpose(1, 2)
        # 可学习选择性门控（Mamba 式）：log-space gate，sigmoid 后限幅
        g = torch.sigmoid(gg(h).view(B, T, nh, dk).transpose(1, 2))
        g = torch.log(g.clamp(min=1e-4, max=1.0))       # log-space
        res = chunk_gla(q, k, v, g, 1.0)
        yKV = res[0] if isinstance(res, tuple) else res  # [B,nh,T,dk]
        yKV = yKV.transpose(1, 2).reshape(B, T, nh*dk)   # [B,T,nh*dk]
        # encoder_v 二阶段 + y_sparse
        y_latent = enc_v(yKV)                            # [B,T,N]  (N=nh*dk? 见下)
        y_sparse = F.relu(y_latent)
        # 通道门控（BDH 核心）：x_sparse * y_sparse
        xy = x_sparse * y_sparse
        out = dec(xy)                                    # [B,T,D]
        h = h + out
        h = F.layer_norm(h, (D,))
        if ffn is not None:
            h = h + ffn(h)
        return h

    def forward_logits(self, x):
        return self.forward(x, None)[0]

    def np(self):
        return sum(p.numel() for p in self.parameters())
